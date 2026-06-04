import copy
import pandas as pd
import numpy as np
from typing import Optional

import mass_spectra.utilits.utilits as ms_utils
import mass_spectra.assign.assign as ms_assign
import mass_spectra.calc_process.calc_process as ms_calc

def recallibrate_optimize(spec: pd.DataFrame, 
                error_table: Optional[pd.DataFrame] = None, 
                how: str = 'assign',
                mode: str = '-',
                draw: bool = True) -> pd.DataFrame:
    '''
    Recallibrate spectrum

    Fine reccalabration based on self (assign or mass differnce map)
    or with external ethalon spectrum

    Parameters
    ----------
    spec: pd.DataFrame object
        Mass spectrum for recallibration
    error_table: ErrorTable object
        Optional. If None - calculate for spec. 
        ErrorTable object contain table error in ppm for mass, default 100 string            
    how: {'assign', 'mdm', filename} 
        Optional. Default 'assign'.
        If error_table is None we can choose how to recalculate.
        'assign' - by assign error, default.
        'mdm' - by calculation mass-difference map.
        filename - path to etalon spectrum, treated and saved by nomspectra
    mode: str
        Optional. Default '-' negative mode. May be +, - or 0
    draw: bool
        Plot error (fit of KDM)

    Returns
    -------
    pd.DataFrame
    '''

    spec = spec.copy()

    if error_table is None:
        if how == 'assign':
            error_table = assign_error(spec, show_map=draw, mode=mode)
        elif how == 'mdm':
            error_table = massdiff_error(spec, show_map=draw)
        else:
            etalon = ms_utils.read_mass_list(how)
            error_table = etalon_error(spec=spec, etalon=etalon, show_map=draw)
        pass

    # Извлекаем массивы для быстрой работы
    spec_masses = spec['mass'].values
    err_masses = error_table['mass'].values
    err_ppm = error_table['ppm'].values

    # 1. Вместо ступенчатых циклов используем линейную интерполяцию ошибки.
    # Это найдет точное значение ppm для каждой массы из spec.
    # По умолчанию np.interp экстраполирует крайние значения.
    interp_ppm = np.interp(spec_masses, err_masses, err_ppm)

    # 2. Векторизованный пересчет всей колонки масс за одну операцию
    # Формула: mass = mass + (mass * ppm / 1,000,000)
    spec['mass'] = spec_masses * (1 + interp_ppm / 1e6)

    spec.attrs['recallibrate'] = how
    return spec

@staticmethod
def assign_error(
    spec: pd.DataFrame,
    ppm: float = 3,
    brutto_dict = {'C':(4,30), 'H':(4,60), 'O':(0,20)},
    mode = '-',
    show_map: bool = True):
    '''
    Recallibrate by assign error

    Parameters
    ----------
    spec: pd.DataFrame object
        Initial mass spectrum for recallibrate
    ppm: float
        Permissible relative error in callibrate error. Default 3.
    brutto_dict: dict
        Dictonary with elements ranges for assignment
    mode: str
        Optional. Default '-' negative mode. May be +, - or 0
    show_error: bool
        Optional. Default True. Show process 

    Return
    ------
    ErrorTable
    '''

    spectr = copy.deepcopy(spec)
    spectr = ms_assign.assign_optimized(spectr,rel_error=ppm, brutto_dict=brutto_dict, sign=mode)
    spectr = ms_calc.calc_error(ms_calc.calc_mass(spectr))

    error_table = spectr
    error_table = error_table.loc[:,['mass','rel_error']]
    error_table.columns = ['mass', 'ppm']
    error_table['ppm'] = - error_table['ppm']
    error_table = error_table.dropna()

    kdm = ms_calc.kernel_density_map(df_error = error_table)
    err = ms_calc.fit_kernel(f=kdm, 
                        show_map=show_map, 
                        mass=ms_utils.drop_unassigned(spectr)['mass'].to_numpy())
    err = ms_calc.extrapolate(err,(spec['mass'].min(), spec['mass'].max()))

    return err

@staticmethod
def massdiff_error( spec: pd.DataFrame,
                    show_map:bool = True):
    '''
    Self-recallibration of mass-spectra by mass-difference map

    Parameters
    -----------
    spec: pd.DataFrame object
        Initial mass spectrum for recallibrate
    show_error: bool
        Optional. Default True. Show process 

    Return
    -------
    ErrorTable

    References
    ----------
    Smirnov, K. S., Forcisi, S., Moritz, F., Lucio, M., & Schmitt-Kopplin, P. 
    (2019). Mass difference maps and their application for the 
    recalibration of mass spectrometric data in nontargeted metabolomics. 
    Analytical chemistry, 91(5), 3350-3358. 
    '''

    spec = copy.deepcopy(spec)
    mde = ms_calc.md_error_map(spec = spec)
    kdm = ms_calc.kernel_density_map(df_error=mde)
    err = ms_calc.fit_kernel(f=kdm, show_map=show_map, mass=spec['mass'].to_numpy())
    
    err['ppm'] = err['ppm'] - err.loc[0, 'ppm'] # type: ignore

    return err

@staticmethod
def etalon_error(spec: pd.DataFrame,
                etalon: pd.DataFrame,
                quart: float = 0.9,
                ppm: float = 3,
                show_map: bool = True): 
    '''
    Recallibrate by etalon

    Parameters
    ----------
    spec: pd.DataFrame object
        Initial mass spectrum for recallibrate
    etalon: pd.DataFrame object
        Etalon mass spectrum
    quart: float
        Optionaly. by default it is 0.9. 
        Usualy it is enough for good callibration
        Quartile, which will be taken for calc recallibrate error
    ppm: float
        Optionaly. Default 3.
        permissible relative error in ppm for seak peak in etalon
    show_map: bool
        Optional. Default True. Show process 

    Return
    ------
    ErrorTable
    '''

    et = etalon['mass'].to_numpy()  # Используем numpy array вместо list
    df = spec.copy()  # Простое копирование вместо deepcopy

    # Фильтрация по квантилю
    treshold = df['intensity'].quantile(quart)
    df_filtered = df[df['intensity'] > treshold].copy()
    
    # Векторизованное вычисление границ
    min_masses = df_filtered['mass'].to_numpy() * (1 - ppm / 1000000)
    max_masses = df_filtered['mass'].to_numpy() * (1 + ppm / 1000000)
    
    # Преобразуем etalon в отсортированный numpy array для бинарного поиска
    et_sorted = np.sort(et)
    
    # Функция для поиска ближайшего эталонного значения в пределах ppm
    def find_closest_etalon(mass, min_mass, max_mass):
        # Бинарный поиск в отсортированном массиве
        idx = np.searchsorted(et_sorted, min_mass)
        if idx < len(et_sorted) and et_sorted[idx] <= max_mass:
            return et_sorted[idx]
        return 0
    
    # Векторизованное применение функции
    df_filtered['cal'] = np.vectorize(find_closest_etalon, otypes=[float])(
        df_filtered['mass'].values, min_masses, max_masses
    )
    
    # Альтернативный вариант с использованием broadcasting (быстрее для больших данных)
    # df_filtered['cal'] = 0
    # for i, (min_m, max_m) in enumerate(zip(min_masses, max_masses)):
    #     mask = (et_sorted >= min_m) & (et_sorted <= max_m)
    #     if np.any(mask):
    #         df_filtered.iloc[i, df_filtered.columns.get_loc('cal')] = et_sorted[mask][0]

    # Фильтрация и вычисления
    df_calibrated = df_filtered[df_filtered['cal'] > 0].copy()
    df_calibrated['dif'] = df_calibrated['cal'] - df_calibrated['mass']
    df_calibrated['ppm'] = df_calibrated['dif'] / df_calibrated['mass'] * 1000000
    
    error_table = df_calibrated[['mass', 'ppm']].dropna()
    
    kdm = ms_calc.kernel_density_map(df_error=error_table)
    err = ms_calc.fit_kernel(f=kdm, show_map=show_map, mass=spec['mass'].to_numpy())

    return err
