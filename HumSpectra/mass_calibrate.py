
from typing import Optional, Tuple
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.stats as st
from scipy.interpolate import CubicSpline

from HumSpectra.brutto import gen_from_brutto
import HumSpectra.spectrum as Hspm

def recallibrate(spec: pd.DataFrame, 
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
            etalon = Hspm.read_mass_list(how)
            error_table = etalon_error(spec=spec, etalon=etalon, show_map=draw)

    err = error_table.copy(deep=True)
    spec = spec.reset_index(drop=True)
    wide = len(err)

    min_mass = err['mass'].min()
    max_mass = err['mass'].max()
    a = np.linspace(min_mass, max_mass, wide+1)

    for i in range(wide):
        for ind in spec.loc[(spec['mass']>a[i]) & (spec['mass']<=a[i+1])].index:
            mass = spec.loc[ind, 'mass']
            e = mass * err.loc[i, 'ppm'] / 1000000
            spec.loc[ind, 'mass'] = spec.loc[ind, 'mass'] + e

    spec.attrs['recallibrate'] = how

    return spec



@staticmethod
def md_error_map(
    spec: pd.DataFrame, 
    ppm: float = 3
    ) -> pd.DataFrame:
    '''
    Calculate mass differnce map

    Parameters
    ----------
    spec: pd.DataFrame
        data
    ppm: float
        Optional. Default 3.
        Permissible error in ppm
    show_map: bool
        Optional. Default False.
        Show error in ppm versus mass

    Return
    ------
    Pandas Dataframe
    '''

    #common neutral mass loses: CH2, CO, CH2O, C2HO, H2O, CO2
    df = pd.DataFrame({ 'C':[1,1,1,2,0,1],
                        'H':[2,0,2,1,2,0],
                        'O':[0,1,1,1,1,2]})

    dif_masses = gen_from_brutto(df)['calc_mass'].values.astype('float')
    dif = np.unique([dif_masses*i for i in range(1,10)])

    data = copy.deepcopy(spec)
    masses = data['mass'].values.astype('float')

    data = data.sort_values(by='intensity', ascending=False).reset_index(drop=True)
    if len(data) > 1000:
        data = data[:1000]
    data = data.sort_values(by='mass').reset_index(drop=True)

    data_error = [] #array for new data

    for index, row in data.iterrows(): #take every mass in list
        
        mass = row["mass"]

        for i in dif:
            mz = mass + i #massdif

            idx = np.searchsorted(masses, mz, side='left')                
            if idx > 0 and (idx == len(masses) or np.fabs(mz - masses[idx - 1]) < np.fabs(mz - masses[idx])):
                idx -= 1

            if np.fabs(masses[idx] - mz) / mz * 1e6 <= ppm:
                data_error.append([mass, (masses[idx] - mz)/mz*1000000])
    
    df_error = pd.DataFrame(data = data_error, columns=['mass', 'ppm' ])

    return df_error

@staticmethod
def fit_kernel(
    f: np.ndarray,
    mass: np.ndarray,
    err_ppm: float = 3,
    show_map: bool = True) -> pd.DataFrame:
    '''
    Fit max intesity of kernel density map

    Parameters
    ----------
    f: np.array
        keerndel density map in numpy array 100*100
    show_map: bool
        Optional. Default true.
        Plot how fit kernel

    Return
    ------
    Pandas Dataframe
    '''

    df = pd.DataFrame(f, index=np.linspace(err_ppm,-err_ppm,100))

    out = []
    for i in df.columns:
        max_kernel = df[i].quantile(q=0.95)
        ppm = df.loc[df[i] > max_kernel].index.values
        out.append([i, np.mean(ppm)])
    kde_err = pd.DataFrame(data=out, columns=['i','ppm'])
    
    #smooth data
    kde_err['ppm'] = savgol_filter(kde_err['ppm'], 31,3)
    
    xmin = min(mass)
    xmax = max(mass)
    
    kde_err['mass'] = np.linspace(xmin, xmax, 100)

    ymin = -err_ppm
    ymax = err_ppm

    if show_map:
        fig = plt.figure(figsize=(4,4), dpi=75)
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.imshow(df, extent=(xmin, xmax, ymin, ymax), aspect='auto')
        ax.plot(kde_err['mass'], kde_err['ppm'], c='r')
        ax.set_xlabel('m/z, Da')
        ax.set_ylabel('error, ppm')      

    return kde_err

@staticmethod
def kernel_density_map(
    df_error: pd.DataFrame, 
    ppm: float = 3, 
    ) -> np.ndarray:
    '''
    Calculate and plot kernel density map 100*100 for data

    Parameters
    ----------
    df_error: pd.Dataframe
        error_table for generate kerle density map
    ppm: float
        Optional. Default 3.
        treshould for generate
    show_map: bool
        Optional. Default False. plot kde

    Return
    ------
    numpy array
    '''
    
    x = np.array(df_error['mass'])
    y = np.array(df_error['ppm'])

    xmin = min(x) 
    xmax = max(x) 

    ymin = -ppm 
    ymax = ppm 

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    kdm = np.reshape(kernel(positions).T, xx.shape)
    kdm = np.rot90(kdm)
    
    return kdm

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
    spectr = spectr.assign(rel_error=ppm, brutto_dict=brutto_dict, sign=mode)
    spectr = Hspm.calc_error(Hspm.calc_mass(spectr))

    error_table = spectr
    error_table = error_table.loc[:,['mass','rel_error']]
    error_table.columns = ['mass', 'ppm']
    error_table['ppm'] = - error_table['ppm']
    error_table = error_table.dropna()

    kdm = kernel_density_map(df_error = error_table)
    err = fit_kernel(f=kdm, 
                        show_map=show_map, 
                        mass=Hspm.drop_unassigned(spectr)['mass'].values)
    err = extrapolate(err,(spec['mass'].min(), spec['mass'].max()))

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
    mde = md_error_map(spec = spec)
    kdm = kernel_density_map(df_error=mde)
    err = fit_kernel(f=kdm, show_map=show_map, mass=spec['mass'].values)
    
    err['ppm'] = err['ppm'] - err.loc[0, 'ppm']

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

    et = etalon['mass'].values.astype('float')  # Используем numpy array вместо list
    df = spec.copy()  # Простое копирование вместо deepcopy

    # Фильтрация по квантилю
    treshold = df['intensity'].quantile(quart)
    df_filtered = df[df['intensity'] > treshold].copy()
    
    # Векторизованное вычисление границ
    min_masses = df_filtered['mass'].values.astype('float') * (1 - ppm / 1000000)
    max_masses = df_filtered['mass'].values.astype('float') * (1 + ppm / 1000000)
    
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
    
    kdm = kernel_density_map(df_error=error_table)
    err = fit_kernel(f=kdm, show_map=show_map, mass=spec['mass'].values)

    return err

def extrapolate(self, ranges:Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """
    Extrapolate error data

    Parameters
    ----------
    ranges: Tuple(numeric, numeric)
        Optionaly. Default None - all width of mass in error table.
        For which diaposone of mass extrapolate data

    Return
    ------
    ErrorTable
    """
    
    if ranges is None:
        ranges = (self['mass'].min(), self['mass'].max())

    interpolation_range = np.linspace(ranges[0], ranges[1], 100)
    linear_interp = CubicSpline(self['mass'], self['ppm'], extrapolate=True)
    linear_results = linear_interp(interpolation_range)
    err = pd.DataFrame()
    err ['mass'] = interpolation_range
    err ['ppm'] = linear_results

    return err

def show_error(self) -> None:
    """
    Plot error map from ErrorTable class data
    """

    fig, ax = plt.subplots(figsize=(4,4), dpi=75)
    ax.plot(self['mass'], self['ppm'])
    ax.set_xlabel('m/z, Da')
    ax.set_ylabel('error, ppm')

if __name__ == '__main__':
    pass