import copy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, Optional

from ..calc_process import calc_process as ms_calc
from ..utilits import utilits as ms_utils
from ..raw_data_process import raw_data_process as ms_raw
from ..assign import assign as ms_assign

def assign_by_tmds_optimize(
    spec: pd.DataFrame, 
    tmds_spec: Optional["pd.DataFrame"] = None,
    tmds_brutto_dict: Optional[Dict] = None, 
    rel_error: float = 3,
    p = 0.2,
    max_num: Optional[int] = None,
    C13_filter: bool = True
    ) -> "pd.DataFrame":
    '''
    Assigne brutto formulas by TMDS

    Additianal assignment of masses that can't be done with usual methods

    Parameters
    ----------
    spec: Spectrum object
        Mass spectrum for assign by tmds
    tmds_spec: Tmds object
        Optional. if None generate tmds spectr with default parameters
        Tmds object, include table with most intensity mass difference
    brutto_dict: dict
        Optional. Deafault None.
        Custom Dictonary for generate brutto table.
        Example: {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)}
    abs_error: float
        Optional, default 1 ppm. Relative error for assign peaks by massdif
    p: float
        Optional. Default 0.2. 
        Relative intensity coefficient for treshold tmds spectrum
    max_num: int
        Optional. Max mass diff numbers
    C13_filter: bool
        Use only peaks with C13 isotope peak for generate tmds. Default True.

    Return
    ------
    Spectrum
        Assigned by tmds masses

    Reference
    ---------
    Kunenkov, Erast V., et al. "Total mass difference 
    statistics algorithm: a new approach to identification 
    of high-mass building blocks in electrospray ionization 
    Fourier transform ion cyclotron mass spectrometry data 
    of natural organic matter." 
    Analytical chemistry 81.24 (2009): 10106-10115.
    '''
    
    if "assign" not in spec:
        raise Exception("Spectrum is not assigned")

    spec = spec.copy()

    #calculstae tmds table
    if tmds_spec is None:
        tmds_spec = calc(spec, p=p, C13_filter=C13_filter) #by varifiy p-value we can choose how much mass-diff we will take
        tmds_spec = assign_tmds(tmds_spec,max_num=max_num, brutto_dict=tmds_brutto_dict)
        tmds_spec = ms_calc.calc_mass(tmds_spec)

    #prepare tmds table
    tmds = tmds_spec.sort_values(by='intensity', ascending=False).reset_index(drop=True)
    tmds = tmds.loc[tmds['intensity'] > p].sort_values(by='mass', ascending=True).reset_index(drop=True)
    elem_tmds = ms_utils.find_elements(tmds_spec)

    #prepare spec table
    assign_false = copy.deepcopy(spec.loc[spec['assign'] == False]).reset_index(drop=True)
    assign_true = copy.deepcopy(spec.loc[spec['assign'] == True]).reset_index(drop=True)
    elem_spec = ms_utils.find_elements(spec)
    # Извлекаем данные в чистые массивы NumPy для скорости
    false_masses = assign_false['mass'].values
    true_masses = assign_true['mass'].values
    
    # Элементы, которые нужно обновить (C, H, O, N и т.д.)
    elem_list = list(elem_spec)
    # Матрица составов для уже назначенных пиков
    true_elements = assign_true[elem_list].values 
    # Буфер для новых составов неназначенных пиков
    false_elements = np.zeros((len(assign_false), len(elem_list)))
    assigned_mask = np.zeros(len(assign_false), dtype=bool)

    for _, row_tmds in tmds.iterrows():
        mass_shift = row_tmds['calc_mass']
        # Целевая масса, которую мы ищем среди уже назначенных
        target_masses = false_masses - mass_shift
        
        # Находим ближайшие индексы в true_masses для всех false_masses разом
        idx = np.searchsorted(true_masses, target_masses) # type: ignore
        idx = np.clip(idx, 0, len(true_masses) - 1)
        
        # Корректировка индексов (поиск ближайшего из двух соседей)
        # Проверяем также соседа слева (idx-1)
        idx_left = np.clip(idx - 1, 0, len(true_masses) - 1)
        dist_right = np.abs(true_masses[idx] - target_masses)
        dist_left = np.abs(true_masses[idx_left] - target_masses)
        closer_idx = np.where(dist_left < dist_right, idx_left, idx)
        
        # Вычисляем относительную ошибку в ppm для всех пар
        errors = np.abs(true_masses[closer_idx] - target_masses) / target_masses * 1e6
        
        # Маска тех, кто попал в допуск и еще не был назначен
        hit_mask = (errors <= rel_error) & (~assigned_mask)
        
        if hit_mask.any():
            assigned_mask |= hit_mask
            # Векторное сложение составов: состав найденного + состав разности
            tmds_comp = row_tmds[elem_list].to_numpy()
            false_elements[hit_mask] = true_elements[closer_idx[hit_mask]] + tmds_comp

    # Обновляем DataFrame один раз в конце
    assign_false['assign'] = assigned_mask
    assign_false[elem_list] = false_elements

    assign_true = pd.concat([assign_true, assign_false], ignore_index=True).sort_values(by='mass').reset_index(drop=True)
    
    out = ms_calc.calc_mass(assign_true)

    out=out[out['calc_mass'].isnull() | ~out[out['calc_mass'].notnull()].duplicated(subset='calc_mass',keep='first')] 
    spec = out.sort_values(by='mass').reset_index(drop=True)

    
    return spec

def calc(
    self: pd.DataFrame,
    other: Optional["pd.DataFrame"] = None,
    p: float = 0.2,
    wide: int = 10,
    C13_filter:bool = True,
    ) -> "pd.DataFrame":
    """
    Total mass difference statistic calculation 

    Parameters
    ----------
    other: Spectrum object
        Optional. If None, TMDS will call by self.
    p: float
        Minimum relative intensity for taking mass-difference. Default 0.2.
    wide: int
        Minimum interval in 0.001*wide Da of peaks finding. Default 10.
    C13_filter: bool
        Use only peaks that have C13 isotope peak. Default True

    Return
    ------
    pd.DataFrame
    """

    spec = self.copy(deep=True)

    if other is None:
        spec2 = self.copy(deep=True)
    else:
        spec2 = other.copy(deep=True)

    if C13_filter:
        spec = ms_raw.filter_by_C13(spec, remove=True)
        spec2 = ms_raw.filter_by_C13(spec2, remove=True)
    else:
        spec = ms_utils.drop_unassigned(spec)
        spec2 = ms_utils.drop_unassigned(spec2)

    masses = spec['mass'].values
    masses2 = spec2['mass'].values

    mass_num = len(masses)
    mass_num2 = len(masses2)

    if mass_num <2 or mass_num2 < 2:
        raise Exception(f"Too low number of assigned peaks")

    mdiff = np.zeros((mass_num, mass_num2), dtype=float)
    for x in range(mass_num):
        for y in range(x, mass_num2):
            dif = np.fabs(masses[x]-masses2[y])
            if dif < 300:
                mdiff[x,y] = dif

    mdiff = np.round(mdiff, 3)
    unique, counts = np.unique(mdiff, return_counts=True)
    counts[0] = 0

    tmds_spec = pd.DataFrame()
    tmds_spec.attrs['name'] = spec.attrs['name']
    tmds_spec['mass'] = unique
    tmds_spec['count'] = counts
    tmds_spec['intensity'] = tmds_spec['count']/mass_num
    tmds_spec = tmds_spec.sort_values(by='mass').reset_index(drop=True)

    value_zero = set([i/1000 for i in range (0, 300000)]) - set (unique)
    unique = np.append(unique, np.array(list(value_zero)))
    counts = np.append(counts, np.zeros(len(value_zero), dtype=float))

    peaks, properties = find_peaks(tmds_spec['intensity'], distance=wide, prominence=p/2)
    prob = []
    for peak in peaks:
        prob.append(tmds_spec.loc[peak-5:peak+5,'intensity'].sum())
    tmds_spec = tmds_spec.loc[peaks].reset_index(drop=True)
    tmds_spec['intensity'] = prob
    tmds_spec = tmds_spec.loc[tmds_spec['intensity'] > p]

    if len(tmds_spec) < 0:
        raise Exception(f"There isn't mass diff mass, decrease p-value")

    self = tmds_spec

    return self

def calc_by_brutto(self) -> "pd.DataFrame":

    """
    Calculate self difference by calculated mass from brutto

    Return
    ------
    pd.DataFrame
    """
    name = self.attrs['name']
    mass = ms_calc.calc_error(
        ms_utils.drop_unassigned(self)
    )['calc_mass'].values

    massl = len(mass)
    mdiff = np.zeros((massl, massl), dtype=float)
    for x in range(massl):
        for y in range(x, massl):
            mdiff[x,y] = np.fabs(mass[x]-mass[y])

    mdiff = np.round(mdiff, 6)
    unique, counts = np.unique(mdiff, return_counts=True)
    counts[0] = 0

    diff_spec = pd.DataFrame()
    diff_spec['mass'] = unique
    diff_spec['count'] = counts
    diff_spec['intensity'] = diff_spec['count']/massl
    diff_spec = diff_spec.sort_values(by='mass').reset_index(drop=True)

    diff_spec.attrs['name'] = name
    
    self = diff_spec

    return self

def assign_tmds(
    self,
    generated_bruttos_table: Optional[pd.DataFrame] = None,
    error: float = 0.001,
    brutto_dict: Optional[dict] = None,
    max_num: Optional[int] = None
    ) -> "pd.DataFrame":

    """
    Finding the nearest mass in generated_bruttos_table

    Parameters
    ----------
    generated_bruttos_table: pandas DataFrame 
        Optional. with column 'mass' and elements, should be sorted by 'mass'
    error: float
        Optional. Default 0.001. 
        absolute error iin Da for assign formulas
    brutto_dict: dict
        Optional, default {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)}
        generate brutto table if generated_bruttos_table is None.
    max_num: int
        Optional. Default 100
    
    Return
    ------
    pd.DataFrame
    """

    if brutto_dict is None:
        brutto_dict = {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)}

    if generated_bruttos_table is None:
        generated_bruttos_table = brutto_gen(brutto_dict, rules=False) # type: ignore
        generated_bruttos_table = generated_bruttos_table.loc[generated_bruttos_table['mass'] > 0] # type: ignore

    res = ms_utils.drop_unassigned(ms_assign.assign_optimized(self,generated_bruttos_table=generated_bruttos_table, abs_error=error, sign='0'))

    if max_num is not None and len(res) > max_num:
        res = res.sort_values(by='intensity', ascending=False).reset_index(drop=True)
        res = res.loc[:max_num].reset_index(drop=True)
        res = res.sort_values(by='mass').reset_index(drop=True)
    

    return res
