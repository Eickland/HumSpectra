from typing import Dict, Optional
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks

import HumSpectra.spectrum as spm
from HumSpectra.brutto import brutto_gen

def assign_by_tmds(
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
        tmds_spec = spm.assign(tmds_spec,max_num=max_num, brutto_dict=tmds_brutto_dict)
        tmds_spec = spm.calc_mass(tmds_spec)

    #prepare tmds table
    tmds = tmds_spec.sort_values(by='intensity', ascending=False).reset_index(drop=True)
    tmds = tmds.loc[tmds['intensity'] > p].sort_values(by='mass', ascending=True).reset_index(drop=True)
    elem_tmds = spm.find_elements(tmds_spec)

    #prepare spec table
    assign_false = copy.deepcopy(spec.loc[spec['assign'] == False]).reset_index(drop=True)
    assign_true = copy.deepcopy(spec.loc[spec['assign'] == True]).reset_index(drop=True)
    masses = assign_true['mass'].values
    elem_spec = spm.find_elements(spec)
    
    
    #Check that all elements in tmds also in spec
    if len(set(elem_tmds)-set(elem_spec)) > 0:
        raise Exception(f"All elements in tmds spectrum must be in regular spectrum too. But {(set(elem_tmds)-set(elem_spec))} not in spectrum")
    for i in set(elem_spec)-set(elem_tmds):
        tmds[i] = 0

    mass_dif_num = len(tmds)
    min_mass = np.min(masses)

    for i, row_tmds in tqdm(tmds.iterrows(), total=mass_dif_num):

        mass_shift = - row_tmds['calc_mass']
        
        for index, row in assign_false.iterrows():
            if row['assign'] == True:
                continue
                    
            mass = row["mass"] + mass_shift
            if mass < min_mass:
                continue

            idx = np.searchsorted(masses, mass, side='left')
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1
                
            if np.fabs(masses[idx] - mass) / mass * 1e6 <= rel_error:
                assign_false.loc[index,'assign'] = True

                for el in elem_spec:
                    assign_false.loc[index,el] = row_tmds[el] + assign_true.loc[idx,el]

    assign_true = pd.concat([assign_true, assign_false], ignore_index=True).sort_values(by='mass').reset_index(drop=True)
    
    out = spm.calc_mass(out)

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
        spec = spm.filter_by_C13(spec, remove=True)
        spec2 = spm.filter_by_C13(spec2, remove=True)
    else:
        spec = spm.drop_unassigned(spec)
        spec2 = spm.drop_unassigned(spec2)

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
    mass = spm.calc_error(
        spm.drop_unassigned(self)
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

    self = diff_spec

    return self

def assign(
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
        generated_bruttos_table = brutto_gen(brutto_dict, rules=False)
        generated_bruttos_table = generated_bruttos_table.loc[generated_bruttos_table['mass'] > 0]

    res = super().assign(generated_bruttos_table=generated_bruttos_table, abs_error=error, sign='0').drop_unassigned().table

    if max_num is not None and len(res) > max_num:
        res = res.sort_values(by='intensity', ascending=False).reset_index(drop=True)
        res = res.loc[:max_num].reset_index(drop=True)
        res = res.sort_values(by='mass').reset_index(drop=True)
    
    self.table = res

    return self


if __name__ == '__main__':
    pass