import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzMLFile
from typing import Optional

from ..decorators import decorators as ms_dec

def extract_mass_list_percentile(mzml_file, ms_level=1, rt_range=None, 
                               low_percentile:float=99,
                               high_percentile:float=99.9):
    """
    Порог как процентиль от распределения интенсивностей
    """
    exp = MSExperiment()
    MzMLFile().load(mzml_file, exp)
    
    # Сбор всех интенсивностей
    all_intensities = []
    for spectrum in exp:
        if spectrum.getMSLevel() == ms_level:
            if rt_range and not (rt_range[0] <= spectrum.getRT() <= rt_range[1]):
                continue
            mz, intensities = spectrum.get_peaks()
            all_intensities.extend(intensities)
    
    if not all_intensities:
        return pd.DataFrame()
    
    # Порог по процентилю
    low_percentile_threshold = np.percentile(all_intensities, low_percentile)
    high_percentile_threshold = np.percentile(all_intensities, high_percentile)
    
    print(f"{low_percentile}-й процентиль интенсивности: {low_percentile_threshold:.2f}")
    print(f"{high_percentile}-й процентиль интенсивности: {high_percentile_threshold:.2f}")
    print(f"Всего точек до фильтрации: {len(all_intensities)}")
    
    # Извлечение с порогом по процентилю
    all_mz = []
    all_intensities_filtered = []
    
    for spectrum in exp:
        if spectrum.getMSLevel() != ms_level:
            continue
            
        rt = spectrum.getRT()
        if rt_range and not (rt_range[0] <= rt <= rt_range[1]):
            continue
            
        mz, intensities = spectrum.get_peaks()
        
        mask = (intensities >= low_percentile_threshold)
        filtered_mz = mz[mask]
        filtered_intensities = intensities[mask]
        
        all_mz.extend(filtered_mz)
        all_intensities_filtered.extend(filtered_intensities)
    
    print(f"Точек после фильтрации: {len(all_mz)}")
    
    return pd.DataFrame({
        'mass': all_mz,
        'intensity': all_intensities_filtered,
    })

@ms_dec._copy
def noise_filter(self,
                    force: float = 1.5,
                    intensity: Optional[float] = None,
                    quantile: Optional[float] = None 
                    ) -> pd.DataFrame:
    """
    Remove noise from spectrum

    Parameters
    ----------
    intensity: float
        Cut by min intensity. 
        Default None and dont apply.
    quantile: float
        Cut by quantile. For example 0.1 mean that 10% 
        of peaks with minimal intensity will be cutted. 
        Default None and dont aplly
    force: float
        How many peaks should cut when auto-search noise level.
        Default 1.5 means that peaks with intensity more 
        than noise level*1.5 will be cutted
    
    Return
    ------
    pd.DataFrame

    Caution
    -------
    There is risk of loosing data. Do it cautiously.
    Level of noise may be determenided wrong. 
    Draw and watch spectrum.
    """
    
    if intensity is not None:
        self = self.loc[self['intensity'] > intensity].reset_index(drop=True)
        self.attrs['noise filter (intensity)'] = intensity
    
    elif quantile is not None:
        tresh = self['intensity'].quantile(quantile)
        self = self.loc[self['intensity'] > tresh].reset_index(drop=True)
        self.attrs['noise filter (quantile)'] = quantile
        
    
    else:

        intens = self['intensity'].values
        cut_diapasone=np.linspace(0, np.mean(intens),100)

        d = []
        for i in cut_diapasone:
            d.append(len(intens[intens > i]))

        dx = np.gradient(d, 1)
        tresh = np.where(dx==np.min(dx))
        cut = cut_diapasone[tresh[0][0]] * force
        self = self.loc[self['intensity'] > cut].reset_index(drop=True)

        self.attrs['noise filter (force)'] = force

    return self

@ms_dec._copy
def filter_by_C13(
    self, 
    rel_error: float = 0.5,
    remove: bool = False,
) -> pd.DataFrame:
    """ 
    Check if peaks have the same brutto with C13 isotope

    Parameters
    ----------
    rel_error: float
        Optional. Default 0.5.
        Allowable ppm error when checking c13 isotope peak
    remove: bool
        Optional, default False. 
        Drop unassigned peaks and peaks without C13 isotope
    
    Return
    ------
    pd.DataFrame
    """
    
    self = self.sort_values(by='mass').reset_index(drop=True)
    
    flags = np.zeros(self.shape[0], dtype=bool)
    masses = self["mass"].values
    
    C13_C12 = 1.003355  # C13 - C12 mass difference

    
    for index, row in self.iterrows():
        mass = row["mass"] + C13_C12
        error = mass * rel_error * 0.000001

        idx = np.searchsorted(masses, mass, side='left')
        
        if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
            idx -= 1
        
        if np.fabs(masses[idx] - mass)  <= error:
            flags[index] = True
    
    self['C13_peak'] = flags

    if remove:
        self = self.loc[(self['C13_peak'] == True) & (self['assign'] == True)].reset_index(drop=True)
        self.attrs['filter_C13'] = True
        

    return self

