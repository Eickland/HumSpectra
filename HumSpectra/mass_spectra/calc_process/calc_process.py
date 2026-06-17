import copy
import pandas as pd
import numpy as np
import scipy.stats as st
from typing import Optional, Tuple, Sequence
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from ..utilits import utilits as ms_utils
from ..brutto import brutto as ms_brutto
from ..decorators import decorators as ms_dec

from scipy.signal import savgol_filter

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

    dif_masses = ms_brutto.gen_from_brutto(df)['calc_mass'].to_numpy()
    dif = np.unique([dif_masses*i for i in range(1,10)])

    data = copy.deepcopy(spec)
    masses = data['mass'].to_numpy()

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
    kde_err['ppm'] = savgol_filter(kde_err['ppm'].to_numpy(), 31, 3) # type: ignore

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

@ms_dec._copy
def normalize(self, how:str='sum') -> pd.DataFrame:
    """
    Intensity normalize by intensity

    Parameters
    ----------
    how: {'sum', 'max', 'median', 'mean'}
        'sum' for normilize by sum of intensity of all peaks. (default)
        'max' for normilize by higher intensity peak.
        'median' for normilize by median of peaks intensity.
        'mean' for normilize by mean of peaks intensity.

    Return
    ------
    pd.DataFrame
    """

    if how=='max':
        self['intensity'] /= self['intensity'].max()
    elif how=='sum':
        self['intensity'] /= self['intensity'].sum()
    elif how=='median':
        self['intensity'] /= self['intensity'].median()
    elif how=='mean':
        self['intensity'] /= self['intensity'].mean()
    else:
        raise Exception(f"There is no such mode: {how}")
    
    self.attrs['normilize'] = how

    return self


@ms_dec._copy
def _calc_sign(self) -> str:
    """
    Determine sign from mass and calculated mass

    '-' for negative mode
    '+' for positive mode
    '0' for neutral

    Return
    ------
    str            
    """

    self = ms_utils.drop_unassigned(self)

    if "calc_mass" not in self:
        self = ms_utils.calc_mass(self)

    if "charge" not in self.columns:
        self["charge"] = 1

    value = (self["calc_mass"]/self["charge"] - self["mass"]).mean()
    value = np.round(value,4)
    if value > 1:
        return '-'
    elif value > 0.0004 and value < 0.01:
        return '+'
    else:
        return '0'

@ms_dec._copy
def calc_error(self, sign: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate relative and absolute error of assigned peaks from measured and calculated masses

    Add columns "abs_error" and "rel_error" to self

    Parameters
    ----------
    sign: {'-', '+', '0'}
        Optional. Default None and get from metatdata or calculated by self. 
        Mode in which mass spectrum was gotten. 
        '-' for negative mode
        '+' for positive mode
        '0' for neutral
    
    Return
    ------
    Spectrum
    """

    if "calc_mass" not in self:
        self = ms_utils.calc_mass(self)

    if "charge" not in self.columns:
        self["charge"] = 1

    if sign is None:
        if 'sign' in self.attrs:
            sign = self.attrs['sign']
        else:
            sign = _calc_sign(self)

    if sign == '-':
        self["abs_error"] = ((self["mass"] + (- 0.00054858 + 1.007825)) * self["charge"]) - self["calc_mass"] #-electron + proton
    elif sign == '+':
        self["abs_error"] = ((self["mass"] + 0.00054858) * self["charge"]) - self["calc_mass"]#+electron
    elif sign == '0':
        self["abs_error"] = (self["mass"] * self["charge"]) - self["calc_mass"]
    else:
        raise ValueError('Sended sign or sign in attrs is not correct. May be "+","-","0"')
    
    self["rel_error"] = self["abs_error"] / self["mass"] * 1e6
    
    return self

