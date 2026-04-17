from typing import Callable, Dict, Sequence, Union, Optional, Mapping, Tuple, Any, List
from functools import wraps, lru_cache
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib.axes import Axes

import HumSpectra.mass_spectra as ms
import HumSpectra.utilits as ut
import HumSpectra.mass_visualizer as mv

def _copy(func):
    """
    Decorator for deep copy pd.DataFrame before apllying methods
    
    Parameters
    ----------
    func: method
        function for decoarate
    
    Return
    ------
    function with deepcopyed pd.DataFrame
    """

    @wraps(func)
    def wrapper(dataframe, *args, **kwargs):
        # Создаем глубокую копию DataFrame
        dataframe_copy = copy.deepcopy(dataframe)
        
        # Вызываем оригинальную функцию с копией
        result = func(dataframe_copy, *args, **kwargs)
        
        return result
    
    return wrapper

@_copy
def cram(self) -> pd.DataFrame:
    """
    Mark rows that fit CRAM conditions
    (carboxylic-rich alicyclic molecules)

    Add column "CRAM" to self

    Return
    ------
    Spectrum

    References
    ----------
    Hertkorn, N. et al. Characterization of a major 
    refractory component of marine dissolved organic matter.
    Geochimica et. Cosmochimica Acta 70, 2990-3010 (2006)
    """

    if "DBE" not in self:
        self = dbe(self)        

    def check(row):
        if row['DBE']/row['C'] < 0.3 or row['DBE']/row['C'] > 0.68:
            return False
        if row['DBE']/row['H'] < 0.2 or row['DBE']/row['H'] > 0.95:
            return False
        if row['O'] == 0:
            return False
        elif row['DBE']/row['O'] < 0.77 or row['DBE']/row['O'] > 1.75:
            return False
        return True

    table = ms.merge_isotopes(self.copy(deep=True))
    self['CRAM'] = table.apply(check, axis=1)

    return self

@_copy
def ai(self) -> pd.DataFrame:
    """
    Calculate AI (aromaticity index)

    Add column "AI" to self

    Return
    ------
    Spectrum

    References
    ----------
    Koch, Boris P., and T. Dittmar. "From mass to structure: An aromaticity 
    index for high resolution mass data of natural organic matter." 
    Rapid communications in mass spectrometry 20.5 (2006): 926-932.
    """

    if "DBE_AI" not in self:
        self = dbe_ai(self)

    if "CAI" not in self:
        self = cai(self)

    self["AI"] = self["DBE_AI"] / self["CAI"]

    clear  = self["AI"].values[np.isfinite(self["AI"].values.astype('float'))].astype('float')
    self['AI'] = self['AI'].replace(-np.inf, np.min(clear))
    self['AI'] = self['AI'].replace(np.inf, np.max(clear))
    self['AI'] = self['AI'].replace(np.nan, np.mean(clear))

    return self

@_copy
def cai(self) -> pd.DataFrame:
    """
    Calculate CAI (C - O - N - S - P)

    Add column "CAI" to self

    Return
    ------
    Spectrum
    """
    
    if "assign" not in self:
        raise Exception("Spectrum is not assigned")

    table = ms.merge_isotopes(self)

    for element in "CONSP":
        if element not in table:
            table[element] = 0

    self['CAI'] = table["C"] - table["O"] - table["N"] - table["S"] - table["P"]

    return self

@_copy
def dbe_ai(self) -> pd.DataFrame:
    """
    Calculate DBE_AI (1 + C - O - S - 0.5 * (H + N + P))

    Add column "DBE_AI" to self

    Return
    ------
    Spectrum
    """

    if "assign" not in self:
        raise Exception("Spectrum is not assigned")

    table = ms.merge_isotopes(self)

    for element in "CHONPS":
        if element not in table:
            table[element] = 0

    self['DBE_AI'] = 1.0 + table["C"] - table["O"] - table["S"] - 0.5 * (table["H"] + table['N'] + table["P"])

    return self

@_copy
def dbe(self) -> pd.DataFrame:
    """
    Calculate DBE (1 + C - 0.5 * (H + N))

    Add column "DBE" to self

    Return
    ------
    Spectrum
    """

    if "assign" not in self:
        raise Exception("Spectrum is not assigned")

    table = ms.merge_isotopes(self)

    for element in "CHON":
        if element not in table:
            table[element] = 0

    self['DBE'] = 1.0 + table["C"] - 0.5 * (table["H"] - table['N'])

    return self

@_copy
def dbe_o(self) -> pd.DataFrame:
    """
    Calculate DBE - O

    Add column "DBE-O" to self

    Return
    ------
    Spectrum 
    """

    if "DBE" not in self:
        self = dbe(self)

    table = ms.merge_isotopes(self)
    self['DBE-O'] = table['DBE'] - table['O']

    return self

@_copy
def dbe_oc(self) -> pd.DataFrame:
    """
    Calculate (DBE - O) / C

    Add column "DBE-OC" to self

    Return
    ------
    Spectrum
    """

    if "DBE" not in self:
        self = dbe(self)

    table = ms.merge_isotopes(self)
    self['DBE-OC'] = (table['DBE'] - table['O'])/table['C']

    return self

@_copy
def hc_oc(self) -> pd.DataFrame:
    """
    Calculate H/C and O/C

    Add columns "H/C" and "O/C" to self

    Return
    ------
    Spectrum
    """

    if "assign" not in self:
        raise Exception("Spectrum is not assigned")

    table = ms.merge_isotopes(self)
    self['H/C'] = table['H']/table['C']
    self['O/C'] = table['O']/table['C']

    return self

@_copy
def kendrick(self) -> pd.DataFrame:
    """
    Calculate Kendrick mass and Kendrick mass defect

    Add columns "Ke" and 'KMD" to self

    Return
    ------
    Spectrum
    """

    if 'calc_mass' not in self:
        self = calc_mass(self)

    self['Ke'] = self['calc_mass'] * 14/14.01565
    self['KMD'] = np.floor(self['calc_mass'].values.astype('float')) - np.array(self['Ke'].values)
    self.loc[self['KMD']<=0, 'KMD'] = self.loc[self['KMD']<=0, 'KMD'] + 1

    return self

@_copy
def nosc(self) -> pd.DataFrame:
    """
    Calculate Normal oxidation state of carbon (NOSC)

    Add column "NOSC" to self

    Notes
    -----
    >0 - oxidate state.
    <0 - reduce state.
    0 - neutral state

    References
    ----------
    Boye, Kristin, et al. "Thermodynamically 
    controlled preservation of organic carbon 
    in floodplains."
    Nature Geoscience 10.6 (2017): 415-419.

    Return
    ------
    Spectrum
    """

    if "assign" not in self:
        raise Exception("Spectrum is not assigned")

    table = ms.merge_isotopes(self)

    for element in "CHONS":
        if element not in table:
            table[element] = 0

    self['NOSC'] = 4.0 - (table["C"] * 4 + table["H"] - table['O'] * 2 - table['N'] * 3 - table['S'] * 2)/table['C']

    return self

@_copy
def mol_class(self, how: Optional[str] = None) -> pd.DataFrame:
    """
    Assign molecular class for formulas

    Add column "class" to self

    Parameters
    ----------
    how: {'kellerman', 'perminova', 'laszakovits'}
        How devide to calsses. Optional. Default 'laszakovits'

    Return
    ------
    Spectrum

    References
    ----------
    Laszakovits, J. R., & MacKay, A. A. Journal of the American Society for Mass Spectrometry, 2021, 33(1), 198-202.
    A. M. Kellerman, T. Dittmar, D. N. Kothawala, L. J. Tranvik. Nat. Commun. 2014, 5, 3804
    Perminova I. V. Pure and Applied Chemistry. 2019. Vol. 91, № 5. P. 851-864
    """

    if 'AI' not in self:
        self = ai(self)
    if 'H/C' not in self or 'O/C' not in self:
        self = hc_oc(self)

    table = ms.merge_isotopes(self)

    for element in "CHON":
        if element not in table:
            table[element] = 0

    def get_zone_kell(row):

        if row['H/C'] >= 1.5:
            if row['O/C'] < 0.3 and row['N'] == 0:
                return 'lipids'
            elif row['N'] >= 1:
                return 'N-satureted'
            else:
                return 'aliphatics'
        elif row['H/C'] < 1.5 and row['AI'] < 0.5:
            if row['O/C'] <= 0.5:
                return 'unsat_lowOC'
            else:
                return 'unsat_highOC'
        elif row['AI'] > 0.5 and row['AI'] <= 0.67:
            if row['O/C'] <= 0.5:
                return 'aromatic_lowOC'
            else:
                return 'aromatic_highOC'
        elif row['AI'] > 0.67:
            if row['O/C'] <= 0.5:
                return 'condensed_lowOC'
            else:
                return 'condensed_highOC'
        else:
            return 'undefinded'
    
    def get_zone_perm(row):

        if row['O/C'] < 0.5:
            if row['H/C'] < 1:
                return 'condensed_tanins'
            elif row['H/C'] < 1.4:
                return 'phenylisopropanoids'
            elif row['H/C'] < 1.8:
                return 'terpenoids'
            elif row['H/C'] <= 2.2:
                if row['O/C'] < 0.25:
                    return 'lipids'
                else:
                    return 'proteins'
            else:
                return 'undefinded'
        elif row['O/C'] <= 1:
            if row['H/C'] < 1.4:
                return 'hydrolyzable_tanins'
            elif row['H/C'] <= 2.2:
                return 'carbohydrates'
            else:
                return 'undefinded'
        else:
            return 'undefinded'

    def get_zone_lasz(row):
        if row['H/C'] >= 0.86 and row['H/C'] <=1.34 and row['O/C'] >= 0.21 and row['O/C'] <=0.44:
            return 'lignin'
        elif row['H/C'] >= 0.7 and row['H/C'] <=1.01 and row['O/C'] >= 0.16 and row['O/C'] <=0.84:
            return 'tannin'
        elif row['H/C'] >= 1.33 and row['H/C'] <=1.84 and row['O/C'] >= 0.17 and row['O/C'] <=0.48:
            return 'peptide'
        elif row['H/C'] >= 1.34 and row['H/C'] <=2.18 and row['O/C'] >= 0.01 and row['O/C'] <=0.35:
            return 'lipid'
        elif row['H/C'] >= 1.53 and row['H/C'] <=2.2 and row['O/C'] >= 0.56 and row['O/C'] <=1.23:
            return 'carbohydrate'
        elif row['H/C'] >= 1.62 and row['H/C'] <=2.35 and row['O/C'] >= 0.56 and row['O/C'] <=0.95:
            return 'aminosugar'
        else:
            return 'undefinded'
    
    if how == 'perminova':
        self['class'] = table.apply(get_zone_perm, axis=1)
    elif how == 'kellerman':
        self['class'] = table.apply(get_zone_kell, axis=1)
    else:
        self['class'] = table.apply(get_zone_lasz, axis=1)

    return self

@_copy
def get_mol_class(self, how_average: str = "weight", how: Optional[str] = None) -> pd.DataFrame:
    """
    get molercular class density

    Parameters
    ----------
    how_average: {'weight', 'count'}
        how average density. Default "weight" - weight by intensity.
        Also can be "count".
    how: {'kellerman', 'perminova', 'laszakovits'}
        How devide to calsses. Optional. Default 'laszakovits'

    Return
    ------
    pandas Dataframe
    
    References
    ----------
    Laszakovits, J. R., & MacKay, A. A. Journal of the American Society for Mass Spectrometry, 2021, 33(1), 198-202.
    A. M. Kellerman, T. Dittmar, D. N. Kothawala, L. J. Tranvik. Nat. Commun. 5, 3804 (2014)
    Perminova I. V. Pure and Applied Chemistry. 2019. Vol. 91, № 5. P. 851-864
    """

    self = mol_class(ms.drop_unassigned(self),how=how)
    count_density = len(self)
    sum_density = self["intensity"].sum()

    out = []

    if how == 'perminova':
        zones = ['condensed_tanins',
                'hydrolyzable_tanins',
                'phenylisopropanoids',
                'terpenoids',
                'lipids',
                'proteins',
                'carbohydrates',
                'undefinded']
    elif how == 'kellerman':
        zones = ['unsat_lowOC',
                'unsat_highOC',
                'condensed_lowOC',
                'condensed_highOC',
                'aromatic_lowOC',
                'aromatic_highOC',
                'aliphatics',            
                'lipids',
                'N-satureted',
                'undefinded']
    else:
        zones = ['aminosugar',
                'carbohydrate',
                'lignin',
                'lipid',
                'peptide',
                'tannin',
                'undefinded']


    for zone in zones:

        if how_average == "count":
            out.append([zone, len(self.loc[self['class'] == zone])/count_density])

        elif how_average == "weight":
            out.append([zone, self.loc[self['class'] == zone, 'intensity'].sum()/sum_density])

        else:
            raise ValueError(f"how_average should be count or intensity not {how_average}")
    
    return pd.DataFrame(data=out, columns=['class', 'density'])

@_copy
def get_dbe_vs_o(self, 
                    olim: Optional[Tuple[int, int]] = None, 
                    draw: bool = True, 
                    ax: Union[Axes, None] = None, 
                    **kwargs) -> Tuple[float, float]:
    """
    Calculate DBE vs nO by linear fit
    
    Parameters
    ----------
    olim: tuple of two int
        limit for nO. Deafult None
    draw: bool
        draw scatter DBE vs nO and how it is fitted
    ax: matplotlib axes
        ax fo outer plot. Default None
    **kwargs: dict
        dict for additional condition to scatter matplotlib

    Return
    ------
    (float, float)
        a and b in fit DBE = a * nO + b

    References
    ----------
    Bae, E., Yeo, I. J., Jeong, B., Shin, Y., Shin, K. H., & Kim, S. (2011). 
    Study of double bond equivalents and the numbers of carbon and oxygen 
    atom distribution of dissolved organic matter with negative-mode FT-ICR MS.
    Analytical chemistry, 83(11), 4193-4199.
    """

    if 'DBE' not in self:
        self = dbe(self)
    
    self = ms.drop_unassigned(self)
    if olim is None:
        no = list(range(int(self['O'].min())+5, int(self['O'].max())-5))
    else:
        no = list(range(olim[0],olim[1]))
    
    dbe_o = []
    
    for i in no:
        dbes = self.loc[self['O'] == i, 'DBE']
        intens = self.loc[self['O'] == i, 'intensity']
        dbe_o.append((dbes*intens).sum()/intens.sum())

    def linear(x, a, b):
        return a*x + b

    x = np.array(no)
    y = np.array(dbe_o)

    popt, pcov = curve_fit(linear, x, y)
    residuals = y- linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    if draw:
        if ax is None:
            fig,ax = plt.subplots(figsize=(3,3), dpi=100)
        
        ax.scatter(x, y, **kwargs)
        ax.plot(x, linear(x, *popt), label=f'y={round(popt[0],2)}x + {round(popt[1],1)} R2={round(r_squared, 4)}', **kwargs)
        ax.set_xlim(4)
        ax.set_ylim(5)
        ax.set_xlabel('number of oxygen')
        ax.set_ylabel('DBE average')
        ax.legend()

    return popt[0], popt[1]

@_copy
def get_squares_vk(self,
                   how_average: str = 'weight',
                   ax: Union[Axes, None] = None,
                   draw: bool = False,
                   hc_bins: Optional[List[Tuple[float, float]]] = None,
                   oc_bins: Optional[List[Tuple[float, float]]] = None,
                   square_indices: Optional[str] = 'row_major') -> pd.DataFrame:
    """
    Calculate density in Van Krevelen diagram divided into custom grid of squares.
    
    Squares index in Van-Krevelen diagram if H/C is rows, O/C is columns:
    [[5, 10, 15, 20],
        [4, 9, 14, 19],
        [3, 8, 13, 18],
        [2, 7, 12, 17],
        [1, 6, 11, 16]]

    H/C divided by [0-0.6, 0.6-1, 1-1.4, 1.4-1.8, 1.8-2.2]
    O/C divided by [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]

    Parameters
    ----------
    how_average: {'weight', 'count'}
        How calculate average. 'count' for number of points, 'weight' for intensity sum
    ax: matplotlib ax, optional
        External ax for plotting
    draw: bool, default False
        Plot heatmap
    hc_bins: List[Tuple[float, float]], optional
        H/C bins as list of (min, max) tuples from high to low
        Default: [(1.8, 2.2), (1.4, 1.8), (1, 1.4), (0.6, 1), (0, 0.6)]
    oc_bins: List[Tuple[float, float]], optional
        O/C bins as list of (min, max) tuples from low to high
        Default: [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    square_indices: str, default 'row_major'
        How to number squares: 'row_major' (row by row), 'col_major' (column by column),
        'custom' (use custom mapping), or 'standard' (original numbering)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'value', 'square_index', 'row', 'col', 'hc_range', 'oc_range'
    References
    ----------
    Perminova I. V. From green chemistry and nature-like technologies towards 
    ecoadaptive chemistry and technology // Pure and Applied Chemistry. 
    2019. Vol. 91, № 5. P. 851-864.
    """

    # Default bins (high H/C to low H/C, low O/C to high O/C)
    if hc_bins is None:
        hc_bins = [(1.8, 2.2), (1.4, 1.8), (1, 1.4), (0.6, 1), (0, 0.6)]
    
    if oc_bins is None:
        oc_bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    
    # Ensure H/C and O/C columns exist
    if 'H/C' not in self or 'O/C' not in self:
        self = ms.drop_unassigned(hc_oc(self))
    
    total_intensity = self['intensity'].sum() if how_average == 'weight' else len(self)
    squares_data = []
    
    # Calculate density for each square
    for row_idx, (hc_min, hc_max) in enumerate(hc_bins):
        for col_idx, (oc_min, oc_max) in enumerate(oc_bins):
            # Filter points in current square
            mask = ((self['O/C'] >= oc_min) & (self['O/C'] < oc_max) &
                    (self['H/C'] >= hc_min) & (self['H/C'] < hc_max))
            temp = self[mask]
            
            # Calculate value
            if how_average == 'count':
                value = len(temp) / total_intensity
            elif how_average == 'weight':
                value = temp['intensity'].sum() / total_intensity
            else:
                raise ValueError("how_average must be 'count' or 'weight'")
            
            # Calculate square index based on numbering scheme
            if square_indices == 'row_major':
                square_idx = row_idx * len(oc_bins) + col_idx + 1
            elif square_indices == 'col_major':
                square_idx = col_idx * len(hc_bins) + row_idx + 1
            elif square_indices == 'standard':
                # Original numbering from the function
                standard_map = {
                    (0,0):5, (0,1):10, (0,2):15, (0,3):20,
                    (1,0):4, (1,1):9, (1,2):14, (1,3):19,
                    (2,0):3, (2,1):8, (2,2):13, (2,3):18,
                    (3,0):2, (3,1):7, (3,2):12, (3,3):17,
                    (4,0):1, (4,1):6, (4,2):11, (4,3):16
                }
                square_idx = standard_map.get((row_idx, col_idx), square_indices)
            else:
                square_idx = row_idx * len(oc_bins) + col_idx + 1
            
            squares_data.append({
                'value': value,
                'square_index': square_idx,
                'row': row_idx,
                'col': col_idx,
                'hc_range': f'{hc_min}-{hc_max}',
                'oc_range': f'{oc_min}-{oc_max}',
                'hc_min': hc_min,
                'hc_max': hc_max,
                'oc_min': oc_min,
                'oc_max': oc_max
            })
    
    # Create DataFrame
    out_df = pd.DataFrame(squares_data)
    
    # Create pivot table for heatmap
    pivot_data = out_df.pivot(index='row', columns='col', values='value')
    pivot_data.index = [f'H/C: {hc_bins[i][0]}-{hc_bins[i][1]}' for i in range(len(hc_bins))]
    pivot_data.columns = [f'O/C: {oc_bins[j][0]}-{oc_bins[j][1]}' for j in range(len(oc_bins))]
    
    if draw:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
        sns.heatmap(pivot_data.round(4), cmap='coolwarm', annot=True, 
                   linewidths=.5, ax=ax, cbar_kws={'label': 'Density'})
        ax.set_xlabel('O/C')
        ax.set_ylabel('H/C')
        ax.set_title('Van Krevelen Diagram Square Densities')
    
    # Store metadata in attrs
    out_df.attrs['hc_bins'] = hc_bins
    out_df.attrs['oc_bins'] = oc_bins
    out_df.attrs['shape'] = (len(hc_bins), len(oc_bins))
    
    return out_df.sort_values('square_index').reset_index(drop=True)

def get_squares_vk_auto(self,
                        n_hc: int = 5,
                        n_oc: int = 4,
                        how_average: str = 'weight',
                        hc_range: Tuple[float, float] = (0, 2.2),
                        oc_range: Tuple[float, float] = (0, 1),
                        ax: Union[Axes, None] = None,
                        draw: bool = False,
                        square_indices: str = 'row_major') -> pd.DataFrame:
    """
    Calculate density in Van Krevelen diagram with automatic grid division.
    
    Automatically divides H/C and O/C ranges into specified number of squares.
    
    Parameters
    ----------
    n_hc : int, default=5
        Number of divisions along H/C axis (rows)
    n_oc : int, default=4
        Number of divisions along O/C axis (columns)
    how_average : {'weight', 'count'}, default='weight'
        How to calculate average: 'count' for number of points, 'weight' for intensity sum
    hc_range : Tuple[float, float], default=(0, 2.2)
        Range for H/C values (min, max)
    oc_range : Tuple[float, float], default=(0, 1)
        Range for O/C values (min, max)
    ax : matplotlib.axes.Axes, optional
        External ax for plotting
    draw : bool, default=False
        Plot heatmap
    square_indices : str, default='row_major'
        How to number squares: 'row_major' (row by row), 'col_major' (column by column),
        'standard' (original numbering pattern for 5x4 grid)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'value', 'square_index', 'row', 'col', 'hc_range', 'oc_range'
    
    Examples
    --------
    >>> # 3x3 grid (9 squares total)
    >>> squares = spectrum.get_squares_vk_auto(n_hc=3, n_oc=3)
    
    >>> # 10x10 grid (100 squares total)
    >>> squares = spectrum.get_squares_vk_auto(n_hc=10, n_oc=10)
    
    >>> # Custom ranges
    >>> squares = spectrum.get_squares_vk_auto(n_hc=4, n_oc=4, 
    ...                                         hc_range=(0.5, 2.0), 
    ...                                         oc_range=(0.1, 0.9))
    """
    
    # Validate inputs
    if n_hc <= 0 or n_oc <= 0:
        raise ValueError("Number of divisions must be positive")
    
    if hc_range[0] >= hc_range[1]:
        raise ValueError("hc_range[0] must be less than hc_range[1]")
    
    if oc_range[0] >= oc_range[1]:
        raise ValueError("oc_range[0] must be less than oc_range[1]")
    
    # Generate bins automatically
    hc_edges = np.linspace(hc_range[0], hc_range[1], n_hc + 1)
    oc_edges = np.linspace(oc_range[0], oc_range[1], n_oc + 1)
    
    # Create bins as tuples (from high to low for H/C to match original orientation)
    hc_bins = [(hc_edges[i], hc_edges[i+1]) for i in range(n_hc - 1, -1, -1)]  # Reverse for high to low
    oc_bins = [(oc_edges[i], oc_edges[i+1]) for i in range(n_oc)]
    
    # Ensure H/C and O/C columns exist
    if 'H/C' not in self or 'O/C' not in self:
        self = ms.drop_unassigned(hc_oc(self))
    
    total_intensity = self['intensity'].sum() if how_average == 'weight' else len(self)
    squares_data = []
    
    # Calculate density for each square
    for row_idx, (hc_min, hc_max) in enumerate(hc_bins):
        for col_idx, (oc_min, oc_max) in enumerate(oc_bins):
            # Filter points in current square
            mask = ((self['O/C'] >= oc_min) & (self['O/C'] < oc_max) &
                    (self['H/C'] >= hc_min) & (self['H/C'] < hc_max))
            temp = self[mask]
            
            # Calculate value
            if how_average == 'count':
                value = len(temp) / total_intensity if len(temp) > 0 else 0
            elif how_average == 'weight':
                value = temp['intensity'].sum() / total_intensity if len(temp) > 0 else 0
            else:
                raise ValueError("how_average must be 'count' or 'weight'")
            
            # Calculate square index
            if square_indices == 'row_major':
                square_idx = row_idx * n_oc + col_idx + 1
            elif square_indices == 'col_major':
                square_idx = col_idx * n_hc + row_idx + 1
            elif square_indices == 'standard' and n_hc == 5 and n_oc == 4:
                # Original numbering pattern for 5x4 grid
                standard_map = {
                    (0,0):5, (0,1):10, (0,2):15, (0,3):20,
                    (1,0):4, (1,1):9, (1,2):14, (1,3):19,
                    (2,0):3, (2,1):8, (2,2):13, (2,3):18,
                    (3,0):2, (3,1):7, (3,2):12, (3,3):17,
                    (4,0):1, (4,1):6, (4,2):11, (4,3):16
                }
                square_idx = standard_map.get((row_idx, col_idx), square_indices)
            else:
                square_idx = row_idx * n_oc + col_idx + 1
            
            squares_data.append({
                'value': value,
                'square_index': square_idx,
                'row': row_idx,
                'col': col_idx,
                'hc_min': hc_min,
                'hc_max': hc_max,
                'oc_min': oc_min,
                'oc_max': oc_max,
                'hc_range': f'{hc_min:.2f}-{hc_max:.2f}',
                'oc_range': f'{oc_min:.2f}-{oc_max:.2f}'
            })
    
    # Create DataFrame
    out_df = pd.DataFrame(squares_data)
    
    # Create pivot table for heatmap
    pivot_data = out_df.pivot(index='row', columns='col', values='value')
    pivot_data.index = [f'H/C: {hc_bins[i][0]:.2f}-{hc_bins[i][1]:.2f}' for i in range(n_hc)]
    pivot_data.columns = [f'O/C: {oc_bins[j][0]:.2f}-{oc_bins[j][1]:.2f}' for j in range(n_oc)]
    
    if draw:
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(4, n_oc * 0.8), max(4, n_hc * 0.8)), dpi=75)
        
        # Adjust annotation size based on grid density
        annot = n_hc * n_oc <= 100  # Don't show annotations for very dense grids
        
        sns.heatmap(pivot_data.round(4), cmap='coolwarm', annot=annot, 
                   linewidths=.5, ax=ax, cbar_kws={'label': 'Density'})
        ax.set_xlabel('O/C')
        ax.set_ylabel('H/C')
        ax.set_title(f'Van Krevelen Diagram ({n_hc}x{n_oc} squares)')
        
        if n_hc * n_oc > 100:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Store metadata
    out_df.attrs['n_hc'] = n_hc
    out_df.attrs['n_oc'] = n_oc
    out_df.attrs['hc_bins'] = hc_bins
    out_df.attrs['oc_bins'] = oc_bins
    out_df.attrs['hc_range'] = hc_range
    out_df.attrs['oc_range'] = oc_range
    out_df.attrs['shape'] = (n_hc, n_oc)
    
    return out_df.sort_values('square_index').reset_index(drop=True)

def extract_square_intensities(
    spectra_list: List,
    n_hc: int = 5,
    n_oc: int = 4,
    square_indices: Optional[List[int]] = None,
    normalize: bool = True,
    how_average: str = 'weight',
    hc_range: Tuple[float, float] = (0, 2.2),
    oc_range: Tuple[float, float] = (0, 1),
    save_excel: bool = False,
    excel_path: Optional[str] = None,
    plot: bool = False,
    plot_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Extract square intensities using automatic grid division.
    
    Parameters
    ----------
    spectra_list : List
        List of spectrum objects
    n_hc : int, default=5
        Number of divisions along H/C axis
    n_oc : int, default=4
        Number of divisions along O/C axis
    square_indices : List[int], optional
        Specific square indices to extract. If None, extracts all squares
    normalize : bool, default=True
        Normalize intensities to sum to 1 for each spectrum
    how_average : str, default='weight'
        'weight' or 'count'
    hc_range : Tuple[float, float], default=(0, 2.2)
        Range for H/C values
    oc_range : Tuple[float, float], default=(0, 1)
        Range for O/C values
    save_excel : bool, default=False
        Save results to Excel
    excel_path : Optional[str], default=None
        Path for Excel output
    plot : bool, default=False
        Create stacked barplot
    plot_path : Optional[str], default=None
        Path for plot output
    
    Returns
    -------
    pd.DataFrame
        DataFrame with square intensities
    """
    
    total_squares = n_hc * n_oc
    
    if square_indices is None:
        square_indices = list(range(1, total_squares + 1))
    else:
        # Validate square indices
        invalid = [idx for idx in square_indices if idx < 1 or idx > total_squares]
        if invalid:
            raise ValueError(f"Invalid square indices {invalid}. Must be between 1 and {total_squares}")
    
    # Extract values
    square_data_dict = {idx: [] for idx in square_indices}
    
    for spectrum in spectra_list:
        squares_df = get_squares_vk_auto(spectrum, n_hc=n_hc, n_oc=n_oc,
                                         how_average=how_average,
                                         hc_range=hc_range, oc_range=oc_range)
        
        square_map = dict(zip(squares_df['square_index'], squares_df['value']))
        
        for idx in square_indices:
            square_data_dict[idx].append(square_map.get(idx, 0.0))
    
    # Create DataFrame
    df_squares = pd.DataFrame(square_data_dict)
    
    # Add names
    df_squares.insert(0, 'name', [getattr(s, 'attrs', {}).get('name', f'spectrum_{i}') 
                                   for i, s in enumerate(spectra_list)])
    
    # Normalize
    if normalize:
        numeric_cols = df_squares.select_dtypes(include=[np.number]).columns
        row_sums = df_squares[numeric_cols].sum(axis=1)
        df_squares[numeric_cols] = df_squares[numeric_cols].div(row_sums, axis=0).fillna(0)
    
    # Store metadata
    df_squares.attrs['n_hc'] = n_hc
    df_squares.attrs['n_oc'] = n_oc
    df_squares.attrs['grid_size'] = f'{n_hc}x{n_oc}'
    
    # Save and plot
    if save_excel and excel_path:
        import os
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        df_squares.to_excel(excel_path, index=False)
        print(f"Saved to {excel_path}")
    
    if plot and plot_path:
        import os
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plot_kwargs = kwargs.get('plot_kwargs', {})
        mv.plot_stacked_barplot(df_squares, **plot_kwargs)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
    
    return df_squares

@_copy
def get_mol_metrics(self, 
                    metrics: set[str], 
                    func: Optional[str] = None) -> pd.DataFrame:
    """
    Get average metrics

    Parameters
    ----------
    metrics: Sequence[str]
        Optional. Default None. Chose metrics fot watch.
    func: {'weight', 'mean', 'median', 'max', 'min', 'std'}
        How calculate average. My be "weight" (default - weight average on intensity),
        "mean", "median", "max", "min", "std" (standard deviation)

    Return
    ------
    pandas DataFrame
    """

    #self = self.calc_all_metrics().drop_unassigned().normalize()
    self = ms.normalize(
        ms.drop_unassigned(
            calc_all_metrics(self)
            )
            )

    if metrics is None:
        metrics = set(self.columns) - set(['intensity', 'calc_mass', 'rel_error','abs_error',
                                                'assign', 'charge', 'class', 'brutto', 'Ke', 'KMD'])

    res = []
    sorted_metrics = np.sort(np.array(list(metrics)))

    if func is None:
        func = 'weight'

    func_dict = {'mean': lambda col : np.average(self[col]),
                'weight': lambda col : np.average(self[col], weights=self['intensity']),
                'median': lambda col : np.median(self[col]),
                'max': lambda col : np.max(self[col]),
                'min': lambda col : np.min(self[col]),
                'std': lambda col : np.std(self[col])}
    if func not in func_dict:
        raise ValueError(f'not correct value - {func}')
    else:
        f = func_dict[func]

    for col in sorted_metrics:
        try:
            res.append([col, f(col)])
        except:
            res.append([col, np.nan])

    return pd.DataFrame(data=res, columns=['metric', 'value'])

@_copy
def calc_mass(self,debug=False) -> pd.DataFrame:
    """
    Calculate mass from assigned brutto formulas and elements exact masses

    Add column "calc_mass" to self

    Return
    ------
    Spectrum
    """

    if "assign" not in self:
        raise Exception("Spectrum is not assigned")
    
    elems = ms.find_elements(self)
    
    if debug:
        print(elems)
        print(self.loc[:,elems])
        
    table = self.loc[:,elems].copy(deep=True)
    
    masses = get_elements_masses(elems)

    self["calc_mass"] = table.multiply(masses).sum(axis=1)
    self["calc_mass"] = np.round(self["calc_mass"], 6)
    self.loc[self["calc_mass"] == 0, "calc_mass"] = np.nan

    return self

@_copy
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

    self = ms.drop_unassigned(self)

    if "calc_mass" not in self:
        self = calc_mass(self)

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

@_copy
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
        self = calc_mass(self)

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

@_copy
def brutto(self) -> pd.DataFrame:
    """
    Calculate string with brutto from assign table

    Add column "britto" to self

    Return
    ------
    Spectrum
    """
    
    if "assign" not in self:
        raise Exception("Spectrum is not assigned")
    
    # Кэшируем элементы один раз
    if 'elems' not in self.attrs:
        self.attrs['elems'] = self.find_elements()
    elems = self.attrs['elems']
    
    processed_elems = []
    for el in elems:
        if '_' in el:
            processed_elems.append(f'({el})')
        else:
            processed_elems.append(el)
    
    # Векторизованная операция
    def build_row(row):
        parts = []
        for el, p_el in zip(elems, processed_elems):
            val = row[el]
            if val == 1:
                parts.append(p_el)
            elif val > 0:
                parts.append(f'{p_el}{int(val)}')
        return ''.join(parts)
    
    self['brutto'] = self.apply(build_row, axis=1)
    return self

def get_elements_masses(elems: Sequence[str]) -> np.ndarray :
    """
    Get elements masses from list

    Parameters
    ----------
    elems: Sequence[str]
        List of elements. Example: ['C', 'H', 'N', 'C_13', 'O']

    Return
    ------
    numpy array
    """
    
    elements = ms.elements_table()    
    elems_masses = []

    for el in elems:
        if '_' not in el:
            temp = elements.loc[elements['element']==el].sort_values(by='abundance',ascending=False).reset_index(drop=True)
            elems_masses.append(temp.loc[0,'mass'])
        else:
            temp = elements.loc[elements['element_isotop']==el].reset_index(drop=True)
            elems_masses.append(temp.loc[0,'mass'])

    return np.array(elems_masses)

@_copy
def calc_all_metrics(self) -> pd.DataFrame:
    """
    Calculated all available metrics

    Return
    ------
    Spectrum
    """

    self = calc_mass(self)
    self = calc_error(self)
    self = dbe(self)
    self = dbe_o(self)
    self = ai(self)
    self = dbe_oc(self)
    self = dbe_ai(self)
    self = mol_class(self)
    self = hc_oc(self)
    self = cai(self)
    self = cram(self)
    self = nosc(self)
    self = brutto(self)
    self = kendrick(self)

    return self

def average_formulas_per_mass_interval(
    spectra_list: List[pd.DataFrame],
    mass_range: Tuple[float, float] = (225.0, 700.0),
    verbose: bool = True
) -> pd.DataFrame:
    """
    Вычисляет среднее количество формул в интервалах масс шириной 1 Да
    для каждого образца в заданном диапазоне масс.

    Параметры:
    ----------
    spectra_list : List[pd.DataFrame]
        Список DataFrame, каждый из которых соответствует одному образцу.
        Должен содержать колонку 'calc_mass' и атрибут attrs['name'] с именем образца.
    mass_range : Tuple[float, float], default (300.0, 450.0)
        Начало и конец диапазона масс (конец не включается в последний интервал).
    verbose : bool, default True
        Выводить ли сообщения о ходе выполнения.

    Возвращает:
    -----------
    avg_df : pd.DataFrame
        DataFrame с индексом = имена образцов и колонкой 'avg_formula_count'.
    """
    if not spectra_list:
        return pd.DataFrame(columns=['avg_formula_count'])

    # Границы бинов: целые числа от start до end+1 для интервалов [i, i+1)
    start_mass = int(np.floor(mass_range[0]))
    end_mass = int(np.ceil(mass_range[1]))
    bins = np.arange(start_mass, end_mass + 1, 1)
    num_intervals = len(bins) - 1  # количество интервалов

    avg_data = []
    total_samples = len(spectra_list)

    for idx, df in enumerate(spectra_list):
        # Получение имени образца
        sample_name = df.attrs.get('name', f'sample_{idx}')
        if verbose:
            print(f"Обработка образца: {sample_name} ({idx+1}/{total_samples})")

        # Проверка наличия колонки 'calc_mass'
        if 'calc_mass' not in df.columns:
            raise ValueError(f"В DataFrame образца '{sample_name}' отсутствует колонка 'calc_mass'")

        masses = df['calc_mass'].values

        # Подсчёт гистограммы
        counts, _ = np.histogram(masses, bins=bins) # type: ignore

        # Среднее количество формул на интервал
        avg_count = np.mean(counts)

        avg_data.append({
            'sample_name': sample_name,
            'avg_formula_count': avg_count
        })

    # Формирование итогового DataFrame
    avg_df = pd.DataFrame(avg_data)
    avg_df.set_index('sample_name', inplace=True)

    if verbose:
        print(f"\nГотово! Средние значения вычислены для {len(avg_df)} образцов.")

    return avg_df
