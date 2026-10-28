import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzMLFile
from typing import Optional

from ..decorators import decorators as ms_dec

def _looks_centroided(peaks, max_rel_step=2e-6):
    """Профиль: соседние точки через ~3e-7 m/z относительных; центроиды — гораздо реже."""
    steps = [np.median(np.diff(mz) / mz[:-1]) for mz, _ in peaks if len(mz) > 10]
    return bool(steps) and float(np.median(steps)) > max_rel_step


def extract_mass_list(mzml_file, ms_level=1, rt_range=None,
                      low_percentile: float = 99, high_percentile: float | None = None,
                      centroid_rel_floor: float = 0.0, centroided: bool | None = None):
    """
    Центроиды: берём все пики (порог шума уже применён при центроидировании),
               при желании отрезаем всё ниже centroid_rel_floor * max.
    Профиль:   старое поведение — порог по процентилю.
               Верхний процентиль по умолчанию выключен: он вырезает вершины сильных пиков.
    """
    exp = MSExperiment()
    MzMLFile().load(mzml_file, exp)

    spectra = [s for s in exp
               if s.getMSLevel() == ms_level
               and (not rt_range or rt_range[0] <= s.getRT() <= rt_range[1])]
    if not spectra:
        return pd.DataFrame(columns=["mass", "intensity"])

    peaks = [s.get_peaks() for s in spectra]
    if centroided is None:
        centroided = _looks_centroided(peaks)
    all_I = np.concatenate([np.asarray(I, float) for _, I in peaks])

    if centroided:
        if len(spectra) > 1:
            print(f"ВНИМАНИЕ: {len(spectra)} центроидных сканов — пики будут продублированы; "
                  "усредняйте профиль до центроидирования")
        lo, hi = centroid_rel_floor * all_I.max(), np.inf
        print(f"Режим: центроиды, порог {lo:.3g} ({centroid_rel_floor:g} от максимума)")
    else:
        lo = np.percentile(all_I, low_percentile)
        hi = np.percentile(all_I, high_percentile) if high_percentile is not None else np.inf
        print(f"Режим: профиль, {low_percentile}-й процентиль {lo:.3g}"
              + (f", верхний {high_percentile}-й {hi:.3g}" if np.isfinite(hi) else ""))

    mz_out, I_out = [], []
    for mz, I in peaks:
        mask = (I >= lo) & (I < hi)
        mz_out.append(mz[mask])
        I_out.append(I[mask])

    df = pd.DataFrame({"mass": np.concatenate(mz_out), "intensity": np.concatenate(I_out)})
    print(f"Пиков до фильтрации: {len(all_I)}, после: {len(df)}")
    return df
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

    self['C13_peak'] = _isotope_peak_found(self["mass"].to_numpy(), C13_C12, rel_error)

    if remove:
        self = self.loc[(self['C13_peak'] == True) & (self['assign'] == True)].reset_index(drop=True)
        self.attrs['filter_C13'] = True


    return self

@ms_dec._copy
def filter_by_S34(
    self,
    rel_error: float = 0.5,
    remove: bool = False,
) -> pd.DataFrame:
    """
    Check if peaks have the same brutto with S34 isotope (+1.995796 Da)

    Parameters
    ----------
    rel_error: float
        Optional. Default 0.5.
        Allowable ppm error when checking S34 isotope peak
    remove: bool
        Optional, default False.
        Drop unassigned peaks and sulfur-containing peaks without S34 isotope.
        Peaks without sulfur are kept.

    Return
    ------
    pd.DataFrame
    """

    self = self.sort_values(by='mass').reset_index(drop=True)

    self['S34_peak'] = _isotope_peak_found(self["mass"].to_numpy(), S34_S32, rel_error)

    if remove:
        has_s = self['S'] > 0 if 'S' in self else False
        self = self.loc[(self['assign'] == True) & (~has_s | self['S34_peak'])].reset_index(drop=True)
        self.attrs['filter_S34'] = True

    return self

C13_C12 = 1.003355  # C13 - C12 mass difference
S34_S32 = 1.995796  # S34 - S32 mass difference


def _isotope_peak_found(masses: np.ndarray,
                        delta: float,
                        rel_error: float) -> np.ndarray:
    """
    For each peak check if there is peak at mass + delta within rel_error ppm.
    masses should be sorted.
    """
    if len(masses) == 0:
        return np.zeros(0, dtype=bool)

    target = masses + delta
    idx = np.searchsorted(masses, target, side='left')
    idx_right = np.clip(idx, 0, len(masses) - 1)
    idx_left = np.clip(idx - 1, 0, len(masses) - 1)
    dist = np.minimum(np.abs(masses[idx_right] - target), np.abs(masses[idx_left] - target))

    return dist <= target * rel_error * 1e-6

