import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
import re
import scipy.interpolate
from scipy.optimize import curve_fit
from scipy import integrate
from typing import Optional, Sequence, Tuple, Callable, Union
from matplotlib.axes import Axes
import HumSpectra.optic.utilits as ut
plt.rcParams['axes.grid'] = False


class Sample:
    def __init__(self, fluo_spectra, uv_spectra, name, mainclass, subclass, debug=False, TOC=None, uv_recall=False, is_uv_bad_spectrum=True):
        self.fluo_spectra = fluo_spectra
        self.uv_spectra = uv_spectra
        self.name = name
        self.mainclass = mainclass
        self.subclass = subclass
        self.debug = debug
        self.TOC = TOC
        self.uv_recall = uv_recall
        self.is_uv_bad_spectrum = is_uv_bad_spectrum
