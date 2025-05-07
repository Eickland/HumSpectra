import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import csv
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
import utilits as ut
plt.rcParams['axes.grid'] = False


def asm_350(data: DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции.
    :return: fluo_param: asm 350, показатель асимметрии спектра при длине возбуждения 350 нм
    """

    row = data[350].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = ut.cut_raman_spline(EM_wavelengths, row, 350)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 420)[0][0]:np.where(EM_wavelengths == 460)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 550)[0][0]:np.where(EM_wavelengths == 600)[0][0]])
    fluo_param = high / low

    return fluo_param


def asm_280(data: DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции.
    :return: fluo_param: asm 280, показатель асимметрии спектра при длине возбуждения 280 нм
    """

    row = data[280].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = ut.cut_raman_spline(EM_wavelengths, row, 280)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 350)[0][0]:np.where(EM_wavelengths == 400)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 475)[0][0]:np.where(EM_wavelengths == 535)[0][0]])
    fluo_param = high / low

    return fluo_param


def cut_spectra(data: DataFrame,
                ex_low_limit: int,
                ex_high_limit: int,
                em_low_limit: int,
                em_high_limit: int) -> DataFrame:
    cut_data = data.loc[em_low_limit:em_high_limit, ex_low_limit:ex_high_limit]

    return cut_data


def plot_heat_map(data: DataFrame,
                  q: float = 0.9995,
                  ax: Optional[plt.axes] = None,
                  xlabel: bool = True,
                  ylabel: bool = True,
                  title: bool = True) -> Axes:
    filtered_data = data.copy()
    filtered_array = filtered_data.to_numpy()
    filtered_array[filtered_array > np.quantile(filtered_array, q)] = 0
    max_value = np.max(filtered_array)
    normalized_data = filtered_array / max_value
    EM_wavelengths = filtered_data.index.to_numpy(dtype="int")
    EX_wavelengths = filtered_data.columns.to_numpy(dtype="int")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.7, 4.8))
    plt.pcolormesh(EM_wavelengths, EX_wavelengths, normalized_data.T, shading="gouraud", vmin=0, vmax=1,
                  cmap=plt.get_cmap('rainbow'))
    if xlabel:
        ax.set_xlabel("λ испускания, нм")
    if ylabel:
        ax.set_ylabel("λ возбуждения, нм")
    if title:
        ax.set_title(f"{data.attrs['name']}")

    return ax


def plot_2d(data: DataFrame,
            ex_wave: int,
            xlabel: bool = True,
            ylabel: bool = True,
            title: bool = True,
            ax: Optional[plt.axes] = None,
            norm: bool = False) -> Axes:
    row = data[ex_wave]
    if norm:
        row = (row - row.min()) / (row.max() - row.min())
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(data.index, row)
    if title:
        ax.set_title(f"{data.attrs['name']}, λ возбуждения: {ex_wave} нм")
    if xlabel:
        ax.set_xlabel("λ испускания, нм")
    if ylabel:
        ax.set_ylabel("Интенсивность")

    return ax
