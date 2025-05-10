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
import utilits as ut
from scipy.interpolate import Rbf
from scipy.signal import medfilt2d
from matplotlib.colors import LogNorm
plt.rcParams['axes.grid'] = False


def asm_350(data: DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: asm 350, показатель асимметрии спектра при длине возбуждения 350 нм
    Функция рассчитывает отношение интеграла длины волны испускания от 420 до 460 нм к интегралу от 550 до 600 нм
    """

    row = data[350].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = ut.cut_raman_spline(EM_wavelengths, row, 350)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 420)[0][0]:np.where(EM_wavelengths == 460)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 550)[0][0]:np.where(EM_wavelengths == 600)[0][0]])
    fluo_param = high / low

    return high


def asm_280(data: DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: asm 280, показатель асимметрии спектра при длине возбуждения 280 нм
    Функция рассчитывает отношение интеграла длины волны испускания от 350 до 400 нм к интегралу от 475 до 535 нм
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
    """
    :param data: DataFrame, спектр флуоресценции.
    :param ex_low_limit: int, нижний предел значения длины волны возбуждения спектра
    :param ex_high_limit: int, верхний предел значения длины волны возбуждения спектра
    :param em_low_limit: int, нижний предел значения длины волны испускания спектра
    :param em_high_limit: int, верхний предел значения длины волны испускания спектра
    :return: fluo_param: asm 280, показатель асимметрии спектра при длине возбуждения 280 нм
    Функция обрезает спектр согласно заданным пределам и возвращает копию спектра
    """
    cut_data = data.loc[em_low_limit:em_high_limit, ex_low_limit:ex_high_limit]

    return cut_data


def remove_outliers_and_interpolate(data_ini: DataFrame,
                                    q=0.995) -> DataFrame:
    
    """
    Удаляет экстремальные выбросы из 3D матрицы флуоресценции и интерполирует удаленные значения
    по ближайшим соседям.  Использует медианный фильтр для обнаружения выбросов.

    Args:
        data (np.ndarray): 2D NumPy массив, представляющий матрицу флуоресценции.
        outlier_threshold (float): Порог для определения выбросов (в единицах стандартного отклонения).
        median_filter_size (int): Размер ядра медианного фильтра (должен быть нечетным).

    Returns:
        np.ndarray: Матрица с удаленными выбросами и интерполированными значениями.
    """
    data = data_ini.to_numpy()
    index = data_ini.index
    columns = data_ini.columns
    # 1. Медианная фильтрация для обнаружения выбросов
    data[data > np.quantile(data,q)] = np.nan

    # 4. Интерполяция NaN значений с использованием Rbf
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)  # Создаем сетку координат

    # Извлекаем координаты известных точек (не NaN)
    valid_x = X[~np.isnan(data)].ravel()
    valid_y = Y[~np.isnan(data)].ravel()
    valid_z = data[~np.isnan(data)].ravel()

    # Создаем функцию Rbf
    rbfi = Rbf(valid_x, valid_y, valid_z, function='linear') # ('linear', 'gaussian', 'multiquadric')

    # Применяем интерполяцию ко всей сетке
    interpolated_data = rbfi(X, Y)
    max_value = np.max(interpolated_data)
    interpolated_data = interpolated_data/max_value
    interpolated_data = pd.DataFrame(data=interpolated_data,index=index,columns=columns)
    return interpolated_data



def plot_heat_map(data: DataFrame,
                  ax: Optional[plt.axes] = None,
                  xlabel: bool = True,
                  ylabel: bool = True,
                  title: bool = True) -> Axes:
    """
    :param data: DataFrame, спектр флуоресценции
    :param q: float, квантиль, определяющий экстремальные выбросы
    :return: ax: Axes, ось графика matplotlib.pyplot
    Функция маскирует экстремальные выбросы нулями, и рисует 2D тепловой график флуоресценции
    """
    
    EM_wavelengths = data.index.to_numpy(dtype="int")
    EX_wavelengths = data.columns.to_numpy(dtype="int")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.7, 4.8))
    ax.pcolormesh(EM_wavelengths, EX_wavelengths, data.T, shading="gouraud", vmin=0, vmax=1,
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
    """
    :param data: DataFrame, спектр флуоресценции
    :param ex_wave: int, длина волны возбуждения, при котором строится график
    :param norm: bool, если установлен True, то нормирует график на максимум
    :return: ax: Axes, ось графика matplotlib.pyplot
    Функция возвращает график 2D флуоресценции при одной длине волны возбуждения
    """
    row = data[ex_wave]
    if norm:
        row = (row - row.min()) / (row.max() - row.min())
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(data.index, row, label = data.attrs['name'])
    if title:
        ax.set_title(f"{data.attrs['name']}, λ возбуждения: {ex_wave} нм")
    if xlabel:
        ax.set_xlabel("λ испускания, нм")
    if ylabel:
        ax.set_ylabel("Интенсивность")

    return ax
