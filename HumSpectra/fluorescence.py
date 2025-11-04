import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from typing import Union, List
from matplotlib.axes import Axes

import HumSpectra.utilits as ut

def asm_350(data: pd.DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: asm 350, показатель асимметрии спектра при длине возбуждения 350 нм
    Функция рассчитывает отношение интеграла длины волны испускания от 420 до 460 нм к интегралу от 550 до 600 нм
    """

    row = data[350].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = cut_raman_spline(EM_wavelengths, row, 350)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 420)[0][0]:np.where(EM_wavelengths == 460)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 550)[0][0]:np.where(EM_wavelengths == 600)[0][0]])
    fluo_param = high / low

    return float(fluo_param)


def asm_280(data: pd.DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: asm 280, показатель асимметрии спектра при длине возбуждения 280 нм
    Функция рассчитывает отношение интеграла длины волны испускания от 350 до 400 нм к интегралу от 475 до 535 нм
    """

    row = data[280].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = cut_raman_spline(EM_wavelengths, row, 280)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 350)[0][0]:np.where(EM_wavelengths == 400)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 475)[0][0]:np.where(EM_wavelengths == 535)[0][0]])
    fluo_param = high / low

    return float(fluo_param)

def fluo_index(data: pd.DataFrame)-> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: fluo_index, отношение интенсивности при длине волны испускания 450 нм к 500 нм при длине волны возбуждения 370 нм
    """
    row = data[370]
    fluo_param = row.loc[450]/row.loc[500]
    
    return float(fluo_param)


def humin_index(data: pd.DataFrame)-> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: humin_index, Функция рассчитывает отношение интеграла длины волны испускания от 435 до 480 нм к интегралу от 300 до 345 нм при длине возбуждения 254 нм.
    """
    row = data[255].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = cut_raman_spline(EM_wavelengths, row, 255)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 435)[0][0]:np.where(EM_wavelengths == 480)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 300)[0][0]:np.where(EM_wavelengths == 345)[0][0]])
    fluo_param = high / low

    return float(fluo_param) 


def cut_spectra(data: pd.DataFrame,
                ex_low_limit: int | None=None,
                ex_high_limit: int | None = None,
                em_low_limit: int | None = None,
                em_high_limit: int | None = None) -> pd.DataFrame:
    """
    :param data: DataFrame, спектр флуоресценции.
    :param ex_low_limit: int, нижний предел значения длины волны возбуждения спектра
    :param ex_high_limit: int, верхний предел значения длины волны возбуждения спектра
    :param em_low_limit: int, нижний предел значения длины волны испускания спектра
    :param em_high_limit: int, верхний предел значения длины волны испускания спектра
    :return: cut_data
    Функция обрезает спектр согласно заданным пределам и возвращает копию спектра
    """
    data = data.copy()
    
    if ex_low_limit is None:
        ex_low_limit = int(data.columns.min())

    if ex_high_limit is None:
        ex_high_limit = int(data.columns.max())
    
    if em_low_limit is None:
        em_low_limit = data.index.min()
    
    if em_high_limit is None:
        em_high_limit = data.index.max()
    
    cut_data = data.loc[em_low_limit:em_high_limit, ex_low_limit:ex_high_limit]

    return cut_data


def plot_heat_map(data: pd.DataFrame,
                  ax: Union[Axes, None] = None,
                  xlabel: bool = True,
                  ylabel: bool = True,
                  title: bool = True,
                  name: str|None = None) -> Axes:
    """
    :param data: DataFrame, спектр флуоресценции
    :param q: float, квантиль, определяющий экстремальные выбросы
    :return: ax: Axes, ось графика matplotlib.pyplot
    Функция маскирует экстремальные выбросы нулями, и рисует 2D тепловой график флуоресценции
    """
    
    data_copy = data.copy()
    
    if name is None:
        
        if "name" in data_copy.attrs:
            name = data_copy.attrs["name"]
        else:
            name = "fluo_spectra"
    
    EM_wavelengths = data_copy.index.to_numpy(dtype="int")
    EX_wavelengths = data_copy.columns.to_numpy(dtype="int")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.7, 4.8))
    ax.pcolormesh(EM_wavelengths, EX_wavelengths, data_copy.T, shading="gouraud",
                  cmap=plt.get_cmap('rainbow'))
    if xlabel:
        ax.set_xlabel("λ испускания, нм")
    if ylabel:
        ax.set_ylabel("λ возбуждения, нм")
    if title:
        ax.set_title(f"{name}")

    return ax


def plot_2d(data: pd.DataFrame,
            ex_wave: int,
            xlabel: bool = True,
            ylabel: bool = True,
            title: bool = True,
            ax: Union[Axes, None] = None,
            norm: bool = False,
            name: str|None = None) -> Axes:
    """
    :param data: DataFrame, спектр флуоресценции
    :param ex_wave: int, длина волны возбуждения, при котором строится график
    :param norm: bool, если установлен True, то нормирует график на максимум
    :return: ax: Axes, ось графика matplotlib.pyplot
    Функция возвращает график 2D флуоресценции при одной длине волны возбуждения
    """
    
    data_copy = data.copy()
    
    if name is None:
        
        if "name" in data_copy.attrs:
            name = data_copy.attrs["name"]
        else:
            name = "fluo_spectra"
    
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

def where_is_raman(exc_wv: float,
                   omega: float = 3430.):
    """
    return the central wavelength of solvent raman scattering band (stokes mode)
    for given excitation wavelength (exc_wv) in nm
    exc_wv(float) - excitation wavelength in nm
    Omega(float) - molecular oscillations frequency in cm^-1
    return(float) central wavelength of raman stokes mode peak
    """
    return 1e7 / (1e7 / exc_wv - omega)


def cut_peak_spline(em_wvs, fl, peak_position, peak_half_width=15., points_in_spline=10):
    """
    em_wvs - 1d numpy array of emission wavelength
    fl - 1d numpy array of fluorescence
    peak_position (float) - position of peak in nanometers
    peak_half_width - half-width at half-maximum of peak to cut
    """
    mask = ((em_wvs < peak_position - peak_half_width) | (em_wvs > peak_position + peak_half_width)) & (~np.isnan(fl))

    knots = em_wvs[mask][1:-1][::points_in_spline]
    spline = scipy.interpolate.LSQUnivariateSpline(em_wvs[mask], fl[mask], knots, k=2, ext=1)
    spline_wvs = em_wvs[mask]

    return knots, mask, spline_wvs, spline


def cut_raman_spline(em_wvs: np.ndarray,
                     fl: np.ndarray,
                     exc_wv: int,
                     raman_freq: float = 3430.,
                     raman_hwhm: int = 15,
                     points_in_spline: int = 10) -> np.ndarray:
    """
    em_wvs - 1d numpy array of emission wavelength
    fl - 1d numpy array of fluorescence
    raman_freq (float) = 3430. - frequency of raman scattering in cm^-1
    raman_hwhm (float) - raman band half width at half maximum to cut in nm
    points_in_spline(int) - number of points in interpolating spline

    return 1d numpy array of fluorescence intensity with cut emission
    """

    res = cut_peak_spline(em_wvs, fl,
                          peak_position=where_is_raman(exc_wv, omega=raman_freq),
                          peak_half_width=raman_hwhm, points_in_spline=points_in_spline)
    
    return np.array(res[-1](em_wvs))


def read_fluo_3d(path: str,
                 sep: str | None = None,
                 index_col: int = 0,
                 debug: bool = False,
                 dropcols: int = 0) -> pd.DataFrame:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.csv").
    :param sep: разделитель в строчном виде (example: ",").
    :param index_col: номер столбца, который считается индексом таблицы.
    :return: DataFrame: Таблица, в котором индексами строк являются длины волн испускания, имена столбцов - длины волн
            возбуждения. Таблица имеет метаданные - имя образца, класс, подкласс
    """

    extension = path.split(sep=".")[-1]
    

    if sep is None and (extension == "csv" or extension == "txt"):
        sep = ut.check_sep(path)

    try:

        if extension == "xlsx":
            headers = pd.read_csv(path, nrows=0, index_col=index_col).columns
            unnamed_columns = [col for col in headers if col.startswith('Unnamed:')]

            data = pd.read_excel(path, index_col=index_col)
            data = data.drop(unnamed_columns, axis=1)

        elif extension == "csv" or extension == "txt":
            headers = pd.read_csv(path, nrows=0, index_col=index_col).columns
            unnamed_columns = [col for col in headers if col.startswith('Unnamed:')]

            data = pd.read_csv(path, sep=sep, index_col=index_col)
            data = data.drop(unnamed_columns, axis=1)

        else:
            raise KeyError("Тип данных не поддерживается")

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Файл пуст: {path}")
    
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")
    
    name = ut.extract_name_from_path(path)

    if "nm" in data.index:
        data.drop("nm", inplace=True)

    if debug:
        print(name)
        print(data)

    data = data.astype("float64")
    
    data.columns = data.columns.astype(int)
    data.index = data.index.astype(int)

    data.attrs['name'] = name
    data.attrs['class'] = ut.extract_class_from_name(name)
    data.attrs['subclass'] = ut.extract_subclass_from_name(name)

    return data

