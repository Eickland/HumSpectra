import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union, List
from matplotlib.axes import Axes
from scipy import signal
from scipy.ndimage import uniform_filter1d

from HumSpectra import utilits as ut

def check_recall_flag(data: DataFrame) -> bool:
    """
    :param data: DataFrame, уф спектр
    :return: флаг, опредеяющий, что спектр рекалиброван или нет
    Функция проверяет наличие в метаданных спектра атрибута "recall" и  значение как True.
    Также есть возможность проигноировать проверки, добавив атрибут "debug" и установив значение "True"
    """
    if 'debug' not in data.attrs:
        if not data.attrs['recall']:
            raise ValueError("Спектр должен быть откалиброван")
    else:
        if not data.attrs['debug']:
            if not data.attrs['recall']:
                raise ValueError("Спектр должен быть откалиброван")
    return True

def base_recall_uv(data: DataFrame) -> DataFrame:
    """
    :param data: DataFrame, уф спектр
    :return: data_copy: DataFrame, рекалиброванный уф спектр
    Функция добавляет базовую линию: рассчитывает минимальное значение и добвляет модуль ко всему спектру.
    Также функция добавляет флаг в метаданные спектра - "recall", и устанавливает значение как True
    """
    data_copy = data.copy()
    min_value = data_copy.min()
    data_copy.attrs['recall'] = True

    return data_copy + 1.001 * abs(min_value)

def calc_r490_555(data: DataFrame,
                     wave_1: int = 490,
                     wave_2: int = 555,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :wave_1/wave_2: значение длины волны в нм (целое число)
    :return: uv_param: float, значение параметра % отражения wave 1 деленной на % отражения wave 2
    
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_1 = series.sub(wave_1).abs().idxmin()
    index_2 = series.sub(wave_2).abs().idxmin()
    uv_param = data.loc[index_1] / data.loc[index_2]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_r412_547(data: DataFrame,
                     wave_1: int = 412,
                     wave_2: int = 547,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :wave_1/wave_2: значение длины волны в нм (целое число)
    :return: uv_param: float, значение параметра % отражения wave 1 деленной на % отражения wave 2
    
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_1 = series.sub(wave_1).abs().idxmin()
    index_2 = series.sub(wave_2).abs().idxmin()
    uv_param = data.loc[index_1] / data.loc[index_2]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_r412_670(data: DataFrame,
                     wave_1: int = 412,
                     wave_2: int = 670,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :wave_1/wave_2: значение длины волны в нм (целое число)
    :return: uv_param: float, значение параметра % отражения wave 1 деленной на % отражения wave 2
    
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_1 = series.sub(wave_1).abs().idxmin()
    index_2 = series.sub(wave_2).abs().idxmin()
    uv_param = data.loc[index_1] / data.loc[index_2]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_r460_560(data: DataFrame,
                     wave_1: int = 460,
                     wave_2: int = 560,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :wave_1/wave_2: значение длины волны в нм (целое число)
    :return: uv_param: float, значение параметра % отражения wave 1 деленной на % отражения wave 2
    
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_1 = series.sub(wave_1).abs().idxmin()
    index_2 = series.sub(wave_2).abs().idxmin()
    uv_param = data.loc[index_1] / data.loc[index_2]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_r280_460(data: DataFrame,
                     wave_1: int = 280,
                     wave_2: int = 460,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :wave_1/wave_2: значение длины волны в нм (целое число)
    :return: uv_param: float, значение параметра % отражения wave 1 деленной на % отражения wave 2
    
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_1 = series.sub(wave_1).abs().idxmin()
    index_2 = series.sub(wave_2).abs().idxmin()
    uv_param = data.loc[index_1] / data.loc[index_2]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param


def calc_ratio_descriptor_uv(data: DataFrame,
                     wave_1: int,
                     wave_2: int,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :wave_1/wave_2: значение длины волны в нм (целое число)
    :return: uv_param: float, значение параметра оптической плотности wave 1 деленной на оптическую плотность wave 2
    
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_1 = series.sub(wave_1).abs().idxmin()
    index_2 = series.sub(wave_2).abs().idxmin()
    uv_param = data.loc[index_1] / data.loc[index_2]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_integral_ratio_uv(data: DataFrame,
                      low_wv_left: float,
                      low_wv_right: float,
                      high_wv_left: float,
                      high_wv_right: float,
          debug: bool=True) -> float:

    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    # Предполагаем, что data - это DataFrame с одним столбцом (поглощение)
    # и индексом, представляющим длины волн
    
    # Находим ближайшие индексы для заданных диапазонов
    series = pd.Series(data.index, index=data.index)
    
    index_low_left = series.sub(low_wv_left).abs().idxmin()
    index_low_right = series.sub(low_wv_right).abs().idxmin()
    index_high_left = series.sub(high_wv_left).abs().idxmin()
    index_high_right = series.sub(high_wv_right).abs().idxmin()
    
    # Извлекаем данные для каждого диапазона
    # Используем .iloc для гарантии правильного порядка
    high_data = data.loc[index_high_left:index_high_right]
    low_data = data.loc[index_low_left:index_low_right]
    
    # Если data имеет несколько столбцов, нужно указать конкретный столбец
    # или проинтегрировать каждый столбец отдельно
    if len(data.columns) > 1:
        # Вариант 1: интегрируем первый столбец
        high = np.trapezoid(high_data.iloc[:, 0], x=high_data.index)
        low = np.trapezoid(low_data.iloc[:, 0], x=low_data.index)
    else:
        # Вариант 2: интегрируем единственный столбец
        high = np.trapezoid(high_data.values.flatten(), x=high_data.index)
        low = np.trapezoid(low_data.values.flatten(), x=low_data.index)
    
    # Проверяем деление на ноль
    if low == 0:
        raise ValueError("Знаменатель (интеграл низкого диапазона) равен нулю")
    
    uv_param = float(high / low)

    return uv_param


def calc_ir_2_18(data: DataFrame,
                      low_wv_left: float = 710,
                      low_wv_right: float = 740,
                      high_wv_left: float = 230,
                      high_wv_right: float = 260,
          debug: bool=True) -> float:

    return calc_integral_ratio_uv(data,
                      low_wv_left,
                      low_wv_right,
                      high_wv_left,
                      high_wv_right)

def calc_ir_19_20(data: DataFrame,
                      low_wv_left: float = 770,
                      low_wv_right: float = 800,
                      high_wv_left: float = 740,
                      high_wv_right: float = 770,
          debug: bool=True) -> float:

    return calc_integral_ratio_uv(data,
                      low_wv_left,
                      low_wv_right,
                      high_wv_left,
                      high_wv_right)

def calc_ir_4_5(data: DataFrame,
                      low_wv_left: float = 320,
                      low_wv_right: float = 350,
                      high_wv_left: float = 290,
                      high_wv_right: float = 320,
          debug: bool=True) -> float:

    return calc_integral_ratio_uv(data,
                      low_wv_left,
                      low_wv_right,
                      high_wv_left,
                      high_wv_right)

def calc_ir_7_8(data: DataFrame,
                      low_wv_left: float = 410,
                      low_wv_right: float = 440,
                      high_wv_left: float = 380,
                      high_wv_right: float = 410,
          debug: bool=True) -> float:

    return calc_integral_ratio_uv(data,
                      low_wv_left,
                      low_wv_right,
                      high_wv_left,
                      high_wv_right)

def calc_e2_e3(data: DataFrame,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :return: uv_param: float, значение параметра E2/E3
    Функция проверяет наличие рекалибровки и рассчитывает отношение оптической плотности при длине волны 265 к 365 нм.
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_265 = series.sub(265).abs().idxmin()
    index_365 = series.sub(365).abs().idxmin()
    uv_param = data.loc[index_265] / data.loc[index_365]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_e3_e4(data: DataFrame,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :return: uv_param: float, значение параметра E4/E6
    Функция проверяет наличие рекалибровки и рассчитывает отношение оптической плотности при длине волны 465 к 665 нм.
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калsибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_365 = series.sub(365).abs().idxmin()
    index_465 = series.sub(465).abs().idxmin()
    uv_param = data.loc[index_365] / data.loc[index_465]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_e4_e6(data: DataFrame,
          debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :return: uv_param: float, значение параметра E4/E6
    Функция проверяет наличие рекалибровки и рассчитывает отношение оптической плотности при длине волны 465 к 665 нм.
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_465 = series.sub(465).abs().idxmin()
    index_665 = series.sub(665).abs().idxmin()
    uv_param = data.loc[index_465] / data.loc[index_665]
    uv_param = float(uv_param.iloc[0].item())

    return uv_param

def calc_single_density(data: DataFrame,
            wave: int = 254,
            debug: bool=True) -> float:
    """
    :param data: DataFrame, уф спектр
    :param wave: int, длина волны, по которой ищется оптическая плотность
    :return: uv_param: float, значение оптической плотности
    Функция проверяет наличие рекалибровки и рассчитывает оптическую плотность при заданной длине волны
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_wave = series.sub(wave).abs().idxmin()
    uv_param = data.loc[index_wave].iloc[0].item()

    return uv_param

def calc_suva(data: DataFrame,
              debug: bool = False,
              toc: float | None = None) -> float:
    """
    :param data: DataFrame, уф спектр
    :param debug: bool, режим отладки
    :param toc: float | None, значение TOC (если не передано, берется из атрибутов)
    :return: uv_param: float, значение параметра SUVA 254
    Функция проверяет наличие рекалибровки и рассчитывает параметр SUVA 254,
    для функции необходимо значение TOC (из атрибутов или параметра)
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    # Определяем значение TOC
    if toc is None:
        # Если toc не передан, пытаемся взять из атрибутов
        if "TOC" not in data.attrs:
            raise KeyError("В метаданных таблицы должно быть значения содержания органического углерода")
        
        toc_value = data.attrs['TOC']
        
        # Проверяем, что в атрибутах не None
        if toc_value is None:
            raise KeyError("Нет значения органического углерода в атрибутах таблицы")
    else:
        # Используем переданное значение
        toc_value = toc
    
    a_254 = calc_single_density(data, 254)
    uv_param = a_254 / toc_value

    return uv_param

def calc_lambda_UV(data: DataFrame,
                       short_wave: int = 450,
                       long_wave: int = 550,
                       debug: bool = True) -> float:
    
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_short = series.sub(short_wave).abs().idxmin()
    index_long = series.sub(long_wave).abs().idxmin()
    
    # Исправление: берем интенсивность, а не длины волн
    wavelengths = data.loc[index_short:index_long].index.to_numpy()
    intensities = data.loc[index_short:index_long][data.columns[0]].to_numpy()
    
    # Правильная аппроксимация: ln(интенсивность) ~ длина волны
    p, *rest = np.polyfit(wavelengths, np.log(intensities), 1, full=True)
    a, b = p
    uv_param = 1 / abs(a)

    return uv_param

def calc_ag(absorbance_series, path_length_cm=1.0):
    """
    Calculate Napierian absorption coefficient a_g(λ)
    
    Parameters:
    - absorbance_series: pandas Series with wavelengths as index and absorbance values
    - path_length_cm: optical path length in cm (default=1 cm)
    
    Returns:
    - pandas Series with a_g(λ) values in m⁻¹
    """
    path_length_m = path_length_cm / 100  # Convert cm to meters
    ag_values = 2.303 * absorbance_series / path_length_m
    return ag_values

def calc_spectral_slope(ag_series, lambda1, lambda2) -> float:
    """
    Calculate spectral slope S between two wavelengths
    
    Parameters:
    - ag_series: pandas Series with a_g(λ) values
    - lambda1, lambda2: wavelength range in nm
    
    Returns:
    - spectral slope S in nm⁻¹
    """
    if lambda1 not in ag_series.index or lambda2 not in ag_series.index:
        # Interpolate if exact wavelengths not present
        ag1 = np.interp(lambda1, ag_series.index, ag_series.values)
        ag2 = np.interp(lambda2, ag_series.index, ag_series.values)
    else:
        ag1 = ag_series.loc[lambda1]
        ag2 = ag_series.loc[lambda2]
    
    if ag1 <= 0 or ag2 <= 0:
        return np.nan
    
    S = - (np.log(ag1) - np.log(ag2)) / (lambda1 - lambda2)
    return S

def calc_B1_band(ag_series) -> float:
    """
    Calculate B1' band intensity
    
    Parameters:
    - ag_series: pandas Series with a_g(λ) values
    
    Returns:
    - B1' value in m⁻¹
    """
    # Wavelength ranges for B1 calculation
    wavelengths = [295, 305, 330]
    
    # Check if we have the required wavelengths
    available_wavelengths = ag_series.index
    if not all(wl in available_wavelengths for wl in wavelengths):
        # Interpolate missing wavelengths
        ag_295 = np.interp(295, available_wavelengths, ag_series.values)
        ag_305 = np.interp(305, available_wavelengths, ag_series.values)
        ag_330 = np.interp(330, available_wavelengths, ag_series.values)
    else:
        ag_295 = ag_series.loc[295]
        ag_305 = ag_series.loc[305]
        ag_330 = ag_series.loc[330]
    
    # Calculate exponential slope between 295-330 nm
    slope_295_330 = (np.log(ag_330) - np.log(ag_295)) / (330 - 295)
    
    # Calculate expected ag(305) from exponential fit
    ag_305_expected = ag_295 * np.exp(slope_295_330 * (305 - 295))
    
    # B1' = measured ag(305) - expected ag(305) from exponential fit
    B1_prime = ag_305 - ag_305_expected
    
    return B1_prime

def calc_B2_band(ag_series) -> float:
    """
    Calculate B2' band intensity
    
    Parameters:
    - ag_series: pandas Series with a_g(λ) values
    
    Returns:
    - B2' value in m⁻¹
    """
    # Wavelength ranges for B2 calculation
    wavelengths = [380, 410, 443]
    
    # Check if we have the required wavelengths
    available_wavelengths = ag_series.index
    if not all(wl in available_wavelengths for wl in wavelengths):
        # Interpolate missing wavelengths
        ag_380 = np.interp(380, available_wavelengths, ag_series.values)
        ag_410 = np.interp(410, available_wavelengths, ag_series.values)
        ag_443 = np.interp(443, available_wavelengths, ag_series.values)
    else:
        ag_380 = ag_series.loc[380]
        ag_410 = ag_series.loc[410]
        ag_443 = ag_series.loc[443]
    
    # Calculate exponential slope between 380-443 nm
    slope_380_443 = (np.log(ag_443) - np.log(ag_380)) / (443 - 380)
    
    # Calculate expected ag(410) from exponential fit
    ag_410_expected = ag_380 * np.exp(slope_380_443 * (410 - 380))
    
    # B2' = measured ag(410) - expected ag(410) from exponential fit
    B2_prime = ag_410 - ag_410_expected
    
    return B2_prime

def calc_cdom_descriptors(absorbance_df, path_length_cm=1.0, doc_concentration=None):
    """
    Calculate all CDOM spectral descriptors from absorbance data
    
    Parameters:
    - absorbance_df: DataFrame with one column of absorbance values and wavelength index
    - path_length_cm: optical path length in cm
    - doc_concentration: DOC concentration for normalized parameters (optional)
    
    Returns:
    - Dictionary with all spectral descriptors
    """
    
    # Get the absorbance series (assuming single column)
    absorbance_series = absorbance_df.iloc[:, 0]
    
    # Calculate a_g(λ) series
    ag_series = calc_ag(absorbance_series, path_length_cm)
    
    # Calculate specific a_g values at key wavelengths
    ag_275 = np.interp(275, ag_series.index, ag_series.values) if 275 not in ag_series.index else ag_series.loc[275]
    ag_380 = np.interp(380, ag_series.index, ag_series.values) if 380 not in ag_series.index else ag_series.loc[380]
    
    # Calculate spectral slopes
    S_275_295 = calc_spectral_slope(ag_series, 275, 295)
    S_380_443 = calc_spectral_slope(ag_series, 380, 443)
    
    # Calculate B bands
    B1_prime = calc_B1_band(ag_series)
    B2_prime = calc_B2_band(ag_series)
    
    # Compile results
    descriptors = {
        'ag_275': ag_275,           # m⁻¹
        'ag_380': ag_380,           # m⁻¹
        'S_275_295': S_275_295,     # nm⁻¹
        'S_380_443': S_380_443,     # nm⁻¹
        'B1_prime': B1_prime,       # m⁻¹
        'B2_prime': B2_prime        # m⁻¹
    }
    
    # Add DOC-normalized values if DOC concentration is provided
    if doc_concentration is not None and doc_concentration > 0:
        descriptors['ag_275_DOC'] = ag_275 / doc_concentration  # L·μmol⁻¹·m⁻¹
        descriptors['ag_380_DOC'] = ag_380 / doc_concentration  # L·μmol⁻¹·m⁻¹
        descriptors['B1_prime_DOC'] = B1_prime / doc_concentration  # L·μmol⁻¹·m⁻¹
        descriptors['B2_prime_DOC'] = B2_prime / doc_concentration  # L·μmol⁻¹·m⁻¹
    
    return descriptors

def calc_differential_absorption_spectrum(ag_series, doc_concentration, reference_ag, reference_doc):
    """
    Calculate DOC-normalized differential absorption spectrum (DAS)
    
    Parameters:
    - ag_series: a_g(λ) values for sample
    - doc_concentration: DOC concentration for sample
    - reference_ag: a_g(λ) values for reference
    - reference_doc: DOC concentration for reference
    
    Returns:
    - DAS values
    """
    das = (ag_series / doc_concentration) / (reference_ag / reference_doc)
    return das

def plot_uv(data: DataFrame,
            xlabel: bool = True,
            ylabel: bool = True,
            title: bool = True,
            ylim: Optional[float] = None,
            norm_by_TOC: bool = False,
            ax: Union[Axes, None] = None,
            name: Optional[str] = None) -> Axes:
    """
    Функция для построения УФ спектров.
    
    :param data: DataFrame, уф спектр
    :param xlabel: показывать подпись оси X
    :param ylabel: показывать подпись оси Y
    :param title: показывать заголовок
    :param norm_by_TOC: нормализовать по TOC
    :param ax: ось для построения (если None, создается новая)
    :param name: название спектра
    :return: ax: Axes, ось графика matplotlib.pyplot
    """
    
    data_copy = data.copy()

    # Определяем имя спектра
    if name is None:
        
        if "name" in data_copy.attrs:
            name = data_copy.attrs["name"]
            
        else:
            name = "uv_spectra"

    # Нормализация по TOC
    if norm_by_TOC:
        
        if "TOC" not in data_copy.attrs:
            raise KeyError("В метаданных спектра должно быть значения содержания органического углерода")
        
        data_copy = data_copy / data_copy.attrs['TOC']

    # Создаем ось если не передана
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Проверяем, что в данных есть столбцы
    if len(data_copy.columns) == 0:
        raise ValueError("DataFrame не содержит столбцов с данными")
    
    # Строим график - используем первый столбец данных
    column_name = data_copy.columns[0]
    ax.plot(data_copy.index, data_copy[column_name], label=name)

    # Добавляем подписи
    if title:
        ax.set_title(str(name))
        
    if ylim:
        ax.set_ylim((0,ylim))
        
    if xlabel:
        ax.set_xlabel("λ поглощения, нм")
        
    if ylabel:
        
        if norm_by_TOC:
            ax.set_ylabel("SUVA, $см^{-1}*мг^{-1}*л$")
            
        else:
            
            ax.set_ylabel("Оптическая плотность")

    return ax

def read_csv_uv(path: str,
            sep: str = "",
            index_col: int = 0,
            ignore_name: bool = False,
            baseline: bool = False,
            spectra_type: str|None = None) -> DataFrame|None:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.csv").
    :param sep: разделитель в строчном виде (example: ",").
    :param index_col: номер столбца, который считается индексом таблицы.
    :param ignore_name: параметр, при включении которого игнорируются встроенные классы и подклассы
    :param baseline: параметр, при включении которого производится базовая рекалибровка
    :return: DataFrame: Таблица, в котором индексами строк являются длины волн.
    """
    file_type = ut.check_file_type(path)

    if sep == "" and file_type == "csv_type" :
        sep = ut.check_sep(path)

    try:
        data = pd.read_csv(path, sep=sep, index_col=index_col)

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Файл пуст: {path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")

    data = standart_uv_formatting(data)
    data.sort_index(inplace=True)
    
    if spectra_type:
        if check_uv_spectra_type_by_path(path) != spectra_type:
            
            return None
    
    else:
        spectra_type = check_uv_spectra_type_by_path(path)
            
    data.attrs['spectra_type'] = spectra_type
    data.rename(columns={data.columns[0]: spectra_type}, inplace=True)

    name = ut.extract_name_from_path(path)
    data = ut.attributting_order(data, ignore_name=ignore_name, name=name)

    if 'recall' not in data.attrs:
    
        if baseline and (data.attrs['spectra_type'] == "absorption"):
            data = base_recall_uv(data)
    
    data.attrs['path'] = path
    
    return data
    
def read_excel_uv(path: str,
        index_col: int = 0,
        ignore_name: bool = False,
        baseline: bool = True,
        spectra_type: str|None = None,
        debug:bool = False) -> DataFrame | List[DataFrame]:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.xlsx").
    :param index_col: номер столбца, который считается индексом таблицы.
    :param ignore_name: параметр, при включении которого игнорируются встроенные классы и подклассы
    :param baseline: параметр, при включении которого производится базовая рекалибровка
    :return: DataFrame: Таблица, в котором индексами строк являются длины волн.
    """
    
    file_type = ut.check_file_type(path)
    
    if debug:
        print(file_type)

    try:
        raw_data = pd.read_excel(path, index_col=index_col, sheet_name=None)

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Файл пуст: {path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")

    if file_type == "excel_single":

        data = pd.read_excel(path, index_col= index_col)

        data = standart_uv_formatting(data)
        data.sort_index(inplace=True)
        
        if spectra_type:
            if check_uv_spectra_type_by_path(path) != spectra_type:
            
                raise KeyError(f"Ошибка приписывания класса для образца{data.attrs['name']}") 
        else:
            spectra_type = check_uv_spectra_type_by_path(path)
            
        data.attrs['spectra_type'] = spectra_type
        data.rename(columns={data.columns[0]: spectra_type}, inplace=True)

        name = ut.extract_name_from_path(path)

        data = ut.attributting_order(data, ignore_name=ignore_name, name=name)

        if baseline and (data.attrs['spectra_type'] == "absorption"):
            data = base_recall_uv(data)
            
        data.attrs['path'] = path

        return data
    
    elif file_type == "excel_many":
 
        data_list = []

        xlsx = pd.ExcelFile(path)
        list_sheet_names = xlsx.sheet_names
        i = 0

        for name, data in raw_data.items():
            
            data = standart_uv_formatting(data)
            data.sort_index(inplace=True)
            
            if spectra_type:
                if check_uv_spectra_type_by_path(path) != spectra_type:
                
                    raise KeyError(f"Ошибка приписывания класса для образца{data.attrs['name']}")
            else:
                spectra_type = check_uv_spectra_type_by_path(path)

            sample_name = str(list_sheet_names[i])

            if debug:
                print(sample_name)

            data = ut.attributting_order(data, ignore_name=ignore_name, name=sample_name)
            
            data.attrs['spectra_type'] = spectra_type
            data.rename(columns={data.columns[0]: spectra_type}, inplace=True)
            
            if baseline and (data.attrs['spectra_type'] == "absorption"):
                data = base_recall_uv(data)

            data.attrs['path'] = path
            
            data_list.append(data)

            i += 1
        
        data_list:List[DataFrame]

        return data_list
    
    else:
        raise KeyboardInterrupt("Нет доступного типа!")

def standart_uv_formatting(data: DataFrame)-> DataFrame:
    """
    :param data: DataFrame, сырой уф спектр
    :return: Отформатированный уф спектр
    Функция заменяет строковые данные на числовые в уф спектре
    """
    data_copy = data.copy()

    if all(isinstance(x, str) for x in data_copy[data_copy.columns[0]]):
        data_copy = data_copy.replace(',', '.', regex=True)
        data_copy = data_copy.replace(' ', '', regex=True)
    
    if all(isinstance(x, str) for x in data_copy.index):
        data_copy.index = data_copy.index.str.replace(',', '.')
        data_copy.index = data_copy.index.str.replace(' ', '')

    data_copy = data_copy.astype(float)
    data_copy.index = data_copy.index.astype(float)

    return data_copy

def check_uv_spectra_type_by_path(path: str):
    
    name = ut.extract_name_from_path(path)
    
    if "R" in name:
        spectra_type = "reflection"
    
    else:
        spectra_type = "absorption"
        
    return spectra_type
        
def add_toc_to_spectra(spectra: List[pd.DataFrame], 
                            toc_table: pd.DataFrame, 
                            name_col: str = 'Sample', 
                            toc_col: str = 'C_org',
                            verbose: bool = True) -> List[pd.DataFrame]:
    """
    Добавляет параметр 'TOC' в attrs каждого спектра и удаляет спектры без TOC
    
    Args:
        spectra: Список DataFrame спектров
        toc_table: DataFrame с таблицей соответствия имен и значений TOC
        name_col: Название столбца с именами в toc_table (по умолчанию 'Sample')
        toc_col: Название столбца со значениями TOC в toc_table (по умолчанию 'C_org')
        verbose: Выводить ли информацию об удаленных спектрах
        
    Returns:
        List[pd.DataFrame]: Список спектров с добавленным параметром 'TOC' в attrs
                            (спектры без TOC удалены)
    """
    
            # Создаем словарь для быстрого поиска TOC по имени
    toc_dict = dict(zip(toc_table[name_col], toc_table[toc_col]))
    
    updated_spectra = []
    removed_count = 0
    
    for spectrum in spectra:
        # Создаем копию DataFrame
        spectrum_copy = spectrum.copy()
        
        # Проверяем наличие атрибута 'name'
        if hasattr(spectrum, 'attrs') and 'name' in spectrum.attrs:
            spectrum_name = spectrum.attrs['name']
            
            # Ищем соответствие в таблице TOC
            if spectrum_name in toc_dict:
                # Создаем копию attrs и добавляем TOC
                new_attrs = spectrum.attrs.copy()
                new_attrs['TOC'] = toc_dict[spectrum_name]
                spectrum_copy.attrs = new_attrs
                updated_spectra.append(spectrum_copy)
            else:
                removed_count += 1
                if verbose:
                    print(f"Удален спектр: '{spectrum_name}' - TOC не найден")
        else:
            removed_count += 1
            if verbose:
                print(f"Удален спектр: отсутствует атрибут 'name'")
    
    if verbose:
        print(f"Обработка завершена. Удалено спектров: {removed_count}, осталось: {len(updated_spectra)}")
    
    return updated_spectra

def smooth_uv_spectrum(df, window_size=5, threshold_factor=2.0, min_peak_height=0.0001):
    """
    Находит и сглаживает участки с резкими перепадами интенсивности в УФ спектре
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с колонками 'nm' (длина волны) и 'absorption' (интенсивность)
    window_size : int
        Размер окна для скользящего среднего (по умолчанию 5)
    threshold_factor : float
        Коэффициент для определения порога перепадов (по умолчанию 2.0)
    min_peak_height : float
        Минимальная высота пика для обнаружения (по умолчанию 0.01)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame с исходными данными и дополнительной колонкой 'absorption_smooth'
        содержащей сглаженный спектр
    """
    
    # Создаем копию DataFrame чтобы не изменять исходные данные
    result_df = df.copy()
    intensity = df[df.columns[0]].values
    
    # 1. Вычисляем первую производную (градиент) для обнаружения резких изменений
    gradient = np.gradient(intensity)
    
    # 2. Находим точки с резкими изменениями интенсивности
    threshold = threshold_factor * np.std(gradient)
    
    # Находим индексы где производная превышает порог (резкие изменения)
    sharp_changes = np.where(np.abs(gradient) > threshold)[0]
    
    # 3. Обнаруживаем пики в исходном сигнале для идентификации проблемных зон
    peaks, _ = signal.find_peaks(intensity, height=min_peak_height, distance=10)
    valleys, _ = signal.find_peaks(-intensity, height=-max(intensity), distance=10)
    
    # Объединяем все проблемные точки
    problematic_indices = np.unique(np.concatenate([sharp_changes, peaks, valleys]))
    
    # 4. Создаем маску для участков с перепадами
    mask = np.zeros(len(intensity), dtype=bool)
    if len(problematic_indices) > 0:
        # Расширяем область вокруг проблемных точек
        for idx in problematic_indices:
            start = max(0, idx - window_size)
            end = min(len(intensity), idx + window_size + 1)
            mask[start:end] = True
    
    # 5. Применяем сглаживание только к проблемным участкам
    smoothed = intensity.copy()
    
    if np.any(mask):
        # Для проблемных участков применяем сглаживание
        smoothed_mask = uniform_filter1d(intensity, size=window_size)
        smoothed[mask] = smoothed_mask[mask]
        
        # Дополнительное сглаживание для очень резких перепадов
        extreme_changes = np.where(np.abs(gradient) > 2 * threshold)[0]
        if len(extreme_changes) > 0:
            for idx in extreme_changes:
                start = max(0, idx - window_size * 2)
                end = min(len(intensity), idx + window_size * 2 + 1)
                # Используем более сильное сглаживание для экстремальных перепадов
                extreme_smooth = uniform_filter1d(intensity[start:end], size=window_size * 2)
                smoothed[start:end] = extreme_smooth
    
    # 6. Добавляем сглаженные данные в DataFrame
    result_df[result_df.columns[0]] = smoothed
    
    return result_df