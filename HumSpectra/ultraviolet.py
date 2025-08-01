import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional
from matplotlib.axes import Axes


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


def e2_e3(data: DataFrame) -> float:
    """
    :param data: DataFrame, уф спектр
    :return: uv_param: float, значение параметра E2/E3
    Функция проверяет наличие рекалибровки и рассчитывает отношение оптической плотности при длине волны 265 к 365 нм.
    """
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")
    series = pd.Series(data.index, index=data.index)
    index_265 = series.sub(265).abs().idxmin()
    index_365 = series.sub(365).abs().idxmin()
    uv_param = data.loc[index_265] / data.loc[index_365]

    return uv_param.iloc[0]


def e4_e6(data: DataFrame) -> float:
    """
    :param data: DataFrame, уф спектр
    :return: uv_param: float, значение параметра E4/E6
    Функция проверяет наличие рекалибровки и рассчитывает отношение оптической плотности при длине волны 465 к 665 нм.
    """
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")
    series = pd.Series(data.index, index=data.index)
    index_465 = series.sub(465).abs().idxmin()
    index_665 = series.sub(665).abs().idxmin()
    uv_param = data.loc[index_465] / data.loc[index_665]

    return uv_param.iloc[0]


def epsilon(data: DataFrame,
            wave: int = 254) -> float:
    """
    :param data: DataFrame, уф спектр
    :param wave: int, длина волны, по которой ищется оптическая плотность
    :return: uv_param: float, значение оптической плотности
    Функция проверяет наличие рекалибровки и рассчитывает оптическую плотность при заданной длине волны
    """
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")
    series = pd.Series(data.index, index=data.index)
    index_254 = series.sub(wave).abs().idxmin()
    uv_param = data.loc[index_254].iloc[0]

    return uv_param


def suva(data: DataFrame) -> float:
    """
    :param data: DataFrame, уф спектр
    :return: uv_param: float, значение параметра SUVA 254
    Функция проверяет наличие рекалибровки и рассчитывает параметр SUVA 254, для функции необходимо наличие в метаданных таблицы значение TOC
    """
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")

    if "TOC" not in data.attrs:
        raise KeyError("В метаданных таблицы должно быть значения содержания органического углерода")

    a_254 = epsilon(data)
    uv_param = a_254 / data.attrs['TOC']

    return uv_param


def lambda_UV(data: DataFrame,
              short_wave: int = 450,
              long_wave: int = 550) -> float:
    """
    :param data: DataFrame, уф спектр
    :param short_wave: int, длина волны, с которого будет производится аппроксимация
    :param long_wave: int, длина волны, до которой будет производится аппроксимация
    :return: uv_param: float, значение дескриптора лямбда
    Функция проверяет наличие рекалибровки и рассчитывает параметр лямбда при длине волны от 450 до 550 нм
    """
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")
    series = pd.Series(data.index, index=data.index)
    index_short = series.sub(short_wave).abs().idxmin()
    index_long = series.sub(long_wave).abs().idxmin()
    lambda_array = data.loc[index_short:index_long]["intensity"].to_numpy()
    p, *rest = np.polyfit(lambda_array, np.log(lambda_array), 1, full=True)
    a, b = p
    uv_param = 1 / abs(a)

    return uv_param


def plot_uv(data: DataFrame,
            xlabel: bool = True,
            ylabel: bool = True,
            title: bool = True,
            norm_by_TOC: bool = False,
            ax: Optional[plt.axes] = None,
            name:str = None) -> Axes:
    """
    :param data: DataFrame, уф спектр
    :return: ax: Axes, ось графика matplotlib.pyplot
    Функция возвращает график 2D уф-спетра
    """

    

    data_copy = data.copy()

    if "name" in data_copy.attrs:
        name = data_copy.attrs["name"]
    else:
        name = "uv_spectra"
    if norm_by_TOC:
        if "TOC" not in data_copy.attrs:
            raise KeyError("В метаданных таблицы должно быть значения содержания органического углерода")
        data_copy = data_copy/data_copy.attrs['TOC']
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(data_copy.index, data_copy["intensity"], label = name)
    if title:
            ax.set_title(name)
    if xlabel:
        ax.set_xlabel("λ поглощения, нм")
    if ylabel:
        if norm_by_TOC:
            ax.set_ylabel("SUVA, $см^{-1}*мг^{-1}*л$")
        else:
            ax.set_ylabel("Интенсивность")

    return ax
