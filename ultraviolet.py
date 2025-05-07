import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional
from matplotlib.axes import Axes


def check_recall_flag(data: DataFrame) -> bool:
    if 'debug' not in data.attrs:
        if not data.attrs['recall']:
            raise ValueError("Спектр должен быть откалиброван")
    else:
        if not data.attrs['debug']:
            if not data.attrs['recall']:
                raise ValueError("Спектр должен быть откалиброван")
    return True


def base_recall_uv(data: DataFrame) -> DataFrame:
    data_copy = data.copy()
    min_value = data_copy.min()
    data_copy.attrs['recall'] = True

    return data_copy + 1.001 * abs(data_copy)


def e2_e3(data: DataFrame) -> float:
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")
    series = pd.Series(data.index, index=data.index)
    index_265 = series.sub(265).abs().idxmin()
    index_365 = series.sub(365).abs().idxmin()
    uv_param = data.loc[index_265] / data.loc[index_365]

    return uv_param.iloc[0]


def e4_e6(data: DataFrame) -> float:
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")

    series = pd.Series(data.index, index=data.index)
    index_465 = series.sub(465).abs().idxmin()
    index_665 = series.sub(665).abs().idxmin()
    uv_param = data.loc[index_465] / data.loc[index_665]

    return uv_param.iloc[0]


def epsilon(data: DataFrame,
            wave: int = 254) -> float:
    if not check_recall_flag(data):
        raise ValueError("Ошибка проверки статуса калибровки")
    series = pd.Series(data.index, index=data.index)
    index_254 = series.sub(254).abs().idxmin()
    uv_param = data.loc[index_254].iloc[0]

    return uv_param


def suva(data: DataFrame) -> float:
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
            ax: Optional[plt.axes] = None) -> Axes:
    data_copy = data.copy()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(data_copy.index, data_copy["intensity"])
    if title:
        ax.set_title(f"{data_copy.attrs['name']}")
    if xlabel:
        ax.set_xlabel("λ поглощения, нм")
    if ylabel:
        ax.set_ylabel("Интенсивность")
    return ax
