import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional
from matplotlib.axes import Axes
from typing import Union

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


def e2_e3(data: DataFrame,
          debug: bool=False) -> float:
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


def e4_e6(data: DataFrame,
          debug: bool=False) -> float:
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


def epsilon(data: DataFrame,
            wave: int = 254,
            debug: bool=False) -> float:
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
    index_254 = series.sub(wave).abs().idxmin()
    uv_param = data.loc[index_254].iloc[0].item()

    return uv_param


def suva(data: DataFrame,
         debug: bool=False) -> float:
    """
    :param data: DataFrame, уф спектр
    :return: uv_param: float, значение параметра SUVA 254
    Функция проверяет наличие рекалибровки и рассчитывает параметр SUVA 254, для функции необходимо наличие в метаданных таблицы значение TOC
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    

    if "TOC" not in data.attrs:
        raise KeyError("В метаданных таблицы должно быть значения содержания органического углерода")

    a_254 = epsilon(data)
    uv_param = a_254 / data.attrs['TOC']

    return uv_param


def lambda_UV(data: DataFrame,
              short_wave: int = 450,
              long_wave: int = 550,
              debug: bool=False) -> float:
    """
    :param data: DataFrame, уф спектр
    :param short_wave: int, длина волны, с которого будет производится аппроксимация
    :param long_wave: int, длина волны, до которой будет производится аппроксимация
    :return: uv_param: float, значение дескриптора лямбда
    Функция проверяет наличие рекалибровки и рассчитывает параметр лямбда при длине волны от 450 до 550 нм
    """
    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    index_short = series.sub(short_wave).abs().idxmin()
    index_long = series.sub(long_wave).abs().idxmin()
    lambda_array = data.loc[index_short:index_long][data.columns[0]].to_numpy()
    p, *rest = np.polyfit(lambda_array, np.log(lambda_array), 1, full=True)
    a, b = p
    uv_param = 1 / abs(a)

    return uv_param


def plot_uv(data: DataFrame,
            xlabel: bool = True,
            ylabel: bool = True,
            title: bool = True,
            norm_by_TOC: bool = False,
            ax: Union[Axes, None] = None,
            name:Optional[str] = None) -> Axes:
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

    ax.plot(data_copy.index, data_copy[data_copy.columns[0]], label = name)

    if title:
            ax.set_title(str(name))
    if xlabel:
        ax.set_xlabel("λ поглощения, нм")
    if ylabel:
        if norm_by_TOC:
            ax.set_ylabel("SUVA, $см^{-1}*мг^{-1}*л$")
        else:
            ax.set_ylabel("Интенсивность")

    return ax

def read_csv_uv(path: str,
            sep: str = "",
            index_col: int = 0,
            ignore_name: bool = False,
            baseline: bool = True,
            spectra_type: str|None = None) -> DataFrame:
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
    

    data = standart_uv_formatting(data,spectra_type=spectra_type)
    data.sort_index(inplace=True)

    name = ut.extract_name_from_path(path)
    data = ut.attributting_order(data, ignore_name=ignore_name, name=name)

    if baseline and (data.attrs['spectra_type'] == "absorption"):
        data = base_recall_uv(data)
    

    return data
    
def read_excel_uv(path: str,
        sep: str = "",
        index_col: int = 0,
        ignore_name: bool = False,
        baseline: bool = True,
        spectra_type: str|None = None) -> DataFrame | list:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.xlsx").
    :param sep: разделитель в строчном виде (example: ",").
    :param index_col: номер столбца, который считается индексом таблицы.
    :param ignore_name: параметр, при включении которого игнорируются встроенные классы и подклассы
    :param baseline: параметр, при включении которого производится базовая рекалибровка
    :return: DataFrame: Таблица, в котором индексами строк являются длины волн.
    """
    file_type = ut.check_file_type(path)

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

        data = standart_uv_formatting(data,spectra_type=spectra_type)
        data.sort_index(inplace=True)

        name = ut.extract_name_from_path(path)
        data = ut.attributting_order(data, ignore_name=ignore_name, name=name)

        if baseline and (data.attrs['spectra_type'] == "absorption"):
            data = base_recall_uv(data)


        return data
    
    elif file_type == "excel_many":
 
        data_list = []

        xlsx = pd.ExcelFile(path)
        list_sheet_names = xlsx.sheet_names
        i = 0

        for name, data in raw_data.items():
            
            data.set_index(data.columns[0],inplace=True)
            data = standart_uv_formatting(data,spectra_type=spectra_type)
            data.sort_index(inplace=True)

            name = str(list_sheet_names[i])
            data = ut.attributting_order(data, ignore_name=ignore_name, name=name)

            if baseline and (data.attrs['spectra_type'] == "absorption"):
                data = base_recall_uv(data)

            data_list.append(data)

            i += 1

        return data_list
    
    else:
        return pd.DataFrame()

def standart_uv_formatting(data: DataFrame,
                           spectra_type: str|None = None)-> DataFrame:
    """
    :param data: DataFrame, сырой уф спектр
    :return: Отформатированный уф спектр
    Функция заменяет строковые данные на числовые в уф спектре
    """
    data_copy = data.copy()

    spectra_type = check_uv_spectra_type(data_copy,spectra_type=spectra_type)
    data_copy.attrs['spectra_type'] = spectra_type

    data_copy.rename(columns={data_copy.columns[0]: spectra_type}, inplace=True)

    if data_copy[spectra_type].dtype == str:
        data_copy[spectra_type]=data_copy[spectra_type].str.replace(',','.')
    data_copy = data_copy.astype("float64")

    if data_copy[spectra_type].dtype == str:
        data_copy.index = data_copy.index.str.replace(',','.')
    data_copy.index = data_copy.index.astype(float)

    return data_copy

def check_uv_spectra_type(data: DataFrame,
                          spectra_type: str|None = None)-> str:
    """
    :param data: DataFrame, уф спектр
    :return: Тип уф спектра
    Функция определяет тип уф спектра - поглощение, зеркальное отражение
    """
    column_name = data.columns[0]

    if spectra_type:
        uv_spectra_type = spectra_type

    elif "Abs" in column_name:
        uv_spectra_type = "absorption"

    elif "R%" in column_name:
        uv_spectra_type = "reflection"

    else:
        raise KeyError("Недопустимый тип спектра")

    return uv_spectra_type