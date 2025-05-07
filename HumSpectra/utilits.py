import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
import re
import scipy.interpolate
plt.rcParams['axes.grid'] = False


def extract_and_combine_digits_re(text: str) -> int:
    """Извлекает все цифры из строки и объединяет их в одно целое число.

    :param text: Строка для поиска цифр.

    :return Целое число, образованное из цифр в строке. Возвращает 0, если цифр нет.
    """
    digits = re.findall(r'\d', text)  # Находим все цифры в строке
    if digits:
        combined_number_str = ''.join(digits)  # Объединяем цифры в строку
        try:
            return int(combined_number_str)  # Преобразуем строку в целое число
        except ValueError:
            return 0  # Возвращаем 0, если строка не может быть преобразована в число
    else:
        return 0  # Возвращаем 0, если цифры не найдены


def extract_name_from_path(file_path: str) -> str:
    """Извлекает имя файла (без расширения) из пути.

    Поддерживает разные разделители каталогов (/, \, //).

    :param file_path: Полный путь к файлу.

    :return: file_name: Имя файла без расширения.  Возвращает пустую строку, если путь недопустимый.
    """
    try:
        # 1. Нормализация пути: Замена двойных слешей на одинарные,
        #    а также приведение к абсолютному пути (это нужно не всегда,
        #    но может помочь избежать проблем).
        normalized_path = os.path.abspath(os.path.normpath(file_path))

        # 2. Извлечение имени файла (вместе с расширением).
        file_name_with_extension = os.path.basename(normalized_path)

        # 3. Разделение имени файла и расширения.
        file_name, file_extension = os.path.splitext(file_name_with_extension)

        return file_name
    except Exception as e:
        print(f"Ошибка при обработке пути {file_path}: {e}")
        return ""


def extract_class_from_name(file_name: str) -> str:
    """Извлекает класс образца из имени.

       :param file_name: Имя образца.

       :return: sample_class: Класс образца.
       Функция определяет класс по образца по имени. Определяемые классы: Coal(угольные), Soil(почвенные), Peat(торфяные), L(лигногуматы и лигносульфонаты)
       , ADOM(надшламовые воды) 
       """
    sample_class = "SampleClassError"
    str_name = file_name.replace(" ", "-")
    if "ADOM" in str_name:
        sample_class = "ADOM"
    else:
        str_class = str_name.split(sep="-")[0]
        symbol_class = str_class[0]
        if "C" == symbol_class:
            sample_class = "Coal"
        elif "L" == symbol_class:
            sample_class = "Lg"
        elif "P" == symbol_class:
            sample_class = "Peat"
        elif "S" == symbol_class:
            sample_class = "Soil"
        else:
            raise ValueError("Имя образца не соответствует ни одному представленному классу")
    return sample_class


def extract_subclass_from_name(file_name: str) -> str:
    """Извлекает подкласс образца из имени.

       :param file_name: Имя образца.

       :return: sample_subclass: Класс образца.
       Функция определяет подкласс по образца по имени. Определяемые подклассы: номера карты накопителя для надшламовых вод, 
       Lg(лигногуматы), Lst(лигносульфонаты), (C,P,S)HA (гуминовые кислоты), (C,P,S)HA (фульвокислоты), (C,P,S)HF (нефракционированные гуминовые вещества)
       """
    sample_subclass = "SampleSubClassError"
    str_name = file_name.replace(" ", "-")
    if "ADOM" in str_name:
        str_subclass = str_name.split(sep="-")[1]
        sample_subclass = "Storage " + str(extract_and_combine_digits_re(str_subclass))
    elif "L" == str_name[0]:
        if "G" == str_name[1]:
            sample_subclass = "Lg"
        else:
            sample_subclass = "Lst"
    else:
        sample_subclass = str_name.split(sep="-")[0]

    return sample_subclass


def read_fluo_3d(path: str,
                 sep: str = ",",
                 index_col: int = 0) -> DataFrame:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.csv").
    :param sep: разделитель в строчном виде (example: ",").
    :param index_col: номер столбца, который считается индексом таблицы.
    :return: DataFrame: Таблица, в котором индексами строк являются длины волн испускания, а именами столбцов - длины волн
            возбуждения. Таблица имеет метаданные - имя образца, класс, подкласс
    """

    data = pd.read_csv(path, sep=sep, index_col=index_col)
    data.drop("nm", inplace=True)
    data = data.astype("float64")
    name = extract_name_from_path(path)
    data.columns = data.columns.astype(int)
    data.index = data.index.astype(int)
    data.attrs['name'] = name
    data.attrs['class'] = extract_class_from_name(name)
    data.attrs['subclass'] = extract_subclass_from_name(name)

    return data


def read_uv(path: str,
            sep: str = ",",
            index_col: int = 0) -> DataFrame:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.csv").
    :param sep: разделитель в строчном виде (example: ",").
    :param index_col: номер столбца, который считается индексом таблицы.
    :return: DataFrame: Таблица, в котором индексами строк являются длины волн.
    """
    data = pd.read_csv(path, sep=sep, index_col=index_col)
    data = data.astype("float64")
    name = extract_name_from_path(path)
    data.index = data.index.astype(float)
    data.sort_index(inplace=True)
    data.attrs['name'] = name
    data.attrs['class'] = extract_class_from_name(name)
    data.attrs['subclass'] = extract_subclass_from_name(name)
    data.attrs['recall'] = False

    return data


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


def cut_raman_spline(em_wvs: ndarray,
                     fl: ndarray,
                     exc_wv: int,
                     raman_freq: float = 3430.,
                     raman_hwhm: int = 15,
                     points_in_spline: int = 10) -> ndarray:
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
    return res[-1](em_wvs)
