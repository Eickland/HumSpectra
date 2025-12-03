import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union, List
from matplotlib.axes import Axes
from math import ceil, sqrt
from scipy.optimize import curve_fit 

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

def ratio_descriptor_uv(data: DataFrame,
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

def integral_ratio_uv(data: DataFrame,
                      low_wv_left: float,
                      low_wv_right: float,
                      high_wv_left: float,
                      high_wv_right: float,
          debug: bool=True) -> float:

    if not debug:
        if not check_recall_flag(data):
            raise ValueError("Ошибка проверки статуса калибровки")
    
    series = pd.Series(data.index, index=data.index)
    
    index_low_left = series.sub(low_wv_left).abs().idxmin()
    index_low_right = series.sub(low_wv_left).abs().idxmin()
    index_high_left = series.sub(low_wv_left).abs().idxmin()
    index_high_right = series.sub(low_wv_left).abs().idxmin()
    
    high = np.trapezoid(
        data.loc[index_high_left:index_high_right]
    )
    low = np.trapezoid(
        data.loc[index_low_left:index_low_right]
    )
    
    uv_param = float(high / low)

    return uv_param

def e2_e3(data: DataFrame,
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

def e4_e6(data: DataFrame,
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

def single_density(data: DataFrame,
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

    a_254 = single_density(data, 254)
    uv_param = a_254 / data.attrs['TOC']

    return uv_param

def lambda_UV(data: DataFrame,
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
            ax.set_ylabel("Интенсивность")

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
    
def plot_uv_spectra_by_subclass(spectra_list: List[pd.DataFrame],
                            plot_func, 
                           figsize_multiplier: int = 4,
                           sharey: bool = True, 
                           sharex: bool = True,
                           norm_by_TOC: bool = False,
                           show_titles: bool = True,
                           show_xlabels: bool = True,
                           show_ylabels: bool = True) -> None:
    """
    Отображает спектры по подклассам на отдельных графиках с использованием функции.
    
    Parameters:
    -----------
    spectra_list : list of DataFrame
        Список спектров (DataFrame) с атрибутами 'name' и 'subclass' в .attrs
    figsize_multiplier : int, default=4
        Множитель для размера фигуры
    sharey : bool, default=True
        Общий масштаб по оси Y для всех подграфиков
    sharex : bool, default=True
        Общий масштаб по оси X для всех подграфиков
    norm_by_TOC : bool, default=False
        Нормализовать по TOC (передается в plot_uv)
    show_titles : bool, default=True
        Показывать заголовки графиков
    show_xlabels : bool, default=True
        Показывать подписи оси X
    show_ylabels : bool, default=True
        Показывать подписи оси Y
    """
    
    # Группируем спектры по подклассам
    spectra_by_subclass = {}
    for spectrum in spectra_list:
        subclass = spectrum.attrs.get('subclass', 'Unknown')
        name = spectrum.attrs.get('name', 'Unnamed')
        
        if subclass not in spectra_by_subclass:
            spectra_by_subclass[subclass] = []
        
        spectra_by_subclass[subclass].append({
            'data': spectrum,
            'name': name
        })
    
    # Получаем список подклассов
    subclasses = list(spectra_by_subclass.keys())
    n_subclasses = len(subclasses)
    
    if n_subclasses == 0:
        print("Нет спектров для отображения")
        return
    
    # Определяем оптимальную размерность subplots
    n_cols = ceil(sqrt(n_subclasses))
    n_rows = ceil(n_subclasses / n_cols)
    
    # Создаем фигуру с оптимальным размером
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * figsize_multiplier, n_rows * figsize_multiplier),
                            sharey=sharey, sharex=sharex,
                            squeeze=False)
    
    # Выравниваем axes в плоский массив для удобства итерации
    axes_flat = axes.flatten()
    
    # Отрисовываем спектры для каждого подкласса с использованием plot_uv
    for idx, subclass in enumerate(subclasses):
        ax = axes_flat[idx]
        spectra_data = spectra_by_subclass[subclass]
        
        # Рисуем все спектры этого подкласса
        for spectrum_info in spectra_data:
            spectrum_df = spectrum_info['data']
            name = spectrum_info['name']
            
            # Используем функцию plot для отрисовки каждого спектра
            plot_func(data=spectrum_df,
                   xlabel=False,  # Убираем xlabel для отдельных графиков
                   ylabel=False,  # Убираем ylabel для отдельных графиков
                   title=False,   # Убираем title для отдельных графиков
                   ax=ax,
                   name=name)
        
        # Добавляем заголовок подкласса
        if show_titles:
            ax.set_title(f'Подкласс: {subclass}\n(спектров: {len(spectra_data)})', 
                        fontsize=12, fontweight='bold')
        
        # Добавляем подписи осей только если нужно
        if show_xlabels:
            ax.set_xlabel("λ поглощения, нм")
        if show_ylabels:
            if norm_by_TOC:
                ax.set_ylabel("SUVA, $см^{-1}*мг^{-1}*л$")
            else:
                ax.set_ylabel("Интенсивность")
        
        ax.grid(True, alpha=0.3)
        
        # Настраиваем легенду в зависимости от количества спектров
        if len(spectra_data) <= 8:
            ax.legend(fontsize=8)
        else:
            # Для большого количества спектров уменьшаем шрифт или выносим легенду
            ax.legend(fontsize=6, loc='upper right')
    
    # Скрываем пустые subplots
    for idx in range(len(subclasses), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Выводим статистику
    print(f"Всего подклассов: {n_subclasses}")
    for subclass, spectra in spectra_by_subclass.items():
        print(f"  {subclass}: {len(spectra)} спектров")
        
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