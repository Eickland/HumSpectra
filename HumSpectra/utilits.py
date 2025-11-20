import os
import pandas as pd
from pandas import DataFrame
import re
from transliterate import translit
import numpy as np
from scipy.spatial import ConvexHull
from scipy import stats
from typing import Any, Dict
import warnings
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


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
    r"""Извлекает имя файла (без расширения) из пути.

    Поддерживает разные разделители каталогов (/, \, //).

    :param file_path: Полный путь к файлу.

    :return: file_name: Имя файла без расширения.  Возвращает пустую строку, если путь недопустимый.
    """
    try:
        # 1. Нормализация пути: Замена двойных слешей на одинарные.
        normalized_path = os.path.abspath(os.path.normpath(file_path))
        
        folder = normalized_path.split(sep="\\")[-3]
                    
        if folder == "2024":
            date = "2024"
        
        elif folder == "2025":
            date = "2025"
            
        else:
            date = ""

        # 2. Извлечение имени файла (вместе с расширением).
        file_name_with_extension = os.path.basename(normalized_path)

        # 3. Разделение имени файла и расширения.
        file_name, file_extension = os.path.splitext(file_name_with_extension)

        # 4. Земена русских символов на английские транслитом.
        file_name = translit(file_name, 'ru', reversed=True)
        
        if date == "":
            return file_name
        
        else:
            return file_name + "-" + date
    
    except Exception as e:
        print(f"Ошибка при обработке пути {file_path}: {e}")
        return ""

def extract_class_from_name(file_name: str) -> str:
    """Извлекает класс образца из имени.

       :param file_name: Имя образца.

       :return: sample_class: Класс образца.
       Функция определяет класс по образца по имени. Определяемые классы: Coal(угольные), Soil(почвенные), Peat(торфяные), L(лигногуматы и лигносульфонаты), 
       ADOM(надшламовые воды) 
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
        elif "K" == symbol_class:
            sample_class = "ADOM"
        elif "B" == symbol_class:
            sample_class = "ADOM"
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
    sample_class = extract_class_from_name(file_name=file_name)

    str_name = file_name.replace(" ", "-")

    if "ADOM" == sample_class:
        
        split_name = str_name.split(sep="-")

        if "B" in split_name[1]  or "В" in split_name[1]:
            sample_subclass = "Baikal"
            
        else:

            if len(split_name) == 2:
                str_name = split_name[0][:1]+"-"+split_name[0][1:]

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

def check_sep(path: str) -> str:
    """
    :param path: путь к файлу в строчном виде
    :return: sep: разделитель в строчном виде (example: ",")
    Функция определяет разделитель столбцов в исходном спектре - запятая или точка с запятой
    """

    try:
        with open(path, 'r') as f:
            first_line = f.readline()

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")

    if first_line.count(';') > first_line.count(','):
        return ';'
    
    else:
        return ','

def check_file_extension(path: str) -> str:
    """
    :param path: путь к файлу в строчном виде
    :return: file_extension: строка в котором расширение файла
    Функция определяет расширение файла - csv, txt, excel файл и возвращает расширение.
    """

    extension = path.split(sep=".")[-1] 

    return extension   

def check_file_type(path: str) -> str:
    """
    :param path: путь к файлу в строчном виде
    :return: file_type: строка-кодировка типа файла
    Функция определяет тип файла - csv, txt, excel файл и возвращает кодировку
    """

    ext = check_file_extension(path)

    if ext in ["txt","csv"]:
        file_type = "csv_type"
    
    elif ext == "xlsx":
        
        xlsx = pd.ExcelFile(path)
        sheet_num = len(xlsx.sheet_names)

        if sheet_num == 1:
            file_type = "excel_single"

        elif sheet_num > 1:
            file_type = "excel_many"

        else:
            raise ValueError("Неизвестная ошибка")
    
    else:
        raise ValueError("Тип файла не поддерживается")
    
    return file_type

def attributting_order(data: DataFrame,
                       ignore_name: bool,
                       name: str
                       )-> DataFrame:
    """
    :param data: DataFrame, сырой уф спектр
    :param ignore_name: параметр, при включении которого игнорируются встроенные классы и подклассы
    :param name: имя спектра
    :return: Отформатированный уф спектр
    Функция приписывает имя, класс и подкласс (если не игнорируется) спектру
    """

    data_copy = data.copy()

    if not ignore_name:

        data_copy.attrs['name'] = name
        data_copy.attrs['class'] = extract_class_from_name(name)
        data_copy.attrs['subclass'] = extract_subclass_from_name(name)

    else:

        data_copy.attrs['name'] = name

    return data_copy

def load_spectra_data(folder, reader_func, **kwargs)-> list[DataFrame]:
    
    """Загружает все спектры из папки"""
    
    spectra_list = [reader_func(str(path), **kwargs) for path in folder.rglob('*.csv')]
    
    return [x for x in spectra_list if x is not None]

def spectra_to_df(spectra_list, metrics, class_filter="ADOM"):

    df = pd.DataFrame([{

        "Sample": s.attrs['name'],
        "Class": s.attrs['class'], 
        "Subclass": s.attrs['subclass'],

        **{name: func(s) for name, func in metrics.items()}
    } for s in spectra_list])

    return df[df["Class"] == class_filter] if class_filter else df

def analyze_geographical(data):
    """
    Анализирует географические координаты и определяет крайние точки.
    
    Parameters:
    -----------
    data : pandas.DataFrame или pandas.Series
        Если DataFrame, должен содержать столбцы с координатами.
        Если Series, должен содержать координаты в формате (lat, lon) или два столбца.
    
    Returns:
    --------
    dict: Словарь с информацией о крайних точках
    """
    

    df = data.copy()

    
    # Проверяем наличие столбцов с координатами
    lat_col = None
    lon_col = None
    
    # Ищем столбцы с координатами по разным возможным названиям
    possible_lat_names = ['latitude', 'lat', 'y', 'широта']
    possible_lon_names = ['longitude', 'lon', 'lng', 'x', 'долгота']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_lat_names):
            lat_col = col
        if any(name in col_lower for name in possible_lon_names):
            lon_col = col
    
    # Если не нашли стандартные названия, используем первые два столбца
    if lat_col is None or lon_col is None:
        if len(df.columns) >= 2:
            lat_col, lon_col = df.columns[0], df.columns[1]
        else:
            return {"error": "Не удалось определить столбцы с координатами"}
    
    # Проверяем на наличие пропущенных значений
    if df[[lat_col, lon_col]].isnull().any().any():
        df = df.dropna(subset=[lat_col, lon_col])
    
    if len(df) == 0:
        return {"error": "Нет данных для анализа"}
    
    # Находим крайние точки
    northernmost = df.loc[df[lat_col].idxmax()]
    southernmost = df.loc[df[lat_col].idxmin()]
    easternmost = df.loc[df[lon_col].idxmax()]
    westernmost = df.loc[df[lon_col].idxmin()]

    # Находим среднее значение широты и от него опредляем северные и южные

    mean_lat = df[lat_col].mean()
    df["North"] = df[lat_col].apply(lambda x: True if x > mean_lat else False)
    df["South"] = df[lat_col].apply(lambda x: True if x < mean_lat else False)

    mean_lon = df[lon_col].mean()
    df["West"] = df[lon_col].apply(lambda x: True if x < mean_lon else False)
    df["East"] = df[lon_col].apply(lambda x: True if x > mean_lon else False)   
    
    # Находим угловые точки (северо-запад, северо-восток, юго-запад, юго-восток)
    # Для этого используем комбинации экстремальных значений
    
    # Северо-запад: максимальная широта, минимальная долгота
    northwest_mask = (df[lat_col] == df[lat_col].max()) & (df[lon_col] == df[lon_col].min())
    if northwest_mask.any():
        northwest = df[northwest_mask].iloc[0]
    else:
        # Если нет точного совпадения, находим ближайшую точку
        df['nw_distance'] = (df[lat_col] - df[lat_col].max())**2 + (df[lon_col] - df[lon_col].min())**2
        northwest = df.loc[df['nw_distance'].idxmin()]
        df = df.drop('nw_distance', axis=1)
    
    # Северо-восток: максимальная широта, максимальная долгота
    northeast_mask = (df[lat_col] == df[lat_col].max()) & (df[lon_col] == df[lon_col].max())
    if northeast_mask.any():
        northeast = df[northeast_mask].iloc[0]
    else:
        df['ne_distance'] = (df[lat_col] - df[lat_col].max())**2 + (df[lon_col] - df[lon_col].max())**2
        northeast = df.loc[df['ne_distance'].idxmin()]
        df = df.drop('ne_distance', axis=1)
    
    # Юго-запад: минимальная широта, минимальная долгота
    southwest_mask = (df[lat_col] == df[lat_col].min()) & (df[lon_col] == df[lon_col].min())
    if southwest_mask.any():
        southwest = df[southwest_mask].iloc[0]
    else:
        df['sw_distance'] = (df[lat_col] - df[lat_col].min())**2 + (df[lon_col] - df[lon_col].min())**2
        southwest = df.loc[df['sw_distance'].idxmin()]
        df = df.drop('sw_distance', axis=1)
    
    # Юго-восток: минимальная широта, максимальная долгота
    southeast_mask = (df[lat_col] == df[lat_col].min()) & (df[lon_col] == df[lon_col].max())
    if southeast_mask.any():
        southeast = df[southeast_mask].iloc[0]
    else:
        df['se_distance'] = (df[lat_col] - df[lat_col].min())**2 + (df[lon_col] - df[lon_col].max())**2
        southeast = df.loc[df['se_distance'].idxmin()]
        df = df.drop('se_distance', axis=1)

    return df
    
def get_convex_hull_points(points):
    """Получаем точки выпуклой оболочки в правильном порядке"""
    hull = ConvexHull(points)
    return points[hull.vertices]

def delete_eject_quantile_fluo(data: DataFrame,
                            quant: float=0.995)-> DataFrame:

    """
    Удаляет экстремальные выбросы из 3D матрицы флуоресценции.

    Args:
        data (pd.DataFrame): 3D спектр флуоресценции.
        quant (float): Квантиль для определения выбросов.

    Returns:
        pd.DataFrame: Спектр с удаленными выбросами
    """
    data = data.copy()

    data[data > np.quantile(data,quant)] = 0

    return data

def normilize_by_max(data: DataFrame)-> DataFrame:

    """
    Нормирует спектр от 0 до 1

    Args:
        data (pd.DataFrame): 3D спектр флуоресценции.

    Returns:
        pd.DataFrame: Отнормированый спектр
    """

    data = data.copy()

    data = data/data.max(axis=None)

    return data

def get_strong_correlations(corr_matrix: pd.DataFrame, 
                            threshold: float = 0.8, 
                            include_negative: bool = True,
                            exclude_diagonal: bool = True):
    """
    Извлекает самые сильные корреляции из матрицы корреляций (DataFrame.corr()).
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        Матрица корреляций, полученная методом .corr()
    threshold : float, default=0.7
        Пороговое значение для отбора сильных корреляций
    include_negative : bool, default=True
        Включать ли отрицательные корреляции
    exclude_diagonal : bool, default=True
        Исключать ли диагональные элементы (корреляция переменной с самой собой)
    
    Returns:
    --------
    pandas.DataFrame
        Таблица с самыми сильными корреляциями
    """
    # Создаем копию матрицы корреляций
    corr_df = corr_matrix.copy()
    
    # Исключаем диагональные элементы, если нужно
    if exclude_diagonal:
        np.fill_diagonal(corr_df.values, np.nan)
    
    # Преобразуем матрицу в длинный формат
    corr_stacked = corr_df.stack().reset_index()
    corr_stacked.columns = ['Variable1', 'Variable2', 'Correlation']
    
    # Фильтруем по пороговому значению
    if include_negative:
        mask = (corr_stacked['Correlation'].abs() >= threshold)
    else:
        mask = (corr_stacked['Correlation'] >= threshold)
    
    strong_correlations = corr_stacked[mask].copy()
    
    # Удаляем дубликаты (корреляция A-B и B-A)
    strong_correlations['Pair'] = strong_correlations.apply(
        lambda x: tuple(sorted([x['Variable1'], x['Variable2']])), axis=1
    )
    strong_correlations = strong_correlations.drop_duplicates('Pair').drop('Pair', axis=1)
    
    # Сортируем по абсолютному значению корреляции
    strong_correlations['Abs_Correlation'] = strong_correlations['Correlation'].abs()
    strong_correlations = strong_correlations.sort_values('Abs_Correlation', ascending=False)
    strong_correlations = strong_correlations.drop('Abs_Correlation', axis=1)
    
    # Сбрасываем индекс
    strong_correlations = strong_correlations.reset_index(drop=True)
    
    return strong_correlations

def get_top_correlations(corr_matrix: pd.DataFrame,
                         n: int = 10,
                         include_negative: bool = True,
                         exclude_diagonal: bool = True):
    """
    Возвращает топ N самых сильных корреляций.
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        Матрица корреляций
    n : int, default=10
        Количество возвращаемых корреляций
    include_negative : bool, default=True
        Включать ли отрицательные корреляции
    exclude_diagonal : bool, default=True
        Исключать ли диагональные элементы
    """
    # Создаем копию матрицы корреляций
    corr_df = corr_matrix.copy()
    
    # Исключаем диагональные элементы, если нужно
    if exclude_diagonal:
        np.fill_diagonal(corr_df.values, np.nan)
    
    # Преобразуем матрицу в длинный формат
    corr_stacked = corr_df.stack().reset_index()
    corr_stacked.columns = ['Variable 1', 'Variable 2', 'Correlation']
    
    # Создаем столбец с абсолютным значением
    corr_stacked['Abs_Correlation'] = corr_stacked['Correlation'].abs()
    
    # Фильтруем отрицательные корреляции, если нужно
    if not include_negative:
        corr_stacked = corr_stacked[corr_stacked['Correlation'] >= 0]
    
    # Удаляем дубликаты
    corr_stacked['Pair'] = corr_stacked.apply(
        lambda x: tuple(sorted([x['Variable 1'], x['Variable 2']])), axis=1
    )
    corr_stacked = corr_stacked.drop_duplicates('Pair').drop('Pair', axis=1)
    
    # Сортируем и берем топ N
    top_correlations = corr_stacked.sort_values('Abs_Correlation', ascending=False).head(n)
    top_correlations = top_correlations.drop('Abs_Correlation', axis=1)
    top_correlations = top_correlations.reset_index(drop=True)
    
    return top_correlations

def check_normality(df, alpha=0.05):

    """
    Проверяет нормальность распределения для всех числовых столбцов DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Входной DataFrame
    alpha : float, default=0.05
        Уровень значимости
        
    Returns:
    --------
    pandas.DataFrame с результатами тестов
    """
    results = []
    
    for column in df.select_dtypes(include=[np.number]).columns:
        data = df[column].dropna()
        
        # Тест Шапиро-Уилка (для n < 5000)
        if len(data) < 5000:
            stat_sw, p_sw = stats.shapiro(data)
            normal_sw = p_sw > alpha
        else:
            stat_sw, p_sw = np.nan, np.nan
            normal_sw = np.nan
        
        # Тест нормальности Д'Агостино K^2
        stat_da, p_da = stats.normaltest(data)
        normal_da = p_da > alpha
        
        # Тест Андерсона-Дарлинга
        result_ad = stats.anderson(data, dist='norm')

        # Подавление предупреждений типа
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            result: Any = stats.anderson(data, dist='norm')
            
            # Безопасное извлечение critical_values
            critical_values = getattr(result, 'critical_values', [])
            statistic = getattr(result, 'statistic', float('nan'))
        
        normal_ad = statistic < critical_values # type: ignore
        
        results.append({
            'Column': column,
            'Shapiro-Wilk_Stat': stat_sw,
            'Shapiro-Wilk_p': p_sw,
            'Shapiro-Wilk_Normal': normal_sw,
            'D_Agostino_Stat': stat_da,
            'D_Agostino_p': p_da,
            'D_Agostino_Normal': normal_da,
            'Anderson-Darling_Stat': statistic,
            'Anderson-Darling_Normal': normal_ad,
            'Sample_Size': len(data)
        })
    
    return pd.DataFrame(results)

def plot_strong_correlations(corr_matrix,n=3,ax=None):
    """
    Визуализирует сильные корреляции.
    """
    strong_corr = get_top_correlations(corr_matrix, n=n)
    
    if len(strong_corr) == 0:
        print("Нет сильных корреляций выше порога")
        return
    
    if ax == None:

        fig, ax = plt.subplots(1,1,figsize=(4,4))
    

    colors = ['blue' if x < 0 else 'red' for x in strong_corr['Correlation']]
    bars = ax.barh(range(len(strong_corr)), strong_corr['Correlation'], color=colors)
    
    ax.set_yticks(range(len(strong_corr)), 
               [f"{row['Variable 1']}\n{row['Variable 2']}" 
                for _, row in strong_corr.iterrows()],
                rotation=90)
    
    ax.set_xlabel('Корреляция')
    ax.set_title(f'Сильные корреляции')
    ax.grid(axis='x', alpha=0.3)
    
    for bar in bars:
        width = bar.get_width()
        text_color = 'white'
        
        ax.text(width/2, bar.get_y() + bar.get_height()/2, 
                f'{round(width,2)}', 
                ha='center', va='center', 
                color=text_color, fontweight='bold', fontsize=14)
        
def add_median_labels(ax, fmt='.3f'):
    
    '''
    Добавление числовой подписи на медиану в seaborn.boxplot
    '''

    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))

    for median in lines[4:len(lines):lines_per_box]:

        x, y = (data.mean() for data in median.get_data())

        # choose value depending on horizontal or vertical plot orientation

        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
    
