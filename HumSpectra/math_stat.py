import pandas as pd
from pandas import DataFrame
import numpy as np
from scipy import stats
from typing import Any, Dict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

def delete_eject_iqr(data: DataFrame,
                 iqr_param: float = 1.5,
                 level_index: int = 0,
                 multi_index: bool = False,
                 columns: list|None = None) -> DataFrame:
    """
    :param data: DataFrame
    :param iqr_param: Межквартильный множитель
    :param level_index: Уровень индекса, по которому данные группируются и удаляются выбросы
    :return: Отфильтргованная таблица
    """
    data_copy = data.copy()


    if columns is None:
        descriptor_list = data_copy.columns
    
    else:
        descriptor_list = columns

    if multi_index:

        class_list = data_copy.index.unique(level=level_index)

        for descriptor in descriptor_list:

            for gen_class in class_list:

                data_iqr = data_copy.loc[gen_class]

                if data_iqr.shape[0] < 5:
                    continue

                q1 = data_iqr[descriptor].quantile(0.25)
                q3 = data_iqr[descriptor].quantile(0.75)
                iqr = q3 - q1

                data_iqr = data_iqr[(data_iqr[descriptor] < q3 + iqr_param * iqr)]
                data_iqr = data_iqr[(data_iqr[descriptor] > q1 - iqr_param * iqr)]

                data_iqr = pd.concat({gen_class: data_iqr}, names=['Класс'])

                data_copy.loc[gen_class] = data_iqr
                
                data_copy.dropna(inplace=True)

        return data_copy
    
    else:

        for descriptor in descriptor_list:

                data_iqr = data_copy

                if data_iqr.shape[0] < 5:
                    continue

                q1 = data_iqr[descriptor].quantile(0.25)
                q3 = data_iqr[descriptor].quantile(0.75)
                iqr = q3 - q1

                data_iqr = data_iqr[(data_iqr[descriptor] < q3 + iqr_param * iqr)]
                data_iqr = data_iqr[(data_iqr[descriptor] > q1 - iqr_param * iqr)]
                
                data_copy.dropna(inplace=True)

        return data_copy

def delete_eject_quantile(data: DataFrame,
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
    

