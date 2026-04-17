import numpy as np
import pandas as pd
import os
from typing import List, Optional, Tuple
from pathlib import Path
import HumSpectra.mass_spectra as ms
import HumSpectra.utilits as ut
import pandas as pd
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
import re
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
import HumSpectra.mass_descriptors as md

def build_formula(row):
    return 'C'+str(row['C']) +'H'+ str(row['H']) +'O'+ str(row['O'])+'N'+ str(row['N'])

def analyze_mass_intervals(
    spectra_list: List[pd.DataFrame],
    mass_range: Tuple[float, float] = (300.0, 450.0),
    save_to_folder: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame,List[pd.DataFrame]]:
    """
    Подсчитывает количество формул в интервалах масс шириной 1 Да для каждого образца,
    находит три лучших интервала и возвращает сводную таблицу.

    Параметры:
    ----------
    spectra_list : List[pd.DataFrame]
        Список DataFrame, каждый из которых соответствует одному образцу.
        Должен содержать колонку 'calc_mass' и атрибут attrs['name'] с именем образца.
    mass_range : Tuple[float, float], default (300.0, 450.0)
        Начало и конец диапазона масс (конец не включается в последний интервал).
    save_to_folder : str, optional
        Если указана папка, сохраняет CSV-файлы с распределением для каждого образца.
    verbose : bool, default True
        Выводить ли сообщения о ходе выполнения.

    Возвращает:
    -----------
    summary_df : pd.DataFrame
        DataFrame с индексами = имена образцов и колонками:
        'best_interval_1', 'count_1', 'best_interval_2', 'count_2',
        'best_interval_3', 'count_3'.
    """
    formula_list = []
    
    if not spectra_list:
        return pd.DataFrame(), formula_list

    # Подготовка папки для сохранения
    if save_to_folder is not None:
        os.makedirs(save_to_folder, exist_ok=True)

    # Границы бинов: целые числа от start до end+1, чтобы интервалы были [i, i+1)
    start_mass = int(np.floor(mass_range[0]))
    end_mass = int(np.ceil(mass_range[1]))
    bins = np.arange(start_mass, end_mass + 1, 1)  # бины: [start, start+1), ..., [end-1, end)

    summary_data = []
    total_samples = len(spectra_list)

    for idx, df in enumerate(spectra_list):
        # Получение имени образца
        sample_name = df.attrs.get('name', f'sample_{idx}')
        if verbose:
            print(f"Обработка образца: {sample_name} ({idx+1}/{total_samples})")

        # Проверка наличия колонки 'calc_mass'
        if 'calc_mass' not in df.columns:
            raise ValueError(f"В DataFrame образца '{sample_name}' отсутствует колонка 'calc_mass'")

        masses = df['calc_mass'].values

        # Подсчёт гистограммы
        counts, bin_edges = np.histogram(masses, bins=bins) # type: ignore
        intervals = bin_edges[:-1]  # левые границы интервалов

        # Создание DataFrame распределения для образца
        distrib_df = pd.DataFrame({
            'mass_start': intervals,
            'mass_end': intervals + 1,
            'formula_count': counts
        })
        distrib_df.attrs['name'] = df.attrs['name']
        distrib_df.attrs['class'] = df.attrs['class']
        formula_list.append(distrib_df)

        # Сохранение в файл, если требуется
        if save_to_folder is not None:
            filename = f"{sample_name}_mass_distribution.csv"
            filepath = os.path.join(save_to_folder, filename)
            distrib_df.to_csv(filepath, index=False)
            if verbose:
                print(f"  Сохранено распределение в {filepath}")

        # Поиск трёх лучших интервалов (по убыванию количества)
        # Сортируем по убыванию counts, берём первые 3
        sorted_indices = np.argsort(counts)[::-1]
        best_intervals = []
        best_counts = []
        for i in range(3):
            if i < len(sorted_indices):
                best_intervals.append(intervals[sorted_indices[i]])
                best_counts.append(counts[sorted_indices[i]])
            else:
                best_intervals.append(np.nan)
                best_counts.append(0)

        summary_data.append({
            'sample_name': sample_name,
            'best_interval_1': best_intervals[0],
            'count_1': best_counts[0],
            'best_interval_2': best_intervals[1],
            'count_2': best_counts[1],
            'best_interval_3': best_intervals[2],
            'count_3': best_counts[2]
        })

    # Создание сводной таблицы
    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index('sample_name', inplace=True)

    if verbose:
        print("\nГотово! Сводная таблица создана.")

    return summary_df, formula_list

def average_formulas_per_mass_interval(
    spectra_list: List[pd.DataFrame],
    mass_range: Tuple[float, float] = (300.0, 450.0),
    verbose: bool = True
) -> pd.DataFrame:
    """
    Вычисляет среднее количество формул в интервалах масс шириной 1 Да
    для каждого образца в заданном диапазоне масс.

    Параметры:
    ----------
    spectra_list : List[pd.DataFrame]
        Список DataFrame, каждый из которых соответствует одному образцу.
        Должен содержать колонку 'calc_mass' и атрибут attrs['name'] с именем образца.
    mass_range : Tuple[float, float], default (300.0, 450.0)
        Начало и конец диапазона масс (конец не включается в последний интервал).
    verbose : bool, default True
        Выводить ли сообщения о ходе выполнения.

    Возвращает:
    -----------
    avg_df : pd.DataFrame
        DataFrame с индексом = имена образцов и колонкой 'avg_formula_count'.
    """
    if not spectra_list:
        return pd.DataFrame(columns=['avg_formula_count'])

    # Границы бинов: целые числа от start до end+1 для интервалов [i, i+1)
    start_mass = int(np.floor(mass_range[0]))
    end_mass = int(np.ceil(mass_range[1]))
    bins = np.arange(start_mass, end_mass + 1, 1)
    num_intervals = len(bins) - 1  # количество интервалов

    avg_data = []
    total_samples = len(spectra_list)

    for idx, df in enumerate(spectra_list):
        # Получение имени образца
        sample_name = df.attrs.get('name', f'sample_{idx}')
        if verbose:
            print(f"Обработка образца: {sample_name} ({idx+1}/{total_samples})")

        # Проверка наличия колонки 'calc_mass'
        if 'calc_mass' not in df.columns:
            raise ValueError(f"В DataFrame образца '{sample_name}' отсутствует колонка 'calc_mass'")

        masses = df['calc_mass'].values

        # Подсчёт гистограммы
        counts, _ = np.histogram(masses, bins=bins) # type: ignore

        # Среднее количество формул на интервал
        avg_count = np.mean(counts)

        avg_data.append({
            'sample_name': sample_name,
            'avg_formula_count': avg_count
        })

    # Формирование итогового DataFrame
    avg_df = pd.DataFrame(avg_data)
    avg_df.set_index('sample_name', inplace=True)

    if verbose:
        print(f"\nГотово! Средние значения вычислены для {len(avg_df)} образцов.")

    return avg_df

def plot_stacked_barplot(df, name_col='name', figsize=(12, 6), 
                         colormap='viridis', title=None,
                         show_values=True, value_format='{:.2f}'):
    """
    Строит stacked barplot для DataFrame с нормализованными значениями.
    
    Параметры:
    -----------
    df : pandas.DataFrame
        DataFrame с нормализованными данными (сумма числовых столбцов в каждой строке = 1)
    name_col : str, default='name'
        Название столбца с именами образцов/точек
    figsize : tuple, default=(12, 6)
        Размер фигуры (ширина, высота)
    colormap : str or list, default='viridis'
        Цветовая схема для числовых столбцов
    title : str, optional
        Заголовок графика
    show_values : bool, default=True
        Показывать ли значения на столбцах
    value_format : str, default='{:.2f}'
        Формат отображения значений (например, '{:.1%}' для процентов)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Копируем DataFrame, чтобы не изменять исходный
    df_plot = df.copy()
    
    # Определяем числовые столбцы (исключая столбец с именами)
    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
    
    # Проверяем, что столбец с именами существует
    if name_col not in df_plot.columns:
        raise ValueError(f"Столбец '{name_col}' не найден в DataFrame")
    
    # Убеждаемся, что столбец с именами - первый
    cols_order = [name_col] + numeric_cols
    df_plot = df_plot[cols_order]
    
    # Проверяем, что данные нормализованы (опционально)
    sums = df_plot[numeric_cols].sum(axis=1)
    if not np.allclose(sums, 1, rtol=1e-5):
        print("Предупреждение: Суммы строк не равны 1. Результат может быть некорректным.")
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=figsize)
    
    # Получаем список имен и числовых столбцов
    names = df_plot[name_col].values
    n_names = len(names)
    
    # Получаем цветовую палитру
    if isinstance(colormap, str):
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(numeric_cols)))
    else:
        colors = colormap
    
    # Рисуем stacked bars
    bottom = np.zeros(n_names)
    
    for idx, col in enumerate(numeric_cols):
        values = df_plot[col].values
        bars = ax.bar(names, values, bottom=bottom, 
                     label=col, color=colors[idx], 
                     edgecolor='white', linewidth=0.5)
        
        # Добавляем значения на столбцы (опционально)
        if show_values:
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 0.02:  # Показываем только значимые значения (>2%)
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           bottom[i] + height/2.,
                           value_format.format(val),
                           ha='center', va='center', 
                           fontsize=8, fontweight='bold',
                           color='white' if np.mean(colors[idx][:3]) < 0.5 else 'black')
        
        bottom += values
    
    # Настройка графика
    ax.set_xlabel('Образцы', fontsize=12, fontweight='bold')
    ax.set_ylabel('Относительная доля', fontsize=12, fontweight='bold')
    
    if title is None:
        title = f'Stacked Barplot: распределение по {len(numeric_cols)} категориям'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Легенда
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
             title='Категории', title_fontsize=10)
    
    # Поворот подписей оси X, если нужно
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Сетка
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Устанавливаем лимиты
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    return fig, ax

@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} сек")

wdir_spectra = r"D:\lab\MassSpectraNew\processed_results\mzlist"
wdir_statistic = r"D:\lab\MassSpectraNew\processed_results\statistic"
wdir = r"D:\lab\MassSpectraNew\processed_results"

bad_spectra_data = pd.read_csv(r"D:\lab\MassSpectraNew\processed_results\bad_spectra.csv")
bad_spectra_list = bad_spectra_data['name'].to_list()

spectra_list = []
spectra_stat_list = []
spectra_class_list = []
spectra_list_name = []

with timer("Чтение спектров"):
    for path in Path.rglob(Path(wdir_spectra),'*.csv'):
        
        spectra = pd.read_csv(path)
        spectra_name = ut.extract_name_from_path(str(path))
        spectra_name = spectra_name.split(sep='__')[0]
        if spectra_name in bad_spectra_list:
            continue
        spectra_class = ut.extract_class_from_name(spectra_name)
        
        spectra.attrs['name'] = spectra_name
        spectra.attrs['class'] = spectra_class
        spectra['formula'] = spectra.apply(lambda x: build_formula(x),axis=1)
        spectra_list.append(spectra)
        spectra_class_list.append(spectra_class)
        spectra_list_name.append(spectra_name)

data,all_formula_dfs = analyze_mass_intervals(spectra_list)
data.to_excel(r"D:\lab\MassSpectraNew\processed_results\распределение по бакетам.xlsx")

df = average_formulas_per_mass_interval(spectra_list)

square_7_list = []
square_12_list = []
square_8_list = []
square_13_list = []
square_9_list = []
square_14_list = []

for spectra in spectra_list:
    
    square_data = md.get_squares_vk(spectra)
    
    square_7_list.append(square_data.iloc[6]['value'])
    square_12_list.append(square_data.iloc[11]['value'])
    square_8_list.append(square_data.iloc[7]['value'])
    square_13_list.append(square_data.iloc[12]['value'])
    square_9_list.append(square_data.iloc[8]['value'])
    square_14_list.append(square_data.iloc[13]['value'])
    
df_square = pd.DataFrame()
df_square['name'] = spectra_list_name
df_square['7'] = square_7_list
df_square['12'] = square_12_list
df_square['8'] = square_8_list
df_square['13'] = square_13_list
df_square['9'] = square_9_list
df_square['14'] = square_14_list

numeric_cols = df_square.select_dtypes(include=[np.number]).columns
df_numeric = df_square[numeric_cols]
row_sums = df_numeric.sum(axis=1)
df_normalized = df_numeric.div(row_sums, axis=0)
df_square[numeric_cols] = df_normalized

df_square.to_excel(r"D:\lab\MassSpectraNew\processed_results\Интенсивность по квадратам.xlsx")

plot_stacked_barplot(df_square,show_values=False)
plt.savefig(r"D:\lab\MassSpectraNew\processed_results\Интенсивность по квадратам.png")
plt.close()


class FTICR_AnomalyDetector:
    """
    Детектор аномалий для FT-ICR MS данных.
    Обучается на фоновых пробах Байкала и выявляет аномальные сигналы в надшламовых водах.
    """
    
    def __init__(self, hc_bins=50, oc_bins=50, mass_bin_width=1.0):
        """
        Parameters:
        -----------
        hc_bins : int
            Количество бинов для H/C оси диаграммы Ван-Кревелена
        oc_bins : int
            Количество бинов для O/C оси диаграммы Ван-Кревелена
        mass_bin_width : float
            Ширина бина для масс-спектра в Да
        """
        self.hc_bins = hc_bins
        self.oc_bins = oc_bins
        self.mass_bin_width = mass_bin_width
        self.scaler = StandardScaler()
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.pca = None
        self.feature_names = []
        
    def calculate_vk_density_features(self, df, hc_range=(0, 2.5), oc_range=(0, 1.2)):
        """
        Расчет плотности заселенности квадратов диаграммы Ван-Кревелена.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с колонками 'H/C' и 'O/C'
        hc_range : tuple
            Диапазон значений H/C
        oc_range : tuple
            Диапазон значений O/C
            
        Returns:
        --------
        np.array : 1D массив плотностей для каждого бина
        """
        # Фильтруем данные в заданных диапазонах
        mask = (df['H/C'] >= hc_range[0]) & (df['H/C'] <= hc_range[1]) & \
               (df['O/C'] >= oc_range[0]) & (df['O/C'] <= oc_range[1])
        filtered_df = df[mask]
        
        # Создаем 2D гистограмму
        hc_edges = np.linspace(hc_range[0], hc_range[1], self.hc_bins + 1)
        oc_edges = np.linspace(oc_range[0], oc_range[1], self.oc_bins + 1)
        
        hist_2d, _, _ = np.histogram2d(
            filtered_df['H/C'], 
            filtered_df['O/C'],
            bins=[hc_edges, oc_edges],
            weights=filtered_df['intensity'] if 'intensity' in df.columns else None
        )
        
        # Нормализация плотности
        if hist_2d.sum() > 0:
            hist_2d = hist_2d / hist_2d.sum()
        
        return hist_2d.flatten()
    
    def calculate_mass_spectrum_features(self, mass_df):
        """
        Извлечение признаков из масс-спектра (количество формул в интервалах масс).
        
        Parameters:
        -----------
        mass_df : pd.DataFrame
            Датафрейм с колонками 'mass_start', 'mass_end', 'formula_count'
            
        Returns:
        --------
        np.array : массив количества формул в каждом интервале
        """
        # Сортируем по массе
        mass_df = mass_df.sort_values('mass_start')
        
        # Извлекаем количество формул
        formula_counts = mass_df['formula_count'].values
        
        # Нормализация
        if formula_counts.sum() > 0:
            formula_counts = formula_counts / formula_counts.sum()
        
        return formula_counts
    
    def calculate_molecular_class_features(self, df):
        """
        Расчет распределения по молекулярным классам.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с колонкой 'class'
            
        Returns:
        --------
        dict : словарь с долями каждого класса
        """
        class_counts = df['class'].value_counts()
        class_fractions = class_counts / class_counts.sum()
        
        # Основные классы в DOM
        expected_classes = ['CHO', 'CHON', 'CHOS', 'CHONS', 'other']
        features = []
        
        for cls in expected_classes:
            features.append(class_fractions.get(cls, 0))
            
        return np.array(features)
    
    def calculate_aggregated_features(self, df):
        """
        Расчет агрегированных молекулярных характеристик.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с колонками 'H/C', 'O/C', 'calc_mass', 'intensity'
            
        Returns:
        --------
        np.array : массив агрегированных признаков
        """
        features = []
        
        # Средневзвешенные характеристики (по интенсивности)
        if 'intensity' in df.columns:
            weights = df['intensity'] / df['intensity'].sum()
            
            # Средневзвешенный H/C
            weighted_hc = np.average(df['H/C'], weights=weights)
            features.append(weighted_hc)
            
            # Средневзвешенный O/C
            weighted_oc = np.average(df['O/C'], weights=weights)
            features.append(weighted_oc)
            
            # Средневзвешенная масса
            weighted_mass = np.average(df['calc_mass'], weights=weights)
            features.append(weighted_mass)
        else:
            features.extend([df['H/C'].mean(), df['O/C'].mean(), df['calc_mass'].mean()])
        
        # Медианные значения
        features.append(df['H/C'].median())
        features.append(df['O/C'].median())
        
        # Стандартные отклонения
        features.append(df['H/C'].std())
        features.append(df['O/C'].std())
        
        # Квантили
        features.append(df['H/C'].quantile(0.25))
        features.append(df['H/C'].quantile(0.75))
        features.append(df['O/C'].quantile(0.25))
        features.append(df['O/C'].quantile(0.75))
        
        # Индекс ароматичности (AI)
        ai_mod = (1 + df['C'].fillna(0) - 0.5*df['O'].fillna(0) - 
                  df['S'].fillna(0) - 0.5*(df['H'].fillna(0) + df['N'].fillna(0))) / \
                 (df['C'].fillna(0) - 0.5*df['O'].fillna(0) - df['N'].fillna(0) + 1e-10)
        
        ai_mod = ai_mod.clip(0, 1)
        features.append(ai_mod.mean())
        features.append(ai_mod.std())
        
        # DBE (Double Bond Equivalent)
        if all(col in df.columns for col in ['C', 'H', 'N']):
            dbe = 1 + df['C'] - 0.5*df['H'] + 0.5*df['N']
            features.append(dbe.mean())
            features.append(dbe.std())
        else:
            features.extend([0, 0])
        
        # Доля ароматических соединений (AI > 0.5)
        aromatic_fraction = (ai_mod > 0.5).mean()
        features.append(aromatic_fraction)
        
        # Доля конденсированных ароматических соединений (AI > 0.67)
        condensed_aromatic_fraction = (ai_mod > 0.67).mean()
        features.append(condensed_aromatic_fraction)
        
        return np.array(features)
    
    def extract_features(self, formula_df, mass_df):
        """
        Извлечение всех признаков из одного образца.
        
        Parameters:
        -----------
        formula_df : pd.DataFrame
            Датафрейм с формулами и их характеристиками
        mass_df : pd.DataFrame
            Датафрейм с распределением масс
            
        Returns:
        --------
        np.array : вектор признаков для образца
        """
        features = []
        feature_names = []
        
        # 1. Плотность в диаграмме Ван-Кревелена
        vk_density = self.calculate_vk_density_features(formula_df)
        features.extend(vk_density)
        feature_names.extend([f'VK_density_{i}' for i in range(len(vk_density))])
        
        # 2. Распределение масс
        mass_features = self.calculate_mass_spectrum_features(mass_df)
        features.extend(mass_features)
        feature_names.extend([f'Mass_bin_{i}' for i in range(len(mass_features))])
        
        # 3. Распределение по классам соединений
        if 'class' in formula_df.columns:
            class_features = self.calculate_molecular_class_features(formula_df)
            features.extend(class_features)
            feature_names.extend(['CHO_frac', 'CHON_frac', 'CHOS_frac', 'CHONS_frac', 'Other_frac'])
        
        # 4. Агрегированные характеристики
        agg_features = self.calculate_aggregated_features(formula_df)
        features.extend(agg_features)
        feature_names.extend([
            'weighted_HC', 'weighted_OC', 'weighted_mass',
            'median_HC', 'median_OC',
            'std_HC', 'std_OC',
            'q25_HC', 'q75_HC', 'q25_OC', 'q75_OC',
            'mean_AI', 'std_AI',
            'mean_DBE', 'std_DBE',
            'aromatic_frac', 'condensed_aromatic_frac'
        ])
        
        self.feature_names = feature_names
        
        return np.array(features)
    
    def fit(self, baikal_formula_dfs, baikal_mass_dfs):
        """
        Обучение One-Class SVM на фоновых пробах Байкала.
        
        Parameters:
        -----------
        baikal_formula_dfs : list
            Список датафреймов с формулами для проб Байкала
        baikal_mass_dfs : list
            Список датафреймов с распределением масс для проб Байкала
        """
        # Извлекаем признаки для всех проб Байкала
        baikal_features = []
        
        for form_df, mass_df in zip(baikal_formula_dfs, baikal_mass_dfs):
            features = self.extract_features(form_df, mass_df)
            baikal_features.append(features)
        
        baikal_features = np.array(baikal_features)
        
        # Нормализация признаков
        self.scaler.fit(baikal_features)
        baikal_features_scaled = self.scaler.transform(baikal_features)
        
        # Обучение One-Class SVM
        self.ocsvm.fit(baikal_features_scaled)
        
        # PCA для визуализации
        self.pca = PCA(n_components=2)
        self.pca.fit(baikal_features_scaled)
        
        print(f"Модель обучена на {len(baikal_formula_dfs)} пробах Байкала")
        print(f"Количество признаков: {baikal_features.shape[1]}")
        
        return self
    
    def predict_anomaly(self, formula_df, mass_df):
        """
        Предсказание аномальности для нового образца.
        
        Parameters:
        -----------
        formula_df : pd.DataFrame
            Датафрейм с формулами
        mass_df : pd.DataFrame
            Датафрейм с распределением масс
            
        Returns:
        --------
        dict : словарь с результатами
            - 'is_anomaly': bool (True если аномалия)
            - 'anomaly_score': float (расстояние до разделяющей гиперплоскости)
            - 'confidence': float (уверенность в аномальности, 0-1)
        """
        # Извлекаем признаки
        features = self.extract_features(formula_df, mass_df)
        features = features.reshape(1, -1)
        
        # Нормализация
        features_scaled = self.scaler.transform(features)
        
        # Предсказание
        prediction = self.ocsvm.predict(features_scaled)
        score = self.ocsvm.score_samples(features_scaled)[0]
        
        # Нормализация score в уверенность (0-1)
        confidence = 1 / (1 + np.exp(-score))  # Сигмоидная трансформация
        
        is_anomaly = prediction[0] == -1
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': score,
            'confidence': confidence
        }
    
    def analyze_molecular_anomalies(self, formula_df, baikal_formula_dfs):
        """
        Поиск конкретных аномальных молекул в образце.
        
        Parameters:
        -----------
        formula_df : pd.DataFrame
            Датафрейм с формулами анализируемого образца
        baikal_formula_dfs : list
            Список датафреймов с формулами для проб Байкала
            
        Returns:
        --------
        pd.DataFrame : исходный датафрейм с добавленными колонками аномальности
        """
        result_df = formula_df.copy()
        
        # Объединяем все пробы Байкала
        all_baikal = pd.concat(baikal_formula_dfs, ignore_index=True)
        
        # Для каждой формулы считаем статистику встречаемости в Байкале
        baikal_stats = all_baikal.groupby('formula').agg({
            'intensity': ['mean', 'std', 'max', 'count']
        }).reset_index()
        
        baikal_stats.columns = ['formula', 'baikal_mean_int', 'baikal_std_int', 
                               'baikal_max_int', 'baikal_occurrence']
        
        # Объединяем с анализируемым образцом
        result_df = result_df.merge(baikal_stats, on='formula', how='left')
        
        # Маркеры аномальности
        result_df['is_new_formula'] = result_df['baikal_occurrence'].isna()
        result_df['intensity_ratio'] = result_df['intensity'] / result_df['baikal_max_int'].fillna(1)
        result_df['is_intensity_anomaly'] = result_df['intensity_ratio'] > 10
        
        # Флаг общей аномальности молекулы
        result_df['molecular_anomaly'] = result_df['is_new_formula'] | result_df['is_intensity_anomaly']
        
        # Специфические маркеры надшламовых вод
        if all(col in result_df.columns for col in ['H/C', 'O/C', 'AI']):
            # Зона лигносульфонатов и окисленных фенолов
            result_df['sludge_marker'] = (
                (result_df['O/C'] > 0.6) & 
                (result_df['H/C'] < 0.8) & 
                (result_df['AI'] > 0.5)
            )
            
            # S-содержащие соединения (редки в ультрапресном Байкале)
            if 'S' in result_df.columns:
                result_df['sulfur_marker'] = result_df['S'] > 0
            elif 'class' in result_df.columns:
                result_df['sulfur_marker'] = result_df['class'].str.contains('S', na=False)
        
        return result_df
    
    def visualize_results(self, baikal_features, sludge_features, sample_names=None):
        """
        Визуализация результатов анализа в пространстве PCA.
        
        Parameters:
        -----------
        baikal_features : np.array
            Признаки проб Байкала
        sludge_features : np.array
            Признаки проб надшламовых вод
        sample_names : list
            Имена образцов
        """
        if self.pca is None:
            print("Сначала обучите модель!")
            return
        
        # Трансформируем все признаки
        baikal_pca = self.pca.transform(baikal_features)
        sludge_pca = self.pca.transform(sludge_features)
        
        # Создаем график
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: PCA с областями
        ax1 = axes[0]
        
        # Рисуем область нормы (эллипс для Байкала)
        from matplotlib.patches import Ellipse
        
        baikal_mean = baikal_pca.mean(axis=0)
        baikal_cov = np.cov(baikal_pca.T)
        
        # Собственные значения и векторы
        eigenvalues, eigenvectors = np.linalg.eigh(baikal_cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues) * 2  # 2 стандартных отклонения
        
        ellipse = Ellipse(xy=baikal_mean, width=width, height=height, 
                         angle=angle, facecolor='lightblue', alpha=0.3, 
                         edgecolor='blue', linewidth=2, label='Область нормы Байкала')
        ax1.add_patch(ellipse)
        
        # Точки Байкала
        ax1.scatter(baikal_pca[:, 0], baikal_pca[:, 1], 
                   c='blue', label='Байкал (фон)', s=100, alpha=0.7, edgecolors='black')
        
        # Точки надшламовых вод
        ax1.scatter(sludge_pca[:, 0], sludge_pca[:, 1], 
                   c='red', label='Надшламовые воды', s=100, alpha=0.7, edgecolors='black', marker='^')
        
        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('PCA анализ: Байкал vs Надшламовые воды')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Anomaly Scores
        ax2 = axes[1]
        
        # Вычисляем anomaly scores для всех образцов
        baikal_scores = self.ocsvm.score_samples(baikal_features)
        sludge_scores = self.ocsvm.score_samples(sludge_features)
        
        # Объединяем данные для boxplot
        data = [baikal_scores, sludge_scores]
        labels = ['Байкал (n={})'.format(len(baikal_scores)), 
                 'Надшлам (n={})'.format(len(sludge_scores))]
        
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        # Добавляем индивидуальные точки
        for i, scores in enumerate(data):
            x = np.random.normal(i+1, 0.04, size=len(scores))
            ax2.scatter(x, scores, alpha=0.6, c=['blue' if i==0 else 'red'][0])
        
        ax2.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5, label='Порог аномалии')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Распределение anomaly scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Статистика
        print("\n=== Статистика анализа ===")
        print(f"Байкал: средний score = {baikal_scores.mean():.3f} ± {baikal_scores.std():.3f}")
        print(f"Надшлам: средний score = {sludge_scores.mean():.3f} ± {sludge_scores.std():.3f}")
        print(f"Аномальных проб надшлама: {(sludge_scores < -0.5).sum()} из {len(sludge_scores)}")

# ===== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====

def prepare_data_for_analysis(list_of_formula_dfs, list_of_mass_dfs):
    """
    Подготовка данных для анализа.
    
    Parameters:
    -----------
    list_of_formula_dfs : list
        Список всех датафреймов с формулами
    list_of_mass_dfs : list
        Список всех датафреймов с распределением масс
        
    Returns:
    --------
    tuple : (baikal_formula_dfs, baikal_mass_dfs, sludge_formula_dfs, sludge_mass_dfs)
    """
    baikal_formula_dfs = []
    baikal_mass_dfs = []
    sludge_formula_dfs = []
    sludge_mass_dfs = []
    
    for form_df, mass_df in zip(list_of_formula_dfs, list_of_mass_dfs):
        sample_class = form_df.attrs.get('class', '')
        
        if sample_class == 'Baikal':
            baikal_formula_dfs.append(form_df)
            baikal_mass_dfs.append(mass_df)
        elif sample_class == 'ADOM':
            sludge_formula_dfs.append(form_df)
            sludge_mass_dfs.append(mass_df)
    
    return baikal_formula_dfs, baikal_mass_dfs, sludge_formula_dfs, sludge_mass_dfs

# Инициализация и обучение модели
detector = FTICR_AnomalyDetector(hc_bins=30, oc_bins=30)

# Разделяем данные
baikal_forms, baikal_masses, sludge_forms, sludge_masses = prepare_data_for_analysis(
    spectra_list, all_formula_dfs
)

# Обучаем модель на Байкале
detector.fit(baikal_forms, baikal_masses)

# Анализируем надшламовые воды
results = []
for i, (form_df, mass_df) in enumerate(zip(sludge_forms, sludge_masses)):
    sample_name = form_df.attrs.get('name', f'Sample_{i}')
    result = detector.predict_anomaly(form_df, mass_df)
    result['sample_name'] = sample_name
    results.append(result)
    
    print(f"\n{sample_name}:")
    print(f"  Аномалия: {result['is_anomaly']}")
    print(f"  Score: {result['anomaly_score']:.3f}")
    print(f"  Уверенность: {result['confidence']:.2%}")
    
    # Детальный анализ молекул для аномальных проб
    if result['is_anomaly']:
        molecular_analysis = detector.analyze_molecular_anomalies(form_df, baikal_forms)
        n_anomalous = molecular_analysis['molecular_anomaly'].sum()
        n_sludge_markers = molecular_analysis['sludge_marker'].sum() if 'sludge_marker' in molecular_analysis.columns else 0
        n_sulfur = molecular_analysis['sulfur_marker'].sum() if 'sulfur_marker' in molecular_analysis.columns else 0
        
        print(f"  Аномальных молекул: {n_anomalous} из {len(molecular_analysis)}")
        print(f"  Маркеров шлама: {n_sludge_markers}")
        print(f"  S-содержащих соединений: {n_sulfur}")

# Визуализация
baikal_features = np.array([detector.extract_features(f, m) for f, m in zip(baikal_forms, baikal_masses)])
sludge_features = np.array([detector.extract_features(f, m) for f, m in zip(sludge_forms, sludge_masses)])

baikal_features_scaled = detector.scaler.transform(baikal_features)
sludge_features_scaled = detector.scaler.transform(sludge_features)

detector.visualize_results(baikal_features_scaled, sludge_features_scaled)

class MultiModal_FTICR_Optics_AnomalyDetector:
    """
    Мультимодальный детектор аномалий, объединяющий:
    1. FT-ICR MS данные (молекулярный состав)
    2. Оптические данные (УФ и флуоресценция)
    
    Использует late fusion подход: отдельные детекторы для каждой модальности
    с последующим объединением результатов.
    """
    
    def __init__(self, hc_bins=30, oc_bins=30, mass_bin_width=1.0, 
                 fusion_method='weighted_voting', use_pls=True):
        """
        Parameters:
        -----------
        hc_bins, oc_bins : int
            Количество бинов для диаграммы Ван-Кревелена
        mass_bin_width : float
            Ширина бина для масс-спектра в Да
        fusion_method : str
            Метод объединения модальностей:
            - 'weighted_voting': взвешенное голосование детекторов
            - 'feature_concat': конкатенация признаков перед OCSVM
            - 'ensemble': независимые детекторы с мета-классификатором
        use_pls : bool
            Использовать PLS для снижения размерности оптических данных
        """
        self.hc_bins = hc_bins
        self.oc_bins = oc_bins
        self.mass_bin_width = mass_bin_width
        self.fusion_method = fusion_method
        self.use_pls = use_pls
        
        # Скейлеры для каждой модальности
        self.ms_scaler = StandardScaler()
        self.optics_scaler = StandardScaler()
        self.combined_scaler = StandardScaler()
        
        # Модели
        self.ms_ocsvm = None
        self.optics_ocsvm = None
        self.combined_ocsvm = None
        self.isolation_forest = None
        self.pls_model = None
        
        # PCA модели для визуализации
        self.ms_pca = None
        self.optics_pca = None
        self.combined_pca = None
        
        # Веса модальностей
        self.modality_weights = {'ms': 0.5, 'optics': 0.5}
        
        # Имена признаков
        self.ms_feature_names = []
        self.optics_feature_names = []
        
    def calculate_vk_density_features(self, df, hc_range=(0, 2.5), oc_range=(0, 1.2)):
        """Расчет плотности заселенности квадратов диаграммы Ван-Кревелена."""
        mask = (df['H/C'] >= hc_range[0]) & (df['H/C'] <= hc_range[1]) & \
               (df['O/C'] >= oc_range[0]) & (df['O/C'] <= oc_range[1])
        filtered_df = df[mask]
        
        hc_edges = np.linspace(hc_range[0], hc_range[1], self.hc_bins + 1)
        oc_edges = np.linspace(oc_range[0], oc_range[1], self.oc_bins + 1)
        
        hist_2d, _, _ = np.histogram2d(
            filtered_df['H/C'], 
            filtered_df['O/C'],
            bins=[hc_edges, oc_edges],
            weights=filtered_df['intensity'] if 'intensity' in df.columns else None
        )
        
        if hist_2d.sum() > 0:
            hist_2d = hist_2d / hist_2d.sum()
        
        return hist_2d.flatten()
    
    def calculate_mass_spectrum_features(self, mass_df):
        """Извлечение признаков из масс-спектра."""
        mass_df = mass_df.sort_values('mass_start')
        formula_counts = mass_df['formula_count'].values
        
        if formula_counts.sum() > 0:
            formula_counts = formula_counts / formula_counts.sum()
        
        return formula_counts
    
    def calculate_molecular_class_features(self, df):
        """Расчет распределения по молекулярным классам."""
        class_counts = df['class'].value_counts()
        class_fractions = class_counts / class_counts.sum()
        
        expected_classes = ['hydrolyzable_tanins', 'phenylisopropanoids', 'terpenoids', 'carbohydrates', 'proteins','condensed_tanins','lipids']
        features = [class_fractions.get(cls, 0) for cls in expected_classes]
        
        return np.array(features)
    
    def calculate_aggregated_ms_features(self, df):
        """Расчет агрегированных молекулярных характеристик."""
        features = []
        
        # Средневзвешенные характеристики
        if 'intensity' in df.columns and df['intensity'].sum() > 0:
            weights = df['intensity'] / df['intensity'].sum()
            features.extend([
                np.average(df['H/C'], weights=weights),
                np.average(df['O/C'], weights=weights),
                np.average(df['calc_mass'], weights=weights)
            ])
        else:
            features.extend([df['H/C'].mean(), df['O/C'].mean(), df['calc_mass'].mean()])
        
        # Базовые статистики
        features.extend([
            df['H/C'].median(), df['O/C'].median(),
            df['H/C'].std(), df['O/C'].std(),
            df['H/C'].quantile(0.25), df['H/C'].quantile(0.75),
            df['O/C'].quantile(0.25), df['O/C'].quantile(0.75)
        ])
        
        # Индекс ароматичности (AI)
        if all(col in df.columns for col in ['C', 'H', 'O', 'N']):
            ai_mod = (1 + df['C'] - 0.5*df['O']  - 
                     0.5*(df['H'] + df['N'])) / \
                     (df['C'] - 0.5*df['O'] - df['N'] + 1e-10)
            ai_mod = ai_mod.clip(0, 1)
            features.extend([ai_mod.mean(), ai_mod.std(), 
                           (ai_mod > 0.5).mean(), (ai_mod > 0.67).mean()])
        else:
            features.extend([0, 0, 0, 0])
        
        # DBE
        if all(col in df.columns for col in ['C', 'H', 'N']):
            dbe = 1 + df['C'] - 0.5*df['H'] + 0.5*df['N']
            features.extend([dbe.mean(), dbe.std()])
        else:
            features.extend([0, 0])
        
        # Дополнительные маркеры надшламовых вод
        if 'O/C' in df.columns and 'H/C' in df.columns:
            sludge_zone = ((df['O/C'] > 0.6) & (df['H/C'] < 0.8)).mean()
            features.append(sludge_zone)
        else:
            features.append(0)
            
        if 'class' in df.columns:
            sulfur_frac = df['class'].str.contains('S', na=False).mean()
            features.append(sulfur_frac)
        else:
            features.append(0)
        
        return np.array(features)
    
    def extract_ms_features(self, formula_df, mass_df):
        """Извлечение всех масс-спектрометрических признаков."""
        features = []
        feature_names = []
        
        # Плотность Ван-Кревелена
        vk_density = self.calculate_vk_density_features(formula_df)
        features.extend(vk_density)
        feature_names.extend([f'VK_density_{i}' for i in range(len(vk_density))])
        
        # Распределение масс
        mass_features = self.calculate_mass_spectrum_features(mass_df)
        features.extend(mass_features)
        feature_names.extend([f'Mass_bin_{i}' for i in range(len(mass_features))])
        
        # Классы соединений
        if 'class' in formula_df.columns:
            class_features = self.calculate_molecular_class_features(formula_df)
            features.extend(class_features)
            feature_names.extend(['hydrolyzable_tanins', 'phenylisopropanoids', 'terpenoids', 'carbohydrates', 'proteins','condensed_tanins','lipids'])
        
        # Агрегированные характеристики
        agg_features = self.calculate_aggregated_ms_features(formula_df)
        features.extend(agg_features)
        feature_names.extend([
            'weighted_HC', 'weighted_OC', 'weighted_mass',
            'median_HC', 'median_OC', 'std_HC', 'std_OC',
            'q25_HC', 'q75_HC', 'q25_OC', 'q75_OC',
            'mean_AI', 'std_AI',
            'mean_DBE', 'std_DBE'
        ])
        
        self.ms_feature_names = feature_names
        
        return np.array(features)
    
    def extract_optics_features(self, optics_df, sample_id):
        """
        Извлечение оптических признаков для конкретного образца.
        
        Parameters:
        -----------
        optics_df : pd.DataFrame
            Датафрейм с оптическими данными
        sample_id : str
            ID образца для извлечения
            
        Returns:
        --------
        np.array : вектор оптических признаков
        """
        # Находим строку с нужным sample_id
        sample_data = optics_df[optics_df['sample_id'] == sample_id]
        
        if len(sample_data) == 0:
            raise ValueError(f"Образец {sample_id} не найден в оптических данных")
        
        sample_data = sample_data.iloc[0]
        
        features = []
        self.optics_feature_names = []
        
        # Извлекаем все числовые колонки кроме ID и класса
        exclude_cols = ['sample_id', 'Class', 'class','Subclass']
        
        for col in optics_df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(optics_df[col]):
                value = sample_data[col]
                
                # Обработка пропущенных значений
                if pd.isna(value):
                    value = 0
                
                features.append(value)
                self.optics_feature_names.append(col)
        
        return np.array(features)
    
    def calculate_optics_indices(self, optics_features, feature_names):
        """
        Расчет производных оптических индексов, важных для DOM.
        
        Parameters:
        -----------
        optics_features : np.array
            Массив оптических признаков
        feature_names : list
            Имена признаков
            
        Returns:
        --------
        dict : словарь с расчетными индексами
        """
        indices = {}
        
        # Создаем словарь признаков для удобства
        feat_dict = dict(zip(feature_names, optics_features))
        
        # SUVA254 (удельное УФ-поглощение)
        if 'Suva' in feat_dict:
            indices['Suva'] = feat_dict['Suva']
        
        # Индекс гумификации (HIX)
        if 'HIX' in feat_dict:
            indices['HIX'] = feat_dict['HIX']
        elif all(f'F{i}' in feat_dict for i in [3, 1, 2, 4, 5]):
            # Аппроксимация если нет готового HIX
            h_region = feat_dict.get('F3', 0) + feat_dict.get('F5', 0)
            l_region = feat_dict.get('F1', 0) + feat_dict.get('F2', 0) + feat_dict.get('F4', 0)
            indices['HIX'] = h_region / (l_region + 1e-10)
        else:
            indices['HIX'] = 0
        
        # Индекс (FIX - Fluorescence Index)
        if 'FIX' in feat_dict:
            indices['FIX'] = feat_dict['FIX']
        elif 'Em470_Ex370' in feat_dict and 'Em520_Ex370' in feat_dict:
            indices['FIX'] = feat_dict['Em470_Ex370'] / (feat_dict['Em520_Ex370'] + 1e-10)
        else:
            indices['FIX'] = 0
        """
        # Отношение пиков (для идентификации техногенного воздействия)
        if 'T_peak' in feat_dict and 'C_peak' in feat_dict:
            indices['T_C_ratio'] = feat_dict['T_peak'] / (feat_dict['C_peak'] + 1e-10)
        else:
            indices['T_C_ratio'] = 0
        """
        # Спектральный наклон (S275-295)
        if 'S_380_443' in feat_dict:
            indices['S_380_443'] = feat_dict['S_380_443']
        elif 'A275' in feat_dict and 'A295' in feat_dict:
            indices['S_380_443'] = np.log(feat_dict['A275'] / (feat_dict['A295'] + 1e-10)) / 20
        else:
            indices['S_380_443'] = 0
        """
        # Отношение наклонов (SR)
        if 'SR' in feat_dict:
            indices['SR'] = feat_dict['SR']
        elif 'S275_295' in indices and 'S350_400' in feat_dict:
            indices['SR'] = indices['S275_295'] / (feat_dict['S350_400'] + 1e-10)
        else:
            indices['SR'] = 0
        """
        if 'Component_1' in feat_dict:
            indices['Component_1'] = feat_dict['Component_1']        
        if 'Component_2' in feat_dict:
            indices['Component_2'] = feat_dict['Component_2']
        if 'Component_3' in feat_dict:
            indices['Component_3'] = feat_dict['Component_3']
            
        if 'Ag_275' in feat_dict:
            indices['Ag_275'] = feat_dict['Ag_275']  
        if 'Ag_380' in feat_dict:
            indices['Ag_380'] = feat_dict['Ag_380']
        if 'Asm_280' in feat_dict:
            indices['Asm_280'] = feat_dict['Asm_280']  
        if 'Asm_350' in feat_dict:
            indices['Asm_350'] = feat_dict['Asm_350']
            
        if 'B1' in feat_dict:
            indices['B1'] = feat_dict['B1']
        if 'B2' in feat_dict:
            indices['B2'] = feat_dict['B2']
        if 'E2E3' in feat_dict:
            indices['E2E3'] = feat_dict['E2E3']                        
        
        return indices
    
    def extract_combined_features(self, formula_df, mass_df, optics_df, sample_id):
        """
        Извлечение комбинированных признаков из всех модальностей.
        
        Returns:
        --------
        np.array : объединенный вектор признаков
        """
        # MS признаки
        ms_features = self.extract_ms_features(formula_df, mass_df)
        
        # Оптические признаки
        optics_features = self.extract_optics_features(optics_df, sample_id)
        
        # Расчетные оптические индексы
        optics_indices = self.calculate_optics_indices(optics_features, self.optics_feature_names)
        optics_indices_array = np.array(list(optics_indices.values()))
        
        # Объединение всех признаков
        combined_features = np.concatenate([ms_features, optics_features, optics_indices_array])
        
        return combined_features
    
    def fit_ms_detector(self, baikal_formula_dfs, baikal_mass_dfs):
        """Обучение детектора на MS данных."""
        baikal_features = []
        
        for form_df, mass_df in zip(baikal_formula_dfs, baikal_mass_dfs):
            features = self.extract_ms_features(form_df, mass_df)
            baikal_features.append(features)
        
        baikal_features = np.array(baikal_features)
        
        # Нормализация
        self.ms_scaler.fit(baikal_features)
        baikal_features_scaled = self.ms_scaler.transform(baikal_features)
        
        # Обучение OCSVM
        self.ms_ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.ms_ocsvm.fit(baikal_features_scaled)
        
        # PCA для визуализации
        self.ms_pca = PCA(n_components=2)
        self.ms_pca.fit(baikal_features_scaled)
        
        return baikal_features_scaled
    
    def fit_optics_detector(self, optics_df, baikal_sample_ids):
        """Обучение детектора на оптических данных."""
        baikal_features = []
        
        for sample_id in baikal_sample_ids:
            features = self.extract_optics_features(optics_df, sample_id)
            
            # Добавляем расчетные индексы
            indices = self.calculate_optics_indices(features, self.optics_feature_names)
            indices_array = np.array(list(indices.values()))
            
            combined = np.concatenate([features, indices_array])
            baikal_features.append(combined)
        
        baikal_features = np.array(baikal_features)
        
        # PLS снижение размерности (опционально)
        if self.use_pls and baikal_features.shape[0] > 10:
            self.pls_model = PLSRegression(n_components=min(3, baikal_features.shape[0]-1))
            # Создаем фиктивные Y для PLS (все единицы - "норма")
            Y = np.ones((baikal_features.shape[0], 1))
            self.pls_model.fit(baikal_features, Y)
            baikal_features = self.pls_model.transform(baikal_features)
        
        # Нормализация
        self.optics_scaler.fit(baikal_features)
        baikal_features_scaled = self.optics_scaler.transform(baikal_features)
        
        # Обучение OCSVM
        self.optics_ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.optics_ocsvm.fit(baikal_features_scaled)
        
        # PCA для визуализации
        self.optics_pca = PCA(n_components=2)
        self.optics_pca.fit(baikal_features_scaled)
        
        return baikal_features_scaled
    
    def fit(self, baikal_formula_dfs, baikal_mass_dfs, optics_df):
        """
        Обучение мультимодальной модели на фоновых пробах Байкала.
        
        Parameters:
        -----------
        baikal_formula_dfs : list
            Список датафреймов с формулами для проб Байкала
        baikal_mass_dfs : list
            Список датафреймов с распределением масс
        optics_df : pd.DataFrame
            Датафрейм с оптическими данными
        """
        # Получаем ID байкальских проб
        baikal_sample_ids = [df.attrs.get('name', f'Baikal_{i}') 
                            for i, df in enumerate(baikal_formula_dfs)]
        
        print("=== Обучение мультимодальной модели ===")
        print(f"Проб Байкала: {len(baikal_formula_dfs)}")
        
        # 1. Обучаем MS детектор
        print("\n1. Обучение масс-спектрометрического детектора...")
        ms_features = self.fit_ms_detector(baikal_formula_dfs, baikal_mass_dfs)
        print(f"   Размерность MS признаков: {ms_features.shape[1]}")
        
        # 2. Обучаем оптический детектор
        print("\n2. Обучение оптического детектора...")
        optics_features = self.fit_optics_detector(optics_df, baikal_sample_ids)
        print(f"   Размерность оптических признаков: {optics_features.shape[1]}")
        
        # 3. Обучаем комбинированный детектор (если нужно)
        if self.fusion_method == 'feature_concat':
            print("\n3. Обучение комбинированного детектора...")
            combined_features = []
            
            for i, (form_df, mass_df) in enumerate(zip(baikal_formula_dfs, baikal_mass_dfs)):
                sample_id = baikal_sample_ids[i]
                features = self.extract_combined_features(form_df, mass_df, optics_df, sample_id)
                combined_features.append(features)
            
            combined_features = np.array(combined_features)
            self.combined_scaler.fit(combined_features)
            combined_features_scaled = self.combined_scaler.transform(combined_features)
            
            self.combined_ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
            self.combined_ocsvm.fit(combined_features_scaled)
            
            self.combined_pca = PCA(n_components=2)
            self.combined_pca.fit(combined_features_scaled)
            
            print(f"   Размерность комбинированных признаков: {combined_features.shape[1]}")
        
        # 4. Вычисляем веса модальностей на основе их дисперсии в норме
        ms_scores = self.ms_ocsvm.score_samples(ms_features) # type: ignore
        optics_scores = self.optics_ocsvm.score_samples(optics_features) # type: ignore
        
        ms_std = np.std(ms_scores)
        optics_std = np.std(optics_scores)
        print(optics_std,'optics_std')
        # Веса обратно пропорциональны дисперсии (более стабильная модальность получает больший вес)
        total_inv_var = 1/ms_std + 1/optics_std
        print(total_inv_var,'total_inv_var')
        self.modality_weights['ms'] = (1/ms_std) / total_inv_var # type: ignore
        self.modality_weights['optics'] = (1/optics_std) / total_inv_var # type: ignore
        
        print(f"\n4. Веса модальностей:")
        print(f"   MS вес: {self.modality_weights['ms']:.3f}")
        print(f"   Optics вес: {self.modality_weights['optics']:.3f}")
        
        print("\n✓ Модель успешно обучена!")
        
        return self
    
    def predict_anomaly(self, formula_df, mass_df, optics_df, sample_id):
        """
        Предсказание аномальности для нового образца.
        
        Returns:
        --------
        dict : словарь с результатами для каждой модальности и комбинированным решением
        """
        results = {
            'sample_id': sample_id,
            'ms': {},
            'optics': {},
            'combined': {}
        }
        
        # 1. MS предсказание
        ms_features = self.extract_ms_features(formula_df, mass_df).reshape(1, -1)
        ms_features_scaled = self.ms_scaler.transform(ms_features)
        
        results['ms']['prediction'] = self.ms_ocsvm.predict(ms_features_scaled)[0] # type: ignore
        results['ms']['score'] = self.ms_ocsvm.score_samples(ms_features_scaled)[0] # type: ignore
        results['ms']['is_anomaly'] = results['ms']['prediction'] == -1
        results['ms']['confidence'] = 1 / (1 + np.exp(-results['ms']['score']))
        
        # 2. Optics предсказание
        optics_features = self.extract_optics_features(optics_df, sample_id)
        indices = self.calculate_optics_indices(optics_features, self.optics_feature_names)
        indices_array = np.array(list(indices.values()))
        optics_combined = np.concatenate([optics_features, indices_array]).reshape(1, -1)
        
        if self.use_pls and self.pls_model is not None:
            optics_combined = self.pls_model.transform(optics_combined)
        
        optics_scaled = self.optics_scaler.transform(optics_combined)
        
        results['optics']['prediction'] = self.optics_ocsvm.predict(optics_scaled)[0] # type: ignore
        results['optics']['score'] = self.optics_ocsvm.score_samples(optics_scaled)[0] # type: ignore
        results['optics']['is_anomaly'] = results['optics']['prediction'] == -1
        results['optics']['confidence'] = 1 / (1 + np.exp(-results['optics']['score']))
        results['optics']['indices'] = indices
        
        # 3. Комбинированное решение
        if self.fusion_method == 'weighted_voting':
            # Взвешенное голосование
            weighted_score = (self.modality_weights['ms'] * results['ms']['score'] + 
                            self.modality_weights['optics'] * results['optics']['score'])
            print(self.modality_weights['ms'],'ms')
            print(results['ms']['score'],'ms score')
            print(self.modality_weights['optics'],'optics')
            print(results['optics']['score'],'optics')
            results['combined']['score'] = weighted_score
            results['combined']['is_anomaly'] = weighted_score < -0.5
            results['combined']['confidence'] = 1 / (1 + np.exp(-weighted_score))
            
        elif self.fusion_method == 'feature_concat':
            # Конкатенация признаков
            combined_features = self.extract_combined_features(
                formula_df, mass_df, optics_df, sample_id
            ).reshape(1, -1)
            combined_scaled = self.combined_scaler.transform(combined_features)
            
            results['combined']['prediction'] = self.combined_ocsvm.predict(combined_scaled)[0] # type: ignore
            results['combined']['score'] = self.combined_ocsvm.score_samples(combined_scaled)[0] # type: ignore
            results['combined']['is_anomaly'] = results['combined']['prediction'] == -1
            results['combined']['confidence'] = 1 / (1 + np.exp(-results['combined']['score']))
        
        return results
    
    def analyze_molecular_anomalies(self, formula_df, baikal_formula_dfs):
        """Поиск конкретных аномальных молекул в образце."""
        result_df = formula_df.copy()
        
        # Объединяем все пробы Байкала
        all_baikal = pd.concat(baikal_formula_dfs, ignore_index=True)
        
        # Статистика встречаемости
        baikal_stats = all_baikal.groupby('formula').agg({
            'intensity': ['mean', 'std', 'max', 'count']
        }).reset_index()
        
        baikal_stats.columns = ['formula', 'baikal_mean_int', 'baikal_std_int', 
                               'baikal_max_int', 'baikal_occurrence']
        
        # Объединяем
        result_df = result_df.merge(baikal_stats, on='formula', how='left')
        
        # Маркеры аномальности
        result_df['is_new_formula'] = result_df['baikal_occurrence'].isna()
        result_df['intensity_ratio'] = result_df['intensity'] / result_df['baikal_max_int'].fillna(1)
        result_df['is_intensity_anomaly'] = result_df['intensity_ratio'] > 10
        result_df['molecular_anomaly'] = result_df['is_new_formula'] | result_df['is_intensity_anomaly']
        
        # Маркеры надшламовых вод
        if all(col in result_df.columns for col in ['H/C', 'O/C']):
            # Индекс ароматичности
            if all(col in result_df.columns for col in ['C', 'H', 'O', 'N', 'S', 'P']):
                ai_mod = (1 + result_df['C'] - 0.5*result_df['O'] - 
                         0.5*(result_df['H'] + result_df['N'])) / \
                         (result_df['C'] - 0.5*result_df['O'] - 
                          result_df['N'] + 1e-10)
                result_df['AI_mod'] = ai_mod.clip(0, 1)
            else:
                result_df['AI_mod'] = 0
            
            result_df['sludge_marker'] = (
                (result_df['O/C'] > 0.6) & 
                (result_df['H/C'] < 0.8) & 
                (result_df['AI_mod'] > 0.5)
            )
        
        return result_df
    
    def visualize_multimodal_results(self, results_list, baikal_formula_dfs, 
                                    baikal_mass_dfs, optics_df):
        """
        Комплексная визуализация мультимодальных результатов.
        """
        # Подготовка данных
        baikal_sample_ids = [df.attrs.get('name', f'Baikal_{i}') 
                            for i, df in enumerate(baikal_formula_dfs)]
        
        # Собираем scores для Байкала
        baikal_ms_scores = []
        baikal_optics_scores = []
        
        for i, (form_df, mass_df) in enumerate(zip(baikal_formula_dfs, baikal_mass_dfs)):
            sample_id = baikal_sample_ids[i]
            
            # MS score
            ms_features = self.extract_ms_features(form_df, mass_df).reshape(1, -1)
            ms_scaled = self.ms_scaler.transform(ms_features)
            baikal_ms_scores.append(self.ms_ocsvm.score_samples(ms_scaled)[0]) # type: ignore
            
            # Optics score
            optics_features = self.extract_optics_features(optics_df, sample_id)
            indices = self.calculate_optics_indices(optics_features, self.optics_feature_names)
            indices_array = np.array(list(indices.values()))
            optics_combined = np.concatenate([optics_features, indices_array]).reshape(1, -1)
            
            if self.use_pls and self.pls_model is not None:
                optics_combined = self.pls_model.transform(optics_combined)
            
            optics_scaled = self.optics_scaler.transform(optics_combined)
            baikal_optics_scores.append(self.optics_ocsvm.score_samples(optics_scaled)[0]) # type: ignore
        
        # Scores для надшламовых вод
        sludge_ms_scores = [r['ms']['score'] for r in results_list]
        sludge_optics_scores = [r['optics']['score'] for r in results_list]
        sludge_combined_scores = [r['combined']['score'] for r in results_list]
        
        # Создаем фигуру
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Сравнение MS и оптических scores
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(baikal_ms_scores, baikal_optics_scores, 
                   c='blue', label='Байкал', s=100, alpha=0.7, edgecolors='black')
        ax1.scatter(sludge_ms_scores, sludge_optics_scores, 
                   c='red', label='Надшлам', s=100, alpha=0.7, edgecolors='black', marker='^')
        
        # Добавляем пороговые линии
        ax1.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Закрашиваем квадранты
        ax1.axhspan(-0.5, ax1.get_ylim()[1], xmin=0.5, xmax=1, facecolor='green', alpha=0.1, label='Норма (обе)')
        ax1.axhspan(ax1.get_ylim()[0], -0.5, xmin=0.5, xmax=1, facecolor='yellow', alpha=0.1, label='Оптическая аномалия')
        ax1.axhspan(-0.5, ax1.get_ylim()[1], xmin=0, xmax=0.5, facecolor='orange', alpha=0.1, label='MS аномалия')
        ax1.axhspan(ax1.get_ylim()[0], -0.5, xmin=0, xmax=0.5, facecolor='red', alpha=0.1, label='Двойная аномалия')
        
        ax1.set_xlabel('MS Anomaly Score')
        ax1.set_ylabel('Optics Anomaly Score')
        ax1.set_title('Мультимодальный анализ аномалий')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Boxplot scores по модальностям
        ax2 = fig.add_subplot(2, 3, 2)
        data_to_plot = [
            baikal_ms_scores, sludge_ms_scores,
            baikal_optics_scores, sludge_optics_scores,
            [0]*len(baikal_ms_scores), sludge_combined_scores  # Байкал не имеет комбинированных
        ]
        labels = ['MS\nБайкал', 'MS\nНадшлам', 'Optics\nБайкал', 'Optics\nНадшлам', 'Combined\nБайкал', 'Combined\nНадшлам']
        
        bp = ax2.boxplot(data_to_plot, label=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral', 'lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Порог аномалии')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Распределение scores по модальностям')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA визуализация MS данных
        ax3 = fig.add_subplot(2, 3, 3)
        
        # Получаем MS PCA координаты
        baikal_ms_features = np.array([self.extract_ms_features(f, m) 
                                      for f, m in zip(baikal_formula_dfs, baikal_mass_dfs)])
        baikal_ms_scaled = self.ms_scaler.transform(baikal_ms_features)
        baikal_ms_pca = self.ms_pca.transform(baikal_ms_scaled) # type: ignore
        
        sludge_ms_features = np.array([self.extract_ms_features(
            r.get('formula_df'), r.get('mass_df')) for r in results_list])
        sludge_ms_scaled = self.ms_scaler.transform(sludge_ms_features)
        sludge_ms_pca = self.ms_pca.transform(sludge_ms_scaled) # type: ignore
        
        ax3.scatter(baikal_ms_pca[:, 0], baikal_ms_pca[:, 1], 
                   c='blue', label='Байкал', s=100, alpha=0.7, edgecolors='black')
        ax3.scatter(sludge_ms_pca[:, 0], sludge_ms_pca[:, 1], 
                   c='red', label='Надшлам', s=100, alpha=0.7, edgecolors='black', marker='^')
        
        ax3.set_xlabel(f'PC1 ({self.ms_pca.explained_variance_ratio_[0]:.1%})') # type: ignore
        ax3.set_ylabel(f'PC2 ({self.ms_pca.explained_variance_ratio_[1]:.1%})') # type: ignore
        ax3.set_title('PCA масс-спектрометрических данных')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Тепловая карта корреляции модальностей
        ax4 = fig.add_subplot(2, 3, 4)
        
        # Создаем матрицу корреляции между MS и оптическими признаками
        correlation_data = pd.DataFrame({
            'MS_Score': baikal_ms_scores + sludge_ms_scores,
            'Optics_Score': baikal_optics_scores + sludge_optics_scores,
            'Sample_Type': ['Baikal']*len(baikal_ms_scores) + ['Sludge']*len(sludge_ms_scores)
        })
        
        # Добавляем ключевые оптические индексы для sludge образцов
        for r in results_list:
            if 'indices' in r['optics']:
                for key, value in r['optics']['indices'].items():
                    if key not in correlation_data.columns:
                        correlation_data[key] = np.nan
                    correlation_data.loc[correlation_data['Sample_Type'] == 'Sludge', key] = value
        
        # Вычисляем корреляции только для sludge
        sludge_corr = correlation_data[correlation_data['Sample_Type'] == 'Sludge'].corr(numeric_only=True)
        
        # Оставляем только значимые признаки
        important_cols = ['MS_Score', 'Optics_Score', 'Suva', 'HIX', 'FIX', 'Component_1','Component_2','Component_3']
        available_cols = [col for col in important_cols if col in sludge_corr.columns]
        
        if len(available_cols) > 1:
            sns.heatmap(sludge_corr.loc[available_cols, available_cols], 
                       annot=True, cmap='coolwarm', center=0, ax=ax4)
            ax4.set_title('Корреляция MS и оптических показателей\n(надшламовые воды)')
        
        # 5. Диаграмма консенсуса
        ax5 = fig.add_subplot(2, 3, 5)
        
        # Классифицируем образцы
        classification = []
        for r in results_list:
            ms_anomaly = r['ms']['is_anomaly']
            optics_anomaly = r['optics']['is_anomaly']
            
            if not ms_anomaly and not optics_anomaly:
                classification.append('Норма')
            elif ms_anomaly and not optics_anomaly:
                classification.append('MS аномалия')
            elif not ms_anomaly and optics_anomaly:
                classification.append('Оптическая аномалия')
            else:
                classification.append('Двойная аномалия')
        
        class_counts = pd.Series(classification).value_counts()
        colors_pie = ['green', 'yellow', 'orange', 'red']
        ax5.pie(class_counts.to_numpy(), labels=class_counts.index, autopct='%1.1f%%',  # type: ignore
               colors=colors_pie, startangle=90)
        ax5.set_title('Классификация надшламовых проб')
        
        # 6. Сравнение комбинированного score с порогом
        ax6 = fig.add_subplot(2, 3, 6)
        
        sample_names = [r['sample_id'] for r in results_list]
        combined_scores = sludge_combined_scores
        confidences = [r['combined']['confidence'] for r in results_list]
        
        bars = ax6.bar(range(len(combined_scores)), combined_scores)
        
        # Цветовое кодирование
        for i, (score, bar) in enumerate(zip(combined_scores, bars)):
            if score < -1.0:
                bar.set_color('darkred')
            elif score < -0.5:
                bar.set_color('red')
            elif score < 0:
                bar.set_color('orange')
            else:
                bar.set_color('green')
            
            # Добавляем confidence как текст
            ax6.text(i, score, f'{confidences[i]:.2f}', 
                    ha='center', va='bottom' if score > 0 else 'top')
        
        ax6.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Порог аномалии')
        ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax6.set_xlabel('Образец')
        ax6.set_ylabel('Combined Anomaly Score')
        ax6.set_title('Комбинированная оценка аномальности\n(числа - confidence)')
        ax6.set_xticks(range(len(sample_names)))
        ax6.set_xticklabels(sample_names, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Вывод статистики
        print("\n=== Статистика мультимодального анализа ===")
        print(f"Всего надшламовых проб: {len(results_list)}")
        print(f"\nMS детектор:")
        print(f"  Аномалий: {sum(1 for r in results_list if r['ms']['is_anomaly'])}")
        print(f"  Средний score: {np.mean(sludge_ms_scores):.3f} ± {np.std(sludge_ms_scores):.3f}")
        print(f"\nОптический детектор:")
        print(f"  Аномалий: {sum(1 for r in results_list if r['optics']['is_anomaly'])}")
        print(f"  Средний score: {np.mean(sludge_optics_scores):.3f} ± {np.std(sludge_optics_scores):.3f}")
        print(f"\nКомбинированный детектор:")
        print(f"  Аномалий: {sum(1 for r in results_list if r['combined']['is_anomaly'])}")
        print(f"  Средний score: {np.mean(combined_scores):.3f} ± {np.std(combined_scores):.3f}")
        
        # Анализ согласованности модальностей
        agreement = sum(1 for r in results_list 
                       if r['ms']['is_anomaly'] == r['optics']['is_anomaly'])
        print(f"\nСогласованность модальностей: {agreement}/{len(results_list)} "
              f"({agreement/len(results_list):.1%})")
# ===== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====
optics_dataframe=pd.read_csv(r"D:\lab\KNP-analysis\Received_data\Statistics\all_descriptors.csv")
optics_dataframe['sample_id'] = optics_dataframe['sample_id'].apply(lambda x: x.replace('-2025',''))
# Инициализация модели
detector = MultiModal_FTICR_Optics_AnomalyDetector(
    hc_bins=30, 
    oc_bins=30,
    fusion_method='weighted_voting',  # или 'feature_concat'
    use_pls=True
)

# Обучение на байкальских пробах
detector.fit(
    baikal_formula_dfs=baikal_forms,
    baikal_mass_dfs=baikal_masses,
    optics_df=optics_dataframe
)

# Анализ надшламовых вод
results_list = []
for i, (form_df, mass_df) in enumerate(zip(sludge_forms, sludge_masses)):
    sample_id = form_df.attrs.get('name', f'Sludge_{i}')
    
    # Предсказание
    result = detector.predict_anomaly(
        formula_df=form_df,
        mass_df=mass_df,
        optics_df=optics_dataframe,
        sample_id=sample_id
    )
    
    # Добавляем датафреймы для дальнейшего анализа
    result['formula_df'] = form_df
    result['mass_df'] = mass_df
    
    results_list.append(result)
    
    # Вывод результатов
    print(f"\n{'='*50}")
    print(f"Образец: {sample_id}")
    print(f"{'='*50}")
    print(f"MS аномалия: {result['ms']['is_anomaly']} (score: {result['ms']['score']:.3f})")
    print(f"Оптическая аномалия: {result['optics']['is_anomaly']} (score: {result['optics']['score']:.3f})")
    print(f"Комбинированное решение: {result['combined']['is_anomaly']} "
          f"(score: {result['combined']['score']:.3f}, confidence: {result['combined']['confidence']:.2%})")
    
    # Оптические индексы
    if 'indices' in result['optics']:
        print("\nОптические индексы:")
        for key, value in result['optics']['indices'].items():
            print(f"  {key}: {value:.3f}")
    
    # Детальный анализ молекул для аномальных проб
    if result['ms']['is_anomaly']:
        molecular_analysis = detector.analyze_molecular_anomalies(form_df, baikal_forms)
        n_anomalous = molecular_analysis['molecular_anomaly'].sum()
        n_sludge = molecular_analysis['sludge_marker'].sum() if 'sludge_marker' in molecular_analysis.columns else 0
        n_sulfur = molecular_analysis['sulfur_marker'].sum() if 'sulfur_marker' in molecular_analysis.columns else 0
        
        print(f"\nМолекулярные аномалии:")
        print(f"  Всего молекул: {len(molecular_analysis)}")
        print(f"  Аномальных: {n_anomalous}")
        print(f"  Маркеров шлама: {n_sludge}")
        print(f"  S-содержащих: {n_sulfur}")

# Визуализация результатов
detector.visualize_multimodal_results(
    results_list, 
    baikal_forms, 
    baikal_masses, 
    optics_dataframe
)

# Экспорт результатов
results_df = pd.DataFrame([{
    'sample_id': r['sample_id'],
    'ms_score': r['ms']['score'],
    'ms_anomaly': r['ms']['is_anomaly'],
    'optics_score': r['optics']['score'],
    'optics_anomaly': r['optics']['is_anomaly'],
    'combined_score': r['combined']['score'],
    'combined_anomaly': r['combined']['is_anomaly'],
    'combined_confidence': r['combined']['confidence'],
    **r['optics'].get('indices', {})
} for r in results_list])

print("\n=== Сводная таблица результатов ===")
print(results_df.to_string())