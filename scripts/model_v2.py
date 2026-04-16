import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from scipy.stats import binned_statistic_2d
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
    mass_range: Tuple[float, float] = (250.0, 650.0),
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

optical_df=pd.read_csv(r"D:\lab\KNP-analysis\Received_data\Statistics\all_descriptors.csv")
optical_df['sample_id'] = optical_df['sample_id'].apply(lambda x: x.replace('-2025',''))
optical_df.drop(columns='lambda',inplace=True)


phenylisopropanoids_list = []
hydrolyzable_tanins_list = []
terpenoids_list = []
condensed_tanins_list = []
carbohydrates = []
proteins = []

for spectra in spectra_list:
    
    phenylisopropanoids_list.append(spectra['class'].value_counts().to_numpy()[0])
    hydrolyzable_tanins_list.append(spectra['class'].value_counts().to_numpy()[1])
    terpenoids_list.append(spectra['class'].value_counts().to_numpy()[2])    
    condensed_tanins_list.append(spectra['class'].value_counts().to_numpy()[3])
    carbohydrates.append(spectra['class'].value_counts().to_numpy()[4])
    proteins.append(spectra['class'].value_counts().to_numpy()[5])  
    
df_square = pd.DataFrame()
df_square['name'] = spectra_list_name
df_square['Фенил-\nизопроп-\nоиды'] = phenylisopropanoids_list
df_square['Гидролиз-\nованные\nтанины'] = hydrolyzable_tanins_list
df_square['Терпен-\nоиды'] = terpenoids_list
df_square['Конденсир-\nованные\nтанины'] = condensed_tanins_list
df_square['Углеводы'] = carbohydrates
df_square['Пептиды'] = proteins

numeric_cols = df_square.select_dtypes(include=[np.number]).columns
df_numeric = df_square[numeric_cols]
row_sums = df_numeric.sum(axis=1)
df_normalized = df_numeric.div(row_sums, axis=0)
df_square[numeric_cols] = df_normalized

df_square.to_excel(r"D:\lab\MassSpectraNew\processed_results\Хемотипирование.xlsx")

plot_stacked_barplot(df_square,show_values=False)
plt.savefig(r"D:\lab\MassSpectraNew\processed_results\Хемотипирование.png")
plt.close()

def visualize_average_formulas_distribution(
    spectra_list: List[pd.DataFrame],
    mass_range: Tuple[float, float] = (250.0, 650.0),
    fixed_ranges: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[int, int] = (15, 10),
    style: str = 'seaborn-v0_8-darkgrid'
) -> Tuple[pd.DataFrame, dict]:
    """
    Визуализирует распределение среднего количества формул в интервалах 1 Да.
    
    Параметры:
    ----------
    spectra_list : List[pd.DataFrame]
        Список DataFrame с колонкой 'calc_mass' и атрибутом attrs['name']
    mass_range : Tuple[float, float], default (250.0, 650.0)
        Общий диапазон масс для анализа
    fixed_ranges : List[Tuple[float, float]], default None
        Фиксированные диапазоны для столбчатых диаграмм.
        Если None, используются [(250, 350), (450, 550), (550, 650)]
    figsize : Tuple[int, int], default (15, 10)
        Размер фигуры
    style : str, default 'seaborn-v0_8-darkgrid'
        Стиль оформления графиков
    
    Возвращает:
    -----------
    avg_df : pd.DataFrame
        DataFrame со средними значениями для каждого образца
    stats : dict
        Статистика по каждому фиксированному диапазону
    """
    
    # Установка стиля
    plt.style.use(style)
    sns.set_palette("husl")
    
    # Фиксированные диапазоны по умолчанию
    if fixed_ranges is None:
        fixed_ranges = [(250, 350), (450, 550), (550, 650)]
    
    # Вычисление средних значений для каждого образца
    avg_df = average_formulas_per_mass_interval(
        spectra_list=spectra_list,
        mass_range=mass_range,
        verbose=False
    )
    
    if avg_df.empty:
        print("Нет данных для визуализации")
        return avg_df, {}
    
    # Создание фигуры с подграфиками
    fig = plt.figure(figsize=figsize)
    
    # 1. Основная гистограмма распределения средних значений
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # Гистограмма с KDE
    n, bins, patches = ax1.hist(
        avg_df['avg_formula_count'], 
        bins=15, 
        edgecolor='black', 
        alpha=0.7,
        density=False,
        label='Распределение'
    )
    
    # Добавление KDE
    from scipy import stats
    kde = stats.gaussian_kde(avg_df['avg_formula_count'])
    x_range = np.linspace(avg_df['avg_formula_count'].min(), 
                          avg_df['avg_formula_count'].max(), 200)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    ax1_twin.set_ylabel('Плотность', fontsize=10, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Статистика
    mean_val = avg_df['avg_formula_count'].mean()
    median_val = avg_df['avg_formula_count'].median()
    std_val = avg_df['avg_formula_count'].std()
    
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Медиана: {median_val:.2f}')
    
    ax1.set_xlabel('Среднее количество формул на интервал 1 Да', fontsize=12)
    ax1.set_ylabel('Частота (количество образцов)', fontsize=12)
    ax1.set_title(f'Распределение среднего количества формул\n(диапазон масс: {mass_range[0]}-{mass_range[1]} Да)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Добавление текста со статистикой
    stats_text = f'n = {len(avg_df)}\nСреднее = {mean_val:.2f}\nМедиана = {median_val:.2f}\nСтд. откл. = {std_val:.2f}'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Box plot
    ax2 = plt.subplot(2, 3, 3)
    bp = ax2.boxplot(avg_df['avg_formula_count'], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Среднее количество формул', fontsize=12)
    ax2.set_title('Box-plot распределения', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['Все образцы'])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3-5. Столбчатые диаграммы для фиксированных диапазонов
    stats = {}
    
    for idx, (start, end) in enumerate(fixed_ranges, 1):
        ax = plt.subplot(2, 3, 3 + idx)
        
        # Вычисление средних значений для конкретного диапазона
        range_avg_df = average_formulas_per_mass_interval(
            spectra_list=spectra_list,
            mass_range=(start, end),
            verbose=False
        )
        
        if not range_avg_df.empty:
            stats[f'{start}-{end}'] = {
                'mean': range_avg_df['avg_formula_count'].mean(),
                'std': range_avg_df['avg_formula_count'].std(),
                'median': range_avg_df['avg_formula_count'].median(),
                'min': range_avg_df['avg_formula_count'].min(),
                'max': range_avg_df['avg_formula_count'].max(),
                'data': range_avg_df
            }
            
            # Сортировка по убыванию для лучшей визуализации
            sorted_df = range_avg_df.sort_values('avg_formula_count', ascending=False)
            
            # Создание столбчатой диаграммы
            #colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_df)))
            colors = plt.colormaps['viridis'](np.linspace(0.2, 0.8, len(sorted_df)))
            bars = ax.bar(range(len(sorted_df)), sorted_df['avg_formula_count'], 
                         color=colors, edgecolor='black', alpha=0.8)
            
            # Добавление значений на столбцы
            for i, (bar, val) in enumerate(zip(bars, sorted_df['avg_formula_count'])):
                if i < 10:  # Показываем значения только для первых 10, чтобы не загромождать
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=45)
            
            ax.set_xlabel('Образцы', fontsize=10)
            ax.set_ylabel('Среднее количество формул', fontsize=10)
            ax.set_title(f'Диапазон масс: {start}-{end} Да\n(среднее: {range_avg_df["avg_formula_count"].mean():.2f})', 
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df.index, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Анализ распределения среднего количества формул в интервалах 1 Да', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Дополнительный график: сравнение диапазонов
    fig2, ax_comp = plt.subplots(figsize=(12, 6))
    
    # Подготовка данных для сравнения
    comparison_data = []
    for range_name, range_stats in stats.items():
        comparison_data.append({
            'Диапазон': range_name,
            'Среднее': range_stats['mean'],
            'Стд. отклонение': range_stats['std'],
            'Медиана': range_stats['median']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Группированные столбцы
    x = np.arange(len(comp_df))
    width = 0.25
    
    bars1 = ax_comp.bar(x - width, comp_df['Среднее'], width, label='Среднее', alpha=0.8, edgecolor='black')
    bars2 = ax_comp.bar(x, comp_df['Медиана'], width, label='Медиана', alpha=0.8, edgecolor='black')
    bars3 = ax_comp.bar(x + width, comp_df['Стд. отклонение'], width, label='Стд. отклонение', alpha=0.8, edgecolor='black')
    
    ax_comp.set_xlabel('Диапазон масс (Да)', fontsize=12)
    ax_comp.set_ylabel('Значение', fontsize=12)
    ax_comp.set_title('Сравнение статистик по фиксированным диапазонам', fontsize=14, fontweight='bold')
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(comp_df['Диапазон'])
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3, axis='y')
    
    # Добавление значений на столбцы
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax_comp.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print("\n" + "="*80)
    print("СТАТИСТИКА ПО ФИКСИРОВАННЫМ ДИАПАЗОНАМ")
    print("="*80)
    for range_name, range_stats in stats.items():
        print(f"\nДиапазон {range_name} Да:")
        print(f"  Среднее: {range_stats['mean']:.2f}")
        print(f"  Медиана: {range_stats['median']:.2f}")
        print(f"  Стд. отклонение: {range_stats['std']:.2f}")
        print(f"  Min: {range_stats['min']:.2f}")
        print(f"  Max: {range_stats['max']:.2f}")
        print(f"  Количество образцов: {len(range_stats['data'])}")
    
    return avg_df, stats


# Пример использования:
# Визуализация
avg_df, stats = visualize_average_formulas_distribution(
    spectra_list=spectra_list,
    mass_range=(550, 700),
    fixed_ranges=[(550, 600),(600,650), (650, 700)],
    figsize=(16, 12)
)
      