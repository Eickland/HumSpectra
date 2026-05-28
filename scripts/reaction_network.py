import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
from itertools import combinations
from contextlib import contextmanager
from collections import defaultdict
from scipy import stats

import HumSpectra.utilits as ut

def merge_mass_spectra_fast(
    spectra: List[pd.DataFrame],
    spectra_list_name,
    columns: Optional[List[str]] = None,
    tolerance: float = 0.01,
    intensity_col: str = 'intensity',
    use_hash: bool = True,
    precision: int = 6,
    
) -> pd.DataFrame:
    """
    Оптимизированная версия объединения масс-спектров.
    
    Оптимизации:
    1. Предварительная нормализация и округление значений
    2. Использование хеширования вместо попарного сравнения
    3. Векторизация операций
    4. Минимизация копирований данных
    """
    
    if not spectra:
        raise ValueError("Список спектров не может быть пустым")
    
    # Устанавливаем столбцы по умолчанию
    if columns is None:
        columns = ['C', 'H', 'O', 'N', 'S']
    
    # Подготавливаем все спектры
    processed_spectra = []
    
    for i, spec in enumerate(spectra):
        # Проверяем наличие столбца интенсивности
        if intensity_col not in spec.columns:
            raise ValueError(f"Столбец '{intensity_col}' отсутствует в спектре {i}")
        
        # Создаем копию только нужных столбцов
        spec_copy = spec[columns + [intensity_col]].copy()
        
        # Заполняем отсутствующие столбцы нулями
        for col in columns:
            if col not in spec_copy.columns:
                spec_copy[col] = 0
        
        processed_spectra.append(spec_copy)
    
    if use_hash:
        return _merge_using_hash(processed_spectra, columns, tolerance, intensity_col, precision, spectra_list_name)
    else:
        return _merge_using_sorting(processed_spectra, columns, tolerance, intensity_col)


def _merge_using_hash(
    spectra: List[pd.DataFrame],
    columns: List[str],
    tolerance: float,
    intensity_col: str,
    precision: int,
    spectra_list_name
) -> pd.DataFrame:
    """
    Использует хеширование для быстрого поиска совпадений.
    Сложность: O(n * k) где n - общее количество пиков, k - количество спектров
    """
    
    # Определяем шаг дискретизации на основе tolerance
    step = tolerance / 2
    factor = 10 ** precision
    
    # Словарь: хеш -> список кортежей (индекс_спектра, индекс_пика, интенсивность)
    hash_map = defaultdict(list)
    
    # Заполняем хеш-таблицу
    for spec_idx, spec in enumerate(spectra):
        # Создаем дискретизированные значения
        discretized = spec[columns].round(precision)
        
        # Создаем хеш для каждой строки
        for row_idx, row in discretized.iterrows():
            # Создаем строковый ключ из округленных значений
            hash_key = tuple(row.values)
            
            # Сохраняем информацию о пике
            hash_map[hash_key].append((
                spec_idx,
                row_idx,
                spec.loc[row_idx] # type: ignore
            ))
    
    # Находим пики, присутствующие во всех спектрах
    n_spectra = len(spectra)
    common_peaks = []
    
    for hash_key, peaks in hash_map.items():
        # Проверяем, есть ли пик во всех спектрах
        if len(peaks) == n_spectra:
            # Сортируем по индексу спектра для консистентности
            peaks.sort(key=lambda x: x[0])
            
            # Собираем данные
            peak_data = {}
            
            # Добавляем химические значения
            for i, col in enumerate(columns):
                peak_data[col] = hash_key[i]
            
            common_peaks.append(peak_data)
            
            # Добавляем интенсивности для всех спектров
            for spec_idx, (_, _, row) in enumerate(peaks):
                peak_data[f'{spectra_list_name[spec_idx]}'] = row['intensity'] 
                   
    if not common_peaks:
        return pd.DataFrame()
    
    return pd.DataFrame(common_peaks)


def _merge_using_sorting(
    spectra: List[pd.DataFrame],
    columns: List[str],
    tolerance: float,
    intensity_col: str
) -> pd.DataFrame:
    """
    Использует сортировку и скользящее окно для поиска совпадений.
    Эффективно при большом количестве спектров.
    """
    
    # Создаем единый DataFrame со всеми спектрами
    all_peaks = []
    
    for spec_idx, spec in enumerate(spectra):
        temp = spec[columns + [intensity_col]].copy()
        temp['_spec_idx'] = spec_idx
        temp['_orig_idx'] = temp.index
        all_peaks.append(temp)
    
    combined = pd.concat(all_peaks, ignore_index=True)
    
    # Сортируем по первому столбцу для эффективного поиска
    sort_col = columns[0]
    combined = combined.sort_values(sort_col).reset_index(drop=True)
    
    # Скользящее окно для поиска близких значений
    n_spectra = len(spectra)
    common_groups = []
    used_indices = set()
    
    for i in range(len(combined)):
        if i in used_indices:
            continue
        
        current = combined.iloc[i]
        current_vals = np.array([current[col] for col in columns])
        
        # Ищем все пики в окне tolerance
        window_indices = []
        
        # Просматриваем соседей (вперед и назад)
        for j in range(max(0, i - 100), min(len(combined), i + 100)):
            if j in used_indices:
                continue
            
            candidate = combined.iloc[j]
            candidate_vals = np.array([candidate[col] for col in columns])
            
            # Быстрая проверка на близость
            if np.all(np.abs(current_vals - candidate_vals) <= tolerance):
                window_indices.append(j)
        
        if not window_indices:
            continue
        
        # Группируем по спектрам
        spec_groups = defaultdict(list)
        for idx in window_indices:
            row = combined.iloc[idx]
            spec_groups[row['_spec_idx']].append(idx)
        
        # Проверяем, есть ли пики во всех спектрах
        if len(spec_groups) == n_spectra:
            # Берем по одному лучшему пику из каждого спектра
            peak_indices = []
            for spec_idx in range(n_spectra):
                if spec_idx in spec_groups:
                    # Выбираем пик с максимальной интенсивностью
                    indices = spec_groups[spec_idx]
                    best_idx = max(indices, key=lambda x: combined.iloc[x][intensity_col])
                    peak_indices.append(best_idx)
                    used_indices.add(best_idx)
            
            # Создаем запись для общего пика
            peak_data = {}
            representative = combined.iloc[peak_indices[0]]
            
            # Добавляем химические значения
            for col in columns:
                peak_data[col] = representative[col]
            
            # Добавляем интенсивности
            for pos, idx in enumerate(peak_indices):
                intensity = combined.iloc[idx][intensity_col]
                if pos == 0:
                    peak_data[intensity_col] = intensity
                peak_data[f'{intensity_col}_spec{pos}'] = intensity
            
            common_groups.append(peak_data)
    
    if not common_groups:
        return pd.DataFrame()
    
    return pd.DataFrame(common_groups)

@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} сек")

wdir_spectra = r"C:\Users\Kirill\Desktop\Lab\Process_mass_spectra\process_result\mzlist"
wdir_statistic = r"C:\Users\Kirill\Desktop\Lab\Process_mass_spectra\process_result\statistic"

spectra_list = []
spectra_stat_list = []
spectra_class_list = []
spectra_list_name = []

with timer("Чтение спектров"):
    for path in Path.rglob(Path(wdir_spectra),'*.csv'):
        
        spectra = pd.read_csv(path)
        spectra_name = ut.extract_name_from_path(str(path))
        spectra_name = spectra_name.split(sep='__')[0]
        
        spectra.attrs['name'] = spectra_name
        
        if 'pos' in spectra_name:
            continue
        
        if 'Me' not in spectra_name:
            continue
        
        if 'O' in spectra_name:
            continue
        
        spectra_list.append(spectra)
        spectra_list_name.append(spectra_name)
        
if spectra_list:
    last_spectrum = spectra_list.pop()
    last_name = spectra_list_name.pop()
    
    spectra_list.insert(0, last_spectrum)
    spectra_list_name.insert(0, last_name) 
           
merge_spectra = merge_mass_spectra_fast(spectra_list,spectra_list_name)
merge_spectra.to_excel(r"C:\Users\Kirill\Desktop\Lab\Process_mass_spectra\process_result\merge_spectra_Me.xlsx")

def _assign_formula(row):
    
    if int(row['S']) > 1 and int(row['N']) >1:
        return 'C'+ str(int(row['C']))+'H'+str(int(row['H'])) +'O'+str(int(row['O']))+'N'+str(int(row['N']))+ 'S'+str(int(row['S']))
    
    elif int(row['S']) > 1:
        return 'C'+str(int(row['C']))+'H'+str(int(row['H'])) +'O'+str(int(row['O']))+ 'S'+str(int(row['S']))
    elif int(row['N']) > 1:
        return 'C'+str(int(row['C']))+'H'+str(int(row['H'])) +'O'+str(int(row['O']))+ 'N'+str(int(row['N']))
    else:        
        return 'C'+str(int(row['C']))+'H'+str(int(row['H'])) +'O'+str(int(row['O']))

class TimeSeriesVisualizer:
    """Класс для визуализации изменений интенсивностей соединений во времени"""
    
    def __init__(self, df: pd.DataFrame, formula_cols: List[str] = ['C', 'H', 'O', 'N', 'S']):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с колонками формул и образцов
        formula_cols : List[str]
            Колонки с составом формулы
        """
        self.df = df
        self.formula_cols = formula_cols
        self.sample_cols = [col for col in df.columns if col not in formula_cols]
        
        # Создаем уникальный идентификатор для каждого соединения
        #self.df['formula_id'] = self.df[formula_cols].astype(str).agg('-'.join, axis=1)
        self.df['formula_id'] = self.df[formula_cols].apply(lambda x: _assign_formula(x), axis=1)
    

        
    def get_top_compounds(self, metric: str = 'max', n: int = 10) -> pd.DataFrame:
        """
        Получает топ N соединений по выбранной метрике
        
        Parameters:
        -----------
        metric : str
            'max', 'mean', 'sum' или 'variance' - метрика для отбора
        n : int
            Количество соединений
        """
        if metric == 'max':
            values = self.df[self.sample_cols].max(axis=1)
        elif metric == 'mean':
            values = self.df[self.sample_cols].mean(axis=1)
        elif metric == 'sum':
            values = self.df[self.sample_cols].sum(axis=1)
        elif metric == 'variance':
            values = self.df[self.sample_cols].var(axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        top_indices = values.nlargest(n).index
        return self.df.loc[top_indices]
    
    def plot_all_trajectories(self, n_plots: int | None = None, 
                            compounds_per_plot: int = 15,
                            figsize: Tuple[int, int] = (15, 10)):
        """
        Построение всех траекторий соединений на нескольких графиках.
        
        Parameters:
        -----------
        n_plots : int | None
            Количество графиков. Если None, рассчитывается автоматически
        compounds_per_plot : int
            Количество соединений на одном графике
        figsize : Tuple[int, int]
            Размер каждого графика
        """
        
        data = self.df
        total_compounds = len(data)
        
        # Рассчитываем количество графиков, если не указано
        if n_plots is None:
            n_plots = int(np.ceil(total_compounds / compounds_per_plot))
        
        # Проверка типа
        if not isinstance(n_plots, int):
            raise ValueError(f'n_plots должен быть целым числом, получено {type(n_plots)}')
        
        # Рассчитываем размер чанка (сколько соединений на график)
        chunk_size = int(np.ceil(total_compounds / n_plots))
        
        print(f"Всего соединений: {total_compounds}")
        print(f"Будет построено графиков: {n_plots}")
        print(f"Соединений на график: ~{chunk_size}")
        
        # Создаем графики
        for plot_num in range(n_plots):
            # Определяем границы текущего чанка
            start_idx = plot_num * chunk_size
            end_idx = min((plot_num + 1) * chunk_size, total_compounds)
            
            # Создаем новый график для каждого чанка
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # Проверяем, есть ли данные в текущем чанке
            if start_idx >= total_compounds:
                break
                
            # Отображаем соединения из текущего чанка
            for idx, row in data.iloc[start_idx:end_idx].iterrows():
                intensities = row[self.sample_cols].to_numpy()
                formula = row['formula_id']
                ax.plot(spectra_list_name, intensities, 
                    marker='o', linewidth=2, markersize=4, label=formula)
            
            # Настройки графика
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('Intensity', fontsize=12)
            ax.set_title(f'Compound Trajectories (Plot {plot_num + 1}/{n_plots}, '
                        f'Compounds {start_idx + 1}-{end_idx})', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
               
    def plot_trajectories(self, n_compounds: int = 10, 
                          normalize: bool = True,
                          metric: str = 'max',
                          figsize: Tuple[int, int] = (15, 10)):
        """
        Построение траекторий изменения интенсивностей
        
        Parameters:
        -----------
        n_compounds : int
            Количество соединений для отображения
        normalize : bool
            Нормализовать ли интенсивности (макс=1)
        metric : str
            Метрика для отбора соединений
        figsize : Tuple[int, int]
            Размер рисунка
        """
        # Выбираем топ соединения
        top_compounds = self.get_top_compounds(metric=metric, n=n_compounds)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Линейные траектории
        ax1 = axes[0, 0]
        for idx, row in top_compounds.iterrows():
            intensities = row[self.sample_cols].to_numpy()
            formula = row['formula_id']
            
            if normalize:
                intensities = intensities / intensities.max()
            
            ax1.plot(range(len(self.sample_cols)), intensities, 
                    marker='o', linewidth=2, markersize=4, label=formula)
        
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Normalized Intensity' if normalize else 'Intensity', fontsize=12)
        ax1.set_title(f'Top {n_compounds} Compound Trajectories\n(selected by {metric})', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Тепловая карта изменений
        ax2 = axes[0, 1]
        data = top_compounds[self.sample_cols].values
        if normalize:
            data = data / data.max(axis=1, keepdims=True)
        
        im = ax2.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Compound Index', fontsize=12)
        ax2.set_title('Intensity Heatmap', fontsize=14)
        ax2.set_xticks(range(len(self.sample_cols)))
        ax2.set_xticklabels([f'S{i}' for i in range(len(self.sample_cols))], rotation=45)
        ax2.set_yticks(range(len(top_compounds)))
        ax2.set_yticklabels(top_compounds['formula_id'].values, fontsize=8)
        plt.colorbar(im, ax=ax2, label='Normalized Intensity' if normalize else 'Intensity')
        
        # 3. Изменение относительно исходного
        ax3 = axes[1, 0]
        for idx, row in top_compounds.iterrows():
            intensities = row[self.sample_cols].values
            if intensities[0] > 0:
                fold_change = intensities / intensities[0]
            else:
                fold_change = intensities
            formula = row['formula_id']
            
            ax3.plot(range(len(self.sample_cols)), fold_change, 
                    marker='s', linewidth=2, markersize=4, label=formula)
        
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Initial level')
        ax3.set_xlabel('Sample Index', fontsize=12)
        ax3.set_ylabel('Fold Change', fontsize=12)
        ax3.set_title('Fold Change Relative to Initial Sample', fontsize=14)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot распределений
        ax4 = axes[1, 1]
        all_intensities = self.df[self.sample_cols].values.flatten()
        ax4.hist(all_intensities, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Intensity', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Overall Intensity Distribution', fontsize=14)
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_clustered_heatmap(self, n_compounds: int = 20, normalize: bool = True, 
                               figsize: Tuple[int, int] = (12, 10)):
        """
        Кластеризованная тепловая карта для выявления паттернов
        
        Parameters:
        -----------
        n_compounds : int
            Количество соединений для отображения
        normalize : bool
            Нормализовать ли интенсивности
        figsize : Tuple[int, int]
            Размер рисунка
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Выбираем топ соединения по максимальной интенсивности
        top_compounds = self.get_top_compounds(metric='max', n=n_compounds)
        data = top_compounds[self.sample_cols].values
        
        if normalize:
            data = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True))
        
        # Кластеризация
        row_linkage = linkage(pdist(data), method='ward')
        col_linkage = linkage(pdist(data.T), method='ward')
        
        fig = plt.figure(figsize=figsize)
        
        # Дендрограмма строк
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
        dendrogram(row_linkage, ax=ax1, no_labels=True)
        ax1.set_title('Compound Clusters')
        ax1.set_ylabel('Distance')
        
        # Тепловая карта
        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
        im = ax2.imshow(data, aspect='auto', cmap='coolwarm', interpolation='nearest')
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Compound Index', fontsize=12)
        ax2.set_title('Clustered Heatmap', fontsize=14)
        ax2.set_xticks(range(len(self.sample_cols)))
        ax2.set_xticklabels([f'S{i}' for i in range(len(self.sample_cols))], rotation=45)
        plt.colorbar(im, ax=ax2, label='Normalized Intensity')
        
        # Дендрограмма колонок
        ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
        dendrogram(col_linkage, ax=ax3, orientation='top', no_labels=True)
        ax3.set_title('Sample Clusters')
        ax3.set_xlabel('Distance')
        
        ax3.set_visible(True)
        plt.tight_layout()
        plt.show()
        
    def find_patterns(self, pattern_type: str = 'increasing', 
                     min_fold_change: float = 2.0,
                     n_show: int = 10) -> pd.DataFrame:
        """
        Поиск соединений с определенными паттернами изменения
        
        Parameters:
        -----------
        pattern_type : str
            'increasing', 'decreasing', 'peak', 'valley', 'stable'
        min_fold_change : float
            Минимальное изменение для паттернов возрастания/убывания
        n_show : int
            Количество соединений для отображения
        
        Returns:
        --------
        pd.DataFrame : DataFrame с найденными паттернами
        """
        results = []
        
        for idx, row in self.df.iterrows():
            intensities = row[self.sample_cols].to_numpy()
            formula = row['formula_id']
            
            # Вычисляем метрики
            fold_change_end = intensities[-1] / intensities[0] if intensities[0] > 0 else np.inf
            max_intensity = intensities.max()
            max_idx = intensities.argmax()
            min_intensity = intensities.min()
            min_idx = intensities.argmin()
            
            # Определяем паттерн
            pattern = None
            score = 0
            
            if pattern_type == 'increasing':
                if fold_change_end >= min_fold_change and all(np.diff(intensities) > -0.1 * intensities.mean()):
                    pattern = 'increasing'
                    score = fold_change_end
                    
            elif pattern_type == 'decreasing':
                if fold_change_end <= 1/min_fold_change and all(np.diff(intensities) < 0.1 * intensities.mean()):
                    pattern = 'decreasing'
                    score = 1/fold_change_end if fold_change_end > 0 else np.inf
                    
            elif pattern_type == 'peak':
                if max_idx not in [0, len(intensities)-1] and max_intensity > intensities[0] * min_fold_change:
                    pattern = 'peak'
                    score = max_intensity / intensities[0]
                    
            elif pattern_type == 'valley':
                if min_idx not in [0, len(intensities)-1] and min_intensity < intensities[0] / min_fold_change:
                    pattern = 'valley'
                    score = intensities[0] / min_intensity if min_intensity > 0 else np.inf
                    
            elif pattern_type == 'stable':
                if np.std(intensities) / np.mean(intensities) < 0.2:
                    pattern = 'stable'
                    score = -np.std(intensities)  # Чем меньше стандартное отклонение, тем лучше
            
            if pattern:
                results.append({
                    'formula': formula,
                    'pattern_score': score,
                    'initial_intensity': intensities[0],
                    'max_intensity': max_intensity,
                    'final_intensity': intensities[-1],
                    'fold_change': fold_change_end,
                    **{f'S{i}': intensities[i] for i in range(len(intensities))}
                })
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values('pattern_score', ascending=False).head(n_show)
        
        return result_df
    
    def plot_individual_compounds(self, formulas: List[str], normalize: bool = True,
                                 figsize: Tuple[int, int] = (15, 8)):
        """
        Детальная визуализация для конкретных соединений
        
        Parameters:
        -----------
        formulas : List[str]
            Список формул для визуализации (формат 'C-H-O-N-S')
        normalize : bool
            Нормализовать ли интенсивности
        figsize : Tuple[int, int]
            Размер рисунка
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Основной график
        ax1 = axes[0]
        for formula in formulas:
            row = self.df[self.df['formula_id'] == formula]
            if row.empty:
                print(f"Formula {formula} not found")
                continue
                
            intensities = row[self.sample_cols].values[0]
            if normalize:
                intensities = intensities / intensities.max()
            
            ax1.plot(range(len(self.sample_cols)), intensities, 
                    marker='o', linewidth=2, markersize=6, label=formula)
        
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Normalized Intensity' if normalize else 'Intensity', fontsize=12)
        ax1.set_title('Compound Trajectories', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # График с доверительными интервалами (если есть повторы)
        ax2 = axes[1]
        
        # Добавляем статистику тренда
        for formula in formulas:
            row = self.df[self.df['formula_id'] == formula]
            if row.empty:
                continue
                
            intensities = row[self.sample_cols].values[0]
            
            # Линейная регрессия для определения тренда
            x = np.arange(len(intensities))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, intensities)
            
            ax2.plot(range(len(self.sample_cols)), intensities, 
                    marker='s', linewidth=2, markersize=5, label=f"{formula} (p={p_value:.3f})")
            
            # Добавляем линию тренда
            trend_line = slope * x + intercept # type: ignore
            ax2.plot(x, trend_line, '--', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Intensity', fontsize=12)
        ax2.set_title('Trend Analysis with Linear Regression', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
visualizer = TimeSeriesVisualizer(merge_spectra)
print(f"Samples: {visualizer.sample_cols}")
visualizer.plot_trajectories(n_compounds=15, normalize=True, metric='max')
visualizer.plot_clustered_heatmap(n_compounds=30, normalize=True)
increasing = visualizer.find_patterns('increasing', min_fold_change=1.5, n_show=5)
print(increasing[['formula', 'fold_change', 'initial_intensity', 'final_intensity']])
peaks = visualizer.find_patterns('peak', min_fold_change=1.5, n_show=5)
print(peaks[['formula', 'pattern_score', 'max_intensity']])

visualizer.plot_all_trajectories()