import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

elenemtal_colors_dict = {
    'CHO' : 'blue',
    'CHON' : 'orange',
    'CHOS' : 'green',
    'CHONS' : 'red',
    'Total' : 'gray'
}

set2_colors = sns.color_palette('Set2').as_hex()
perminova_class_color_dictionary = {
    'hydrolyzable_tanins': set2_colors[0],
    'phenylisopropanoids': set2_colors[1],
    'terpenoids': set2_colors[2],
    'carbohydrates': set2_colors[3],
    'proteins': set2_colors[4],
    'condensed_tanins': set2_colors[5],
    'lipids': set2_colors[6]
}

perminova_class_names_translate_dict = {
    'hydrolyzable_tanins': 'Гидролиз-\nованные\nтанины',
    'phenylisopropanoids': 'Фенил-\nизопроп-\nоиды',
    'terpenoids': 'Терпен-\nоиды',
    'carbohydrates': 'Углеводы',
    'proteins': 'Белки',
    'condensed_tanins': 'Конденсир-\nованные\nтанины',
    'lipids': 'Липиды'    
}

perminova_class_names_list = list(perminova_class_names_translate_dict.keys())
perminova_class_colors_list = list(perminova_class_color_dictionary.values())

def ChooseColor(row):
    """
    Придает формуле цвет на диаграмме Ван-Кревелина.
    """
    if "N" in row:
        if "S" in row:
            if row["S"] != 0:
                return "red" if row["N"] != 0 else "green"
            else:
                return "orange" if row["N"] != 0 else "blue"
        else:
            return "orange" if row["N"] != 0 else "blue"
    else:
        return "blue"

def vk(spec: pd.DataFrame,
               ax=None,
               sizes=(7, 30),
               plot_type='default',
               color_palette='viridis',
               error_type='rel_error',
               kde_fill=True,
               kde_levels=10,
               scatter_kde_enable=False):
    """
    Возвращает диаграмму Ван-Кревелена для подставленного спектра.
    
    Parameters
    ----------
    spec : pd.DataFrame
        DataFrame с данными спектра
    ax : matplotlib.axes.Axes, optional
        Оси для отрисовки, если None - создаются новые
    sizes : tuple, optional
        Размеры точек для scatter plot
    plot_type : str, optional
        Тип визуализации:
        - 'default': базовый scatter plot (по умолчанию)
        - 'heatmap': тепловая карта интенсивности
        - 'scatter': точечная карта интенсивности
        - 'error_scatter': точечная карта ошибки
        - 'error_heatmap': тепловая карта ошибки
        - 'mass_scatter': точечная карта масс
        - 'mass_heatmap': тепловая карта масс
        - 'oxygen_scatter': точечная карта количества кислорода
        - 'oxygen_heatmap': тепловая карта количества кислорода
        - 'density': плотность точек (2D KDE)
    color_palette : str, optional
        Цветовая палитра для тепловой карты
    error_type : str, optional
        Тип ошибки для визуализации: 'rel_error' или 'abs_error'
    kde_fill : bool, optional
        Заливать контуры KDE (True) или отображать только линии (False)
    kde_levels : int, optional
        Количество уровней для KDE
    
    Returns
    -------
    matplotlib.axes.Axes
    """
    spec = spec.copy(deep=True)
    
    # Добавляем соотношения O/C и H/C если их нет
    if "O/C" not in list(spec.columns):
        spec["O/C"] = spec["O"] / spec["C"]
        spec["H/C"] = spec["H"] / spec["C"]
    
    # Создаем оси если не переданы
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 2.2))
        ax.set_title(f"{spec.attrs['name']}, {spec.dropna().shape[0]} formulas")
    
    # Базовый вариант (оригинальный)
    if plot_type == 'default':
        spec["Color value"] = spec.apply(lambda x: ChooseColor(x), axis=1)
        sns.scatterplot(data=spec, x="O/C", y="H/C", 
                       hue="Color value", hue_order=["blue","orange","green","red"], 
                       size="intensity", alpha=0.7, legend=False, 
                       sizes=sizes, ax=ax)
    
    # Тепловая карта интенсивности
    elif plot_type == 'heatmap':
        valid_data = spec.dropna(subset=['O/C', 'H/C', 'intensity'])
        
        x_bins = np.linspace(0, 1, 50)
        y_bins = np.linspace(0, 2.2, 50)
        
        heatmap, xedges, yedges = np.histogram2d(
            valid_data['O/C'], 
            valid_data['H/C'], 
            bins=[x_bins, y_bins],
            weights=valid_data['intensity']
        )
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        im = ax.imshow(heatmap.T, 
                      extent=(0, 1, 0, 2.2), 
                      origin='lower', 
                      aspect='auto',
                      cmap=color_palette,
                      alpha=0.8)
        
        plt.colorbar(im, ax=ax, label='Normalized Intensity')
        ax.contour(heatmap.T, levels=5, extent=(0, 1, 0, 2.2), 
                  colors='white', alpha=0.5, linewidths=0.5)
    
    # Точечная карта интенсивности
    elif plot_type == 'scatter':
        valid_data = spec.dropna(subset=['O/C', 'H/C', 'intensity'])
        
        intensities = valid_data['intensity']
        if intensities.max() > intensities.min():
            normalized_sizes = (intensities - intensities.min()) / (intensities.max() - intensities.min())
            point_sizes = sizes[0] + normalized_sizes * (sizes[1] - sizes[0])
        else:
            point_sizes = [sizes[0]] * len(intensities)
        
        scatter = ax.scatter(valid_data['O/C'], valid_data['H/C'],
                           c=valid_data['intensity'],
                           s=point_sizes,
                           alpha=0.7,
                           cmap=color_palette)
        
        plt.colorbar(scatter, ax=ax, label='Intensity')
    
    # Точечная карта ошибки
    elif plot_type == 'error_scatter':
        valid_data = spec.dropna(subset=['O/C', 'H/C', error_type])
        
        errors = valid_data[error_type]
        if errors.max() > errors.min():
            normalized_sizes = (errors - errors.min()) / (errors.max() - errors.min())
            point_sizes = sizes[0] + normalized_sizes * (sizes[1] - sizes[0])
        else:
            point_sizes = [sizes[0]] * len(errors)
        
        scatter = ax.scatter(valid_data['O/C'], valid_data['H/C'],
                           c=valid_data[error_type],
                           s=point_sizes,
                           alpha=0.7,
                           cmap=color_palette)
        
        error_label = 'Relative Error (ppm)' if error_type == 'rel_error' else 'Absolute Error'
        plt.colorbar(scatter, ax=ax, label=error_label)
        ax.set_title(f"{spec.attrs['name']} - {error_label}")
    
    # Тепловая карта ошибки
    elif plot_type == 'error_heatmap':
        valid_data = spec.dropna(subset=['O/C', 'H/C', error_type])
        
        x_bins = np.linspace(0, 1, 50)
        y_bins = np.linspace(0, 2.2, 50)
        
        heatmap, xedges, yedges = np.histogram2d(
            valid_data['O/C'], 
            valid_data['H/C'], 
            bins=[x_bins, y_bins],
            weights=valid_data[error_type]
        )
        
        # Для ошибки лучше использовать инверсную палитру - меньшие ошибки лучше
        im = ax.imshow(heatmap.T, 
                      extent=(0, 1, 0, 2.2), 
                      origin='lower', 
                      aspect='auto',
                      cmap=color_palette + '_r',  # инверсная палитра
                      alpha=0.8)
        
        error_label = 'Relative Error (ppm)' if error_type == 'rel_error' else 'Absolute Error'
        plt.colorbar(im, ax=ax, label=error_label)
        ax.set_title(f"{spec.attrs['name']} - {error_label}")
    
    # Точечная карта масс
    elif plot_type == 'mass_scatter':
        valid_data = spec.dropna(subset=['O/C', 'H/C', 'mass'])
        
        masses = valid_data['mass']
        if masses.max() > masses.min():
            normalized_sizes = (masses - masses.min()) / (masses.max() - masses.min())
            point_sizes = sizes[0] + normalized_sizes * (sizes[1] - sizes[0])
        else:
            point_sizes = [sizes[0]] * len(masses)
        
        scatter = ax.scatter(valid_data['O/C'], valid_data['H/C'],
                           c=valid_data['mass'],
                           s=point_sizes,
                           alpha=0.7,
                           cmap=color_palette)
        
        plt.colorbar(scatter, ax=ax, label='Mass')
        ax.set_title(f"{spec.attrs['name']} - Mass Distribution")
    
    # Тепловая карта масс
    elif plot_type == 'mass_heatmap':
        valid_data = spec.dropna(subset=['O/C', 'H/C', 'mass'])
        
        x_bins = np.linspace(0, 1, 50)
        y_bins = np.linspace(0, 2.2, 50)
        
        heatmap, xedges, yedges = np.histogram2d(
            valid_data['O/C'], 
            valid_data['H/C'], 
            bins=[x_bins, y_bins],
            weights=valid_data['mass']
        )
        
        im = ax.imshow(heatmap.T, 
                      extent=(0, 1, 0, 2.2), 
                      origin='lower', 
                      aspect='auto',
                      cmap=color_palette,
                      alpha=0.8)
        
        plt.colorbar(im, ax=ax, label='Average Mass')
        ax.set_title(f"{spec.attrs['name']} - Mass Distribution")
    
    # Точечная карта количества кислорода
    elif plot_type == 'oxygen_scatter':
        valid_data = spec.dropna(subset=['O/C', 'H/C', 'O'])
        
        oxygen_counts = valid_data['O']
        if oxygen_counts.max() > oxygen_counts.min():
            normalized_sizes = (oxygen_counts - oxygen_counts.min()) / (oxygen_counts.max() - oxygen_counts.min())
            point_sizes = sizes[0] + normalized_sizes * (sizes[1] - sizes[0])
        else:
            point_sizes = [sizes[0]] * len(oxygen_counts)
        
        scatter = ax.scatter(valid_data['O/C'], valid_data['H/C'],
                           c=valid_data['O'],
                           s=point_sizes,
                           alpha=0.7,
                           cmap=color_palette)
        
        plt.colorbar(scatter, ax=ax, label='Oxygen Count')
        ax.set_title(f"{spec.attrs['name']} - Oxygen Distribution")
    
    # Тепловая карта количества кислорода
    elif plot_type == 'oxygen_heatmap':
        valid_data = spec.dropna(subset=['O/C', 'H/C', 'O'])
        
        x_bins = np.linspace(0, 1, 50)
        y_bins = np.linspace(0, 2.2, 50)
        
        heatmap, xedges, yedges = np.histogram2d(
            valid_data['O/C'], 
            valid_data['H/C'], 
            bins=[x_bins, y_bins],
            weights=valid_data['O']
        )
        
        im = ax.imshow(heatmap.T, 
                      extent=(0, 1, 0, 2.2), 
                      origin='lower', 
                      aspect='auto',
                      cmap=color_palette,
                      alpha=0.8)
        
        plt.colorbar(im, ax=ax, label='Average Oxygen Count')
        ax.set_title(f"{spec.attrs['name']} - Oxygen Distribution")
    
    # Плотность точек (2D KDE)
    elif plot_type == 'density':
        valid_data = spec.dropna(subset=['O/C', 'H/C'])
        
        # Создаем 2D KDE
        if len(valid_data) > 1:  # Нужно как минимум 2 точки для KDE
            try:
                # Используем seaborn для KDE
                if kde_fill:
                    sns.kdeplot(data=valid_data, x='O/C', y='H/C',
                               fill=True, alpha=0.6, levels=kde_levels,
                               cmap=color_palette, ax=ax)
                else:
                    sns.kdeplot(data=valid_data, x='O/C', y='H/C',
                               fill=False, levels=kde_levels,
                               cmap=color_palette, ax=ax, linewidths=1.5)
                
                # Добавляем scatter plot поверх KDE для лучшей видимости отдельных точек
                if scatter_kde_enable:
                    scatter = ax.scatter(valid_data['O/C'], valid_data['H/C'],
                                    c='black', alpha=0.3, s=10, marker='o')
                
            except Exception as e:
                print(f"KDE plotting failed: {e}. Using scatter plot instead.")
                # Если KDE не сработал, используем обычный scatter plot
                ax.scatter(valid_data['O/C'], valid_data['H/C'],
                          alpha=0.5, s=20, c='blue')
        else:
            # Если точек мало, просто рисуем scatter
            ax.scatter(valid_data['O/C'], valid_data['H/C'],
                      alpha=0.7, s=30, c='red')
            ax.text(0.5, 1.1, "Not enough points for KDE", 
                   ha='center', transform=ax.transAxes, color='red')
        
        ax.set_title(f"{spec.attrs['name']} - Point Density (KDE), {len(valid_data)} points")
    
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Available options: 'default', 'heatmap', 'scatter', 'error_scatter', 'error_heatmap', 'mass_scatter', 'mass_heatmap', 'oxygen_scatter', 'oxygen_heatmap', 'density'")
    
    # Общие настройки для всех типов графиков
    ax.set_xlabel('O/C')
    ax.set_ylabel('H/C')
    ax.grid(True, alpha=0.3)
    
    # Устанавливаем пределы если они не были установлены ранее
    if ax.get_xlim() == (0.0, 1.0):
        ax.set_xlim(0.0, 1.0)
    if ax.get_ylim() == (0.0, 1.0):
        ax.set_ylim(0.0, 2.2)
    
    return ax

def filter_dense_region_advanced(spec: pd.DataFrame,
                                method: str = 'kde',
                                percentile: float = 50,
                                cluster_method: str = 'dbscan',
                                visualize: bool = False,
                                ax=None,
                                **kwargs):
    """
    Расширенная функция для выделения плотных регионов.
    
    Parameters
    ----------
    spec : pd.DataFrame
        DataFrame с данными спектра
    method : str
        Метод фильтрации: 'kde' (по умолчанию), 'clustering', 'percentile'
    percentile : float
        Процентиль для отбора (для method='kde' и 'percentile')
    cluster_method : str
        Метод кластеризации: 'dbscan', 'kmeans'
    visualize : bool
        Визуализировать процесс
    ax : matplotlib.axes.Axes
        Оси для визуализации
    **kwargs : dict
        Дополнительные параметры для методов
    
    Returns
    -------
    pd.DataFrame
    """
    
    spec = spec.copy(deep=True)
    
    # Добавляем соотношения O/C и H/C если их нет
    if "O/C" not in list(spec.columns):
        spec["O/C"] = spec["O"] / spec["C"]
        spec["H/C"] = spec["H"] / spec["C"]
    
    valid_data = spec.dropna(subset=['O/C', 'H/C']).copy()
    
    if len(valid_data) == 0:
        return spec
    
    X = valid_data[['O/C', 'H/C']].values
    
    if method == 'kde':
        from scipy.stats import gaussian_kde
        
        bandwidth = kwargs.get('bandwidth', 0.05)
        kde = gaussian_kde(X.T, bw_method=bandwidth)
        densities = kde(X.T)
        valid_data['density'] = densities
        
        density_threshold = np.percentile(densities, percentile)
        filtered_data = valid_data[valid_data['density'] >= density_threshold]
        
    elif method == 'percentile':
        # Простой метод по процентилю координат
        o_c_threshold = np.percentile(valid_data['O/C'], percentile)
        h_c_threshold = np.percentile(valid_data['H/C'], percentile)
        
        filtered_data = valid_data[
            (valid_data['O/C'] >= o_c_threshold) & 
            (valid_data['H/C'] >= h_c_threshold)
        ]
        
    elif method == 'clustering':
        from sklearn.cluster import DBSCAN, KMeans
        
        if cluster_method == 'dbscan':
            eps = kwargs.get('eps', 0.1)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(X)
            
        elif cluster_method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 3)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(X)
        
        valid_data['cluster'] = labels # type: ignore
        
        # Находим самый большой кластер (исключая шум для DBSCAN)
        if cluster_method == 'dbscan':
            cluster_sizes = valid_data[valid_data['cluster'] != -1]['cluster'].value_counts()
        else:
            cluster_sizes = valid_data['cluster'].value_counts()
        
        if len(cluster_sizes) > 0:
            largest_cluster = cluster_sizes.index[0]
            filtered_data = valid_data[valid_data['cluster'] == largest_cluster]
        else:
            filtered_data = valid_data
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Визуализация
    if visualize:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(valid_data)))
        
        if method == 'kde':
            scatter_all = ax.scatter(valid_data['O/C'], valid_data['H/C'], 
                                   c=valid_data['density'], cmap='viridis', 
                                   s=30, alpha=0.5, label='All points')
            scatter_filtered = ax.scatter(filtered_data['O/C'], filtered_data['H/C'], 
                                        c='red', s=50, alpha=0.8, label='Dense region')
            plt.colorbar(scatter_all, ax=ax, label='Density')
            
        elif method in ['clustering', 'percentile']:
            ax.scatter(valid_data['O/C'], valid_data['H/C'], 
                      c='gray', alpha=0.3, s=20, label='All points')
            ax.scatter(filtered_data['O/C'], filtered_data['H/C'], 
                      c='red', alpha=0.8, s=40, label='Selected region')
        
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 2.2)
        ax.set_xlabel('O/C')
        ax.set_ylabel('H/C')
        ax.set_title(f"Filtered: {len(filtered_data)}/{len(valid_data)} points")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Очищаем временные колонки
    result_df = filtered_data.drop(columns=['density', 'cluster'], errors='ignore')
    
    if hasattr(spec, 'attrs'):
        result_df.attrs = spec.attrs.copy()
    
    return result_df

def plot_spectra_kde_comparison(spectra_list: list,
                               classes: list|None = None,
                               bandwidth: float = 0.05,
                               n_levels: int = 10,
                               figsize: tuple = (12, 8),
                               color_palette: str = 'viridis',
                               show_contours: bool = True,
                               show_fill: bool = True,
                               alpha: float = 0.7,
                               subplot_layout: tuple|None = None):
    """
    Рисует 2D KDE графики для сравнения плотных участков нескольких спектров.
    
    Parameters
    ----------
    spectra_list : list
        Список DataFrame с масс-спектрами
    classes : list, optional
        Список классов для группировки, если None - используется spec.attrs['Class']
    bandwidth : float, optional
        Параметр сглаживания для KDE
    n_levels : int, optional
        Количество уровней контуров
    figsize : tuple, optional
        Размер фигуры
    color_palette : str, optional
        Цветовая палитра
    show_contours : bool, optional
        Показывать контурные линии
    show_fill : bool, optional
        Заливать области
    alpha : float, optional
        Прозрачность заливки
    subplot_layout : tuple, optional
        Расположение subplots (rows, cols), если None - определяется автоматически
    
    Returns
    -------
    matplotlib.figure.Figure, numpy.ndarray of matplotlib.axes.Axes
    """
    
    if not spectra_list:
        raise ValueError("Список спектров не может быть пустым")
    
    # Определяем классы для каждого спектра
    if classes is None:
        classes = []
        for spec in spectra_list:
            if hasattr(spec, 'attrs') and 'Class' in spec.attrs:
                classes.append(spec.attrs['Class'])
            else:
                classes.append(f"Spectrum_{len(classes)+1}")
    else:
        if len(classes) != len(spectra_list):
            raise ValueError("Длина списка классов должна совпадать с длиной списка спектров")
    
    # Группируем спектры по классам
    from collections import defaultdict
    class_spectra = defaultdict(list)
    class_names = []
    
    for spec, class_name in zip(spectra_list, classes):
        class_spectra[class_name].append(spec)
        if class_name not in class_names:
            class_names.append(class_name)
    
    # Определяем layout subplots
    if subplot_layout is None:
        n_classes = len(class_names)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
    else:
        n_rows, n_cols = subplot_layout
    
    # Создаем фигуру и оси
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Цветовая схема
    colors = plt.cm.get_cmap(color_palette, len(class_names))
    
    # Для каждого класса строим KDE
    for idx, (class_name, class_specs) in enumerate(class_spectra.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Объединяем все спектры этого класса
        all_class_data = []
        for spec in class_specs:
            spec_copy = spec.copy(deep=True)
            
            # Добавляем соотношения O/C и H/C если их нет
            if "O/C" not in list(spec_copy.columns):
                spec_copy["O/C"] = spec_copy["O"] / spec_copy["C"]
                spec_copy["H/C"] = spec_copy["H"] / spec_copy["C"]
            
            valid_data = spec_copy.dropna(subset=['O/C', 'H/C'])
            if len(valid_data) > 0:
                all_class_data.append(valid_data[['O/C', 'H/C']])
        
        if not all_class_data:
            ax.text(0.5, 0.5, f"No valid data\nfor {class_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2.2)
            continue
        
        # Объединяем все данные класса
        combined_data = pd.concat(all_class_data, ignore_index=True)
        X = combined_data[['O/C', 'H/C']].values
        
        if len(X) < 2:
            ax.text(0.5, 0.5, f"Not enough data\nfor KDE\n{len(X)} points", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2.2)
            continue
        
        # Создаем KDE
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(X.T, bw_method=bandwidth)
            
            # Создаем сетку для KDE
            x_min, x_max = 0, 1.0
            y_min, y_max = 0, 2.2
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            grid_coords = np.vstack([xx.ravel(), yy.ravel()])
            
            # Вычисляем плотность на сетке
            grid_density = kde(grid_coords).reshape(xx.shape)
            
            # Визуализируем KDE
            if show_fill:
                contourf = ax.contourf(xx, yy, grid_density, 
                                     levels=n_levels, 
                                     alpha=alpha, 
                                     cmap=color_palette)
            
            if show_contours:
                contour = ax.contour(xx, yy, grid_density, 
                                   levels=n_levels, 
                                   colors='black', 
                                   alpha=0.5, 
                                   linewidths=0.8)
                # ax.clabel(contour, inline=True, fontsize=8)
            
            # Добавляем цветовую шкалу для каждого subplot
            if show_fill:
                plt.colorbar(contourf, ax=ax, label='Density') # type: ignore
            
        except Exception as e:
            print(f"Ошибка при построении KDE для класса {class_name}: {e}")
            ax.text(0.5, 0.5, f"KDE error\n{class_name}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Настройки графика
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 2.2)
        ax.set_xlabel('O/C')
        ax.set_ylabel('H/C')
        ax.set_title(f'{class_name}\n({len(combined_data)} points, {len(class_specs)} spectra)')
        ax.grid(True, alpha=0.3)
    
    # Скрываем неиспользованные оси
    for idx in range(len(class_spectra), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig, axes

def plot_spectra_kde_overlay(spectra_list: list,
                           classes: list|None = None,
                           bandwidth: float = 0.05,
                           n_levels: int = 8,
                           figsize: tuple = (10, 8),
                           color_palette: str = 'Set2',
                           alpha: float = 0.6):
    """
    Рисует несколько KDE на одном графике для сравнения.
    
    Parameters
    ----------
    spectra_list : list
        Список DataFrame с масс-спектрами
    classes : list, optional
        Список классов для группировки
    bandwidth : float, optional
        Параметр сглаживания для KDE
    n_levels : int, optional
        Количество уровней контуров
    figsize : tuple, optional
        Размер фигуры
    color_palette : str, optional
        Цветовая палитра
    alpha : float, optional
        Прозрачность заливки
    
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    
    if not spectra_list:
        raise ValueError("Список спектров не может быть пустым")
    
    # Определяем классы для каждого спектра
    if classes is None:
        classes = []
        for spec in spectra_list:
            if hasattr(spec, 'attrs') and 'Class' in spec.attrs:
                classes.append(spec.attrs['Class'])
            else:
                classes.append(f"Spectrum_{len(classes)+1}")
    
    # Убираем дубликаты классов
    unique_classes = []
    unique_indices = []
    for i, class_name in enumerate(classes):
        if class_name not in unique_classes:
            unique_classes.append(class_name)
            unique_indices.append(i)
    
    # Создаем фигуру
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Цветовая схема
    colors = plt.cm.get_cmap(color_palette, len(unique_classes))
    
    # Для каждого уникального класса строим KDE
    legend_elements = []
    
    for idx, class_idx in enumerate(unique_indices):
        class_name = classes[class_idx]
        
        # Собираем все спектры этого класса
        class_spectra = [spectra_list[i] for i, cls in enumerate(classes) if cls == class_name]
        
        # Объединяем данные класса
        all_class_data = []
        for spec in class_spectra:
            spec_copy = spec.copy(deep=True)
            
            if "O/C" not in list(spec_copy.columns):
                spec_copy["O/C"] = spec_copy["O"] / spec_copy["C"]
                spec_copy["H/C"] = spec_copy["H"] / spec_copy["C"]
            
            valid_data = spec_copy.dropna(subset=['O/C', 'H/C'])
            if len(valid_data) > 0:
                all_class_data.append(valid_data[['O/C', 'H/C']])
        
        if not all_class_data:
            continue
        
        combined_data = pd.concat(all_class_data, ignore_index=True)
        X = combined_data[['O/C', 'H/C']].values
        
        if len(X) < 2:
            continue
        
        # Создаем KDE
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(X.T, bw_method=bandwidth)
            
            # Создаем сетку для KDE
            x_min, x_max = 0, 1.0
            y_min, y_max = 0, 2.2
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            grid_coords = np.vstack([xx.ravel(), yy.ravel()])
            
            # Вычисляем плотность на сетке
            grid_density = kde(grid_coords).reshape(xx.shape)
            
            # Нормализуем плотность для лучшего сравнения
            if grid_density.max() > 0:
                grid_density = grid_density / grid_density.max()
            
            # Визуализируем контуры
            color = colors(idx)
            contourf = ax.contourf(xx, yy, grid_density, 
                                 levels=n_levels, 
                                 alpha=alpha, 
                                 colors=[color])
            
            contour = ax.contour(xx, yy, grid_density, 
                               levels=n_levels, 
                               colors=[color], 
                               alpha=0.8, 
                               linewidths=1.5)
            
            # Добавляем в легенду
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=color, alpha=alpha, label=class_name))
            
        except Exception as e:
            print(f"Ошибка при построении KDE для класса {class_name}: {e}")
    
    # Настройки графика
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel('O/C')
    ax.set_ylabel('H/C')
    ax.set_title('KDE Comparison of Different Classes')
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax
       
def spectrum(spec: 'pd.DataFrame',
             xlim = (None, None),
             ylim = (None, None),
             color: str = 'black',
             ax = None,
             title= None,
             normalize: bool = False,
             peak_width: float = 1.0,
             remove_outliers: bool = True,
             outlier_percentile: float = 99.7,
             **kwargs):
    """
    Draw mass spectrum with improved visualization
    
    Parameters
    ----------
    spec: Spectrum object
        spectrum for plot
    xlim: Tuple (float, float)
        restrict for mass
    ylim: Tuple (float, float)
        restrict for intensity
    color: str
        color of draw. Default black.
    ax: matplotlib axes object
        send here ax to plot in your own condition
    title: str
        Title of plot. Default None - Take name from metadata and number of peaks.
    normalize: bool
        Normalize intensities to 100%
    peak_width: float
        Width of spectral peaks
    **kwargs: dict
        additional parameters to matplotlib plot method
    """
    
    df = spec.copy() # Работаем с копией
    #df,_ = gmm_noise_filter(df)
    # Filter data based on xlim
    mass = df['mass'].to_numpy()
    intensity = df['intensity'].to_numpy()
    
    # 1. Применяем xlim фильтр ДО всех расчетов
    x_min = xlim[0] if xlim[0] is not None else mass.min()
    x_max = xlim[1] if xlim[1] is not None else mass.max()
    xlim = (x_min, x_max)
    
    mask = (mass >= xlim[0]) & (mass <= xlim[1])
    mass = mass[mask]
    intensity = intensity[mask]

    if len(intensity) == 0:
        print("Warning: No peaks in the specified range.")
        return ax

    # 2. Обработка выбросов для визуализации
    # Мы не удаляем данные из отрисовки (vlines), но используем "чистую" интенсивность для лимитов
    if remove_outliers and len(intensity) > 10:
        # Рассчитываем порог на основе перцентиля (например, 99.5%)
        viz_max = np.percentile(intensity, outlier_percentile)
    else:
        viz_max = intensity.max()

    # 3. Нормализация (если нужна)
    # Важно: нормализуем либо по честному максимуму, либо по "чистому" — зависит от задачи.
    # Обычно в МС нормализуют по Base Peak (100%), даже если это выброс.
    actual_max = intensity.max()
    if normalize and actual_max > 0:
        intensity = (intensity / actual_max) * 100
        viz_max = (viz_max / actual_max) * 100
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Отрисовка всех пиков (выбросы просто уйдут за верхнюю границу графика)
    ax.vlines(mass, ymin=0, ymax=intensity, 
              color=color, linewidth=peak_width, **kwargs)
    
    # Set limits
    ax.set_xlim(xlim)
    
    # 4. Умная установка Y-limit
    if ylim[0] is None or ylim[1] is None:
        y_max_plot = viz_max if viz_max > 0 else 1
        y_padding = y_max_plot * 0.15
        # Нижняя граница чуть ниже нуля для чистоты, верхняя по viz_max
        ax.set_ylim((-y_max_plot * 0.02, y_max_plot + y_padding)) # type: ignore
    else:
        ax.set_ylim(ylim)
    
    # Labels & Styling
    ax.set_xlabel('m/z, Da', fontsize=12)
    ax.set_ylabel('Relative Intensity (%)' if normalize else 'Intensity', fontsize=12)
    
    if title is None:
        title = f'Spectrum: {len(mass)} peaks'
    ax.set_title(title, fontsize=14, pad=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    return ax

def MolClassesSpectrum(specList,draw=True,ax=None):

    density_class_list=[]
    sample_list = []
    for i in range(len(specList)):
        spec = specList[i]
        data_class = spec.get_mol_class(how="perminova")
        density_class_list.append(data_class["density"])
        sample_list.append(spec.metadata["name"])
        mol_class_list=data_class["class"].to_list()
    mol_class_data = pd.DataFrame(np.array(density_class_list), index=sample_list, columns=mol_class_list) # type: ignore
    df_reversed = mol_class_data.sort_index(ascending=False).copy()
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8, 6))
    df_reversed.plot(kind='barh', stacked=True, ax=ax)
    # Настройки графика
    ax.set_title('Распределение классов по образцам')
    ax.set_xlabel('Доля')
    ax.set_ylabel('Образец')
    #plt.xticks(rotation=30)  # Чтобы подписи по оси X не наклонялись
    ax.legend(title='Классы', bbox_to_anchor=(1.05, 1), loc='upper left')  # Легенда справа
    # Показать график
    plt.tight_layout()

    return df_reversed

def CalcMetricSpectrum(specList,func="weight",draw=True):

    list_data = []
    for i in range(len(specList)):
        spec = specList[i]
        data = spec.get_mol_metrics(func=func)
        data.rename(columns={"value": f"{spec.metadata['name']}"},inplace=True)
        if i == 0:
            data_prev = data
            continue
        else:
            data_prev=data.merge(data_prev, on="metric") # type: ignore

    return data_prev # type: ignore

def FormulaSpecData(specList, draw=True):
    all_formula_list = []
    sample_list = []
    CHON_formula_list = []
    CHOS_formula_list = []
    CHONS_formula_list = []
    dict_list = []
    data = pd.DataFrame()
    for i in range(len(specList)):
        spec = specList[i]

        sample_list.append(spec.metadata["name"])

        all_formula_list.append(spec.table.dropna().shape[0])

        dict = spec.table[["N","S"]].value_counts().to_dict()

        result_CHON = [key for key in dict.keys() if key[0] > 0 and key[1] < 1]
        result_CHOS = [key for key in dict.keys() if key[0] < 1 and key[1] > 0]
        result_CHONS = [key for key in dict.keys() if key[0] > 0 and key[1] > 0]
        sum_CHON = []
        sum_CHOS = []
        sum_CHONS = []
        for res in result_CHON:
            sum_CHON.append(dict[res])
        for res in result_CHOS:
            sum_CHOS.append(dict[res])
        for res in result_CHONS:
            sum_CHONS.append(dict[res])
        CHON_formula_list.append(sum(sum_CHON))
        CHOS_formula_list.append(sum(sum_CHOS))
        CHONS_formula_list.append(sum(sum_CHONS))
        dict_list.append(dict)
    data["All formulas"] = all_formula_list
    data["CHON, formulas"] = CHON_formula_list
    data["CHOS, formulas"] = CHOS_formula_list
    data["CHONS, formulas"] = CHONS_formula_list
    data["Dict, {N, S}: count"] = dict_list
    data["Sample name"] = sample_list
    data.set_index("Sample name",inplace=True)

    if draw:
        data.plot(kind='bar', stacked=False, figsize=(10, 8))
        plt.xticks(rotation=30)
        plt.tight_layout()
        
    return data

def elemental_composition(data:pd.DataFrame,
                          figsize=(12, 8)):
    
    data = data.copy()
    
    def classify_formula(row):
        """Классифицирует формулу по наличию элементов"""
        try:
            has_n = row.get('N', 0) > 0
            has_s = row.get('S', 0) > 0
            has_o = row.get('O', 0) > 0
            
            if has_n and has_s:
                return 'CHONS'
            elif has_n and not has_s:
                return 'CHON' 
            elif not has_n and has_s:
                return 'CHOS'
            else:
                return 'CHO'
        except:
            return 'Unknown'
    
    data['formula_type'] = data.apply(classify_formula, axis=1)
    
    counts = data['formula_type'].value_counts()
    
    result_row = {}
    
    for formula_type in ['CHO', 'CHON', 'CHOS', 'CHONS']:
        result_row[formula_type] = counts.get(formula_type, 0)
        
    result_row['Total'] = len(data)
    ax = sns.barplot(x=result_row.keys(), y= result_row.values(),hue=result_row.keys(),
                palette=list(elenemtal_colors_dict.values()))    
    for container in ax.containers[:7]:
        ax.bar_label(container, fontsize=10) # type: ignore

def plot_mass_intensity_relationship(spectrum_df, mass_col='mass', intensity_col='intensity'):
    """
    Визуализирует взаимосвязь между массой и интенсивностью.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    mass = spectrum_df[mass_col].values
    intensity = spectrum_df[intensity_col].values
    
    # 1. Scatter plot
    scatter = ax1.scatter(mass, intensity, alpha=0.6, s=30, c=intensity, cmap='viridis')
    ax1.set_xlabel('Масса')
    ax1.set_ylabel('Интенсивность')
    ax1.set_title('Распределение масса-интенсивность')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Интенсивность')
    
    # 2. Гистограмма масс
    ax2.hist(mass, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(np.mean(mass), color='red', linestyle='--', linewidth=2, label=f'Среднее: {np.mean(mass):.1f}')
    ax2.axvline(np.average(mass, weights=intensity), color='green', linestyle='--', linewidth=2, 
                label=f'Ср.взвеш.: {np.average(mass, weights=intensity):.1f}')
    ax2.set_xlabel('Масса')
    ax2.set_ylabel('Количество пиков')
    ax2.set_title('Распределение масс')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Средняя интенсивность по массовым бинам
    mass_bins = np.linspace(mass.min(), mass.max(), 20)
    bin_centers = (mass_bins[:-1] + mass_bins[1:]) / 2
    mean_intensities = []
    
    for i in range(len(mass_bins)-1):
        mask = (mass >= mass_bins[i]) & (mass < mass_bins[i+1])
        if np.sum(mask) > 0:
            mean_intensities.append(np.mean(intensity[mask]))
        else:
            mean_intensities.append(0)
    
    ax3.plot(bin_centers, mean_intensities, 'o-', linewidth=2, markersize=4)
    ax3.set_xlabel('Масса')
    ax3.set_ylabel('Средняя интенсивность')
    ax3.set_title('Зависимость интенсивности от массы')
    ax3.grid(True, alpha=0.3)
    
    # 4. Накопительная интенсивность
    sorted_indices = np.argsort(mass)
    sorted_mass = mass[sorted_indices]
    sorted_intensity = intensity[sorted_indices]
    cumulative_intensity = np.cumsum(sorted_intensity) / np.sum(intensity) * 100
    
    ax4.plot(sorted_mass, cumulative_intensity, linewidth=2)
    ax4.set_xlabel('Масса')
    ax4.set_ylabel('Накопительная интенсивность (%)')
    ax4.set_title('Накопительное распределение интенсивности')
    ax4.grid(True, alpha=0.3)
    
    # Добавляем линии для 25%, 50%, 75%
    for percentile in [25, 50, 75]:
        idx = np.searchsorted(cumulative_intensity, percentile)
        if idx < len(sorted_mass):
            ax4.axvline(sorted_mass[idx], color='red', linestyle='--', alpha=0.7)
            ax4.text(sorted_mass[idx], percentile, f' {percentile}%', va='center')
    
    plt.tight_layout()
    return fig

def plot_mol_class_distribution(masslist:pd.DataFrame,mod='bar'):
    
    data = masslist.copy()    
    value_counts = data.dropna().value_counts("class")
    
    if mod == 'bar':
        
        ax = sns.barplot(x=list(perminova_class_names_translate_dict.values()),y=value_counts,
                         hue=list(perminova_class_names_translate_dict.values()),
                         palette=perminova_class_colors_list)
        for container in ax.containers[:7]:
            ax.bar_label(container, fontsize=10) # type: ignore
            
    elif mod == 'pie':
        
        print(value_counts)
        plt.pie(value_counts.to_numpy(),
                labels=perminova_class_names_list, autopct='%1.1f%%',
                colors=perminova_class_colors_list,textprops={'color':'black','fontsize':12},
                explode=[0, 0, 0.14, 0, 0, 0,0])
        
    elif mod == 'vk':
        vk(data, plot_type='default')
    
    elif mod == 'mass_scatter':
        vk(data, plot_type='mass_scatter')
        
    elif mod == 'oxygen_scatter':
        vk(data, plot_type='oxygen_scatter')
        
    elif mod == 'density':
        vk(data, plot_type='density')
        
    elif mod == 'mass_distibution':
        plot_mass_intensity_relationship(data)            
