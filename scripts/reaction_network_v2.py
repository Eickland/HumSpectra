import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pth
import seaborn as sns
import HumSpectra.mass_descriptors as md
import HumSpectra.kendric_mass_defect as kmd
from collections import defaultdict
import numpy as np
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from itertools import combinations
from contextlib import contextmanager
import networkx as nx
from collections import Counter

import HumSpectra.utilits as ut

def plot_homologous_series(data:pd.DataFrame, fragment='CH2', min_series_length=3, 
                          highlight_series=True, save_path=None,ylim=[-0.5,0.5]):
    """
    Визуализация гомологических рядов на графике KMD vs Номинальная масса Кендрика
    
    Параметры
    ----------
    data : DataFrame
        Данные с колонками 'calc_mass', 'KMD', 'Nominal_Ke'
    fragment : str
        Фрагмент для расчета KMD ('CH2', 'O', 'CO2' и т.д.)
    min_series_length : int
        Минимальная длина гомологического ряда для отображения
    highlight_series : bool
        Выделять ли найденные ряды цветами
    save_path : str
        Путь для сохранения графика (опционально)
    """
    if type(data) != pd.DataFrame:
        raise ValueError('')
    
    # Рассчитываем KMD если еще не рассчитан
    if 'KMD' not in data.columns or 'Nominal_Ke' not in data.columns:
        data = kmd.kendrick(data, fragment_formula=fragment) # type: ignore
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ===== Левый график: Все точки =====
    scatter1 = ax1.scatter(data['Nominal_Ke'], data['KMD'], 
                          c=data.get('intensity', np.ones(len(data))),
                          s=20, cmap='viridis', alpha=0.6, edgecolors='none')
    ax1.set_ylim(ylim)
    ax1.set_xlabel('Nominal Kendrick Mass', fontsize=12)
    ax1.set_ylabel('Kendrick Mass Defect (KMD)', fontsize=12)
    ax1.set_title(f'All peaks (fragment: {fragment})', fontsize=14)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Intensity' if 'intensity' in data.columns else 'Density')
    
    # ===== Правый график: С выделенными рядами =====
    # Находим гомологические ряды
    groups = group_by_kmd(data)
    series_list = filter_homologous_series(groups, min_length=min_series_length)
    # Рисуем фоновые точки (серые)
    ax2.scatter(data['Nominal_Ke'], data['KMD'], 
               c='lightgray', s=15, alpha=0.3, edgecolors='none')
    
    # Рисуем каждый ряд своим цветом
    colors = plt.colormaps['magma'](np.linspace(0, 1, len(series_list)))
    
    for idx, series in enumerate(series_list):
        series_df = pd.DataFrame(series[0]) if isinstance(series, list) else series
        color = colors[idx % len(colors)]
        
        # Точки ряда
        ax2.scatter(series_df['Nominal_Ke'], series_df['KMD'], 
                   c=[color], s=40, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Соединяем точки линией
        sorted_series = series_df.sort_values('Nominal_Ke')
        ax2.plot(sorted_series['Nominal_Ke'], sorted_series['KMD'], 
                '-', color=color, alpha=0.6, linewidth=1.5)
        
        # Добавляем подписи формул (опционально, для длинных рядов)
        if len(series_df) >= 5 and 'formula' in series_df.columns:
            mid_point = len(sorted_series) // 2
            ax2.annotate(sorted_series.iloc[mid_point]['formula'], 
                        xy=(sorted_series.iloc[mid_point]['Nominal_Ke'], 
                            sorted_series.iloc[mid_point]['KMD']),
                        fontsize=7, alpha=0.7, xytext=(5, 5), 
                        textcoords='offset points')
    
    ax2.set_xlabel('Nominal Kendrick Mass', fontsize=12)
    ax2.set_ylabel('Kendrick Mass Defect (KMD)', fontsize=12)
    ax2.set_title(f'Homologous series (n ≥ {min_series_length}, fragment: {fragment})', fontsize=14)
    
    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightgray', alpha=0.3, label='Non-series peaks'),
                       Patch(facecolor='red', alpha=0.8, label=f'Homologous series ({len(series_list)} found)')]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return series_list

def group_by_kmd(data):
    
    groups = defaultdict(list)

    data['kmd_group'] = data['KMD'].apply(lambda x: assign_kmd_group(x))

    for kmd_group in data['kmd_group'].unique():
        groups[kmd_group].append(data[data['kmd_group'] == kmd_group])
    
    return groups    

def assign_kmd_group(kmd, epsilon=1e-6):
    return round(kmd / epsilon) * epsilon

def filter_homologous_series(series_groups,min_length=2):
    
    filter_series = []
    
    for kmd_group in series_groups:
        
        series = series_groups[kmd_group]
        
        if len(series[0]) > min_length:
            
            filter_series.append(series)
    
    return filter_series

class DifferenceNetwork:
    """
    Построение графа разностей масс (Difference Network) для FT-ICR MS спектров
    
    Основан на подходе, описанном в статьях:
    - Tolić et al. (2017) - Kendrick Mass Defect Spectrum Analysis
    - Longnecker & Kujawinski (2016) - Network analysis for microbial communities
    - Pikovskoi et al. (2025) - для лигносульфонатов (SO3 группы)
    """
    
    # Точные массы распространенных трансформаций (в Да)
    # Положительные значения - добавление группы, отрицательные - удаление
    TRANSFORMATIONS = {
        # Базовые реакции
        'CH2': 14.015650,           # Метилен (гомологический ряд)
        '-CH2': -14.015650,         # Деметилирование
        'H2': 2.015650,             # Гидрирование
        '-H2': -2.015650,           # Дегидрирование
        'O': 15.994915,             # Гидроксилирование / окисление
        '-O': -15.994915,           # Восстановление
        'H2O': 18.010565,           # Гидратация
        '-H2O': -18.010565,         # Дегидратация
        'CO2': 43.989829,           # Карбоксилирование
        '-CO2': -43.989829,         # Декарбоксилирование
        'CO': 27.994915,            # Добавление CO
        '-CO': -27.994915,          # Удаление CO
        
        # Реакции с серой (для лигносульфонатов и РОВ)
        'S': 31.972071,             # Сульфирование
        '-S': -31.972071,           # Десульфирование
        'SO2': 63.961902,           # Добавление SO2
        'SO3': 79.956816,           # Добавление SO3 (сульфогруппы)
        '-SO3': -79.956816,         # Удаление SO3
        'H2SO3': 81.972466,         # Сернистая кислота
        'H2SO4': 97.967381,         # Серная кислота
        
        # Реакции с азотом
        'N': 14.003074,             # Добавление азота
        'NH3': 17.026549,           # Аминирование
        'NO2': 45.992904,           # Нитрирование
        'NO3': 61.987819,           # Нитрация
        
        # Комбинированные реакции (часто встречаются в РОВ)
        'CH2O': 30.010565,          # Гидроксиметилирование
        'C2H2O': 42.010565,         # Ацетилирование
        'C2H4O': 44.026215,         # Этоксилирование
        'CH3': 15.023475,           # Метилирование
        'OCH3': 31.018390,          # Метоксилирование
        'COOH': 44.997655,          # Карбоксильная группа
        'OH': 17.002740,            # Гидроксильная группа
    }
    
    def __init__(self, 
                 tolerance_ppm: float = 3.0,
                 transformations: Dict[str, float]|None = None,
                 min_intensity: float = 0.0,
                 directed: bool = True):
        """
        Параметры
        ----------
        tolerance_ppm : float
            Допуск для разницы масс в ppm
        transformations : Dict[str, float]
            Словарь реакций {название: точная масса}
        min_intensity : float
            Минимальная интенсивность для включения пика
        directed : bool
            Направленный ли граф (True) или ненаправленный (False)
        """
        self.tolerance_ppm = tolerance_ppm
        self.transformations = transformations or self.TRANSFORMATIONS
        self.min_intensity = min_intensity
        self.directed = directed
        self.G = nx.DiGraph() if directed else nx.Graph()
        
    def build(self, 
              data: pd.DataFrame,
              mass_col: str = 'calc_mass',
              intensity_col: str = 'intensity',
              formula_col: str = 'formula',
              additional_cols: List[str]|None = None) -> nx.Graph:
        """
        Построение графа разностей масс
        
        Параметры
        ----------
        data : pd.DataFrame
            Данные спектра
        mass_col : str
            Название колонки с точными массами
        intensity_col : str
            Название колонки с интенсивностями
        formula_col : str
            Название колонки с формулами (опционально)
        additional_cols : List[str]
            Дополнительные колонки для сохранения в узлах
            
        Возвращает
        ----------
        nx.Graph : Построенный граф
        """
        
        # Фильтрация по интенсивности
        if self.min_intensity > 0 and intensity_col in data.columns:
            data = data[data[intensity_col] >= self.min_intensity].copy()
        
        # Сортируем по массе для эффективного поиска
        data = data.sort_values(mass_col).reset_index(drop=True)
        masses = data[mass_col].to_numpy()
        indices = data.index
        
        # Добавляем узлы
        for idx in indices:
            node_attrs = {
                'mass': masses[idx],
                'intensity': data.loc[idx, intensity_col] if intensity_col in data.columns else 1.0
            }
            
            if formula_col in data.columns and pd.notna(data.loc[idx, formula_col]):
                node_attrs['formula'] = data.loc[idx, formula_col]
            
            if additional_cols:
                for col in additional_cols:
                    if col in data.columns:
                        node_attrs[col] = data.loc[idx, col]
            
            self.G.add_node(idx, **node_attrs)
        
        # Поиск ребер (реакций)
        self._find_edges(masses, indices, data)
        
        return self.G
    
    def _find_edges(self, masses: np.ndarray, indices: pd.Index, data: pd.DataFrame):
        """
        Поиск пар пиков, разница масс которых соответствует известным реакциям
        """
        n = len(masses)
        edges_found = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                mass_diff = abs(masses[j] - masses[i])
                
                # Проверяем, соответствует ли разница какой-либо реакции
                for reaction_name, reaction_mass in self.transformations.items():
                    if self._matches_mass_diff(mass_diff, abs(reaction_mass)):
                        # Определяем направление
                        if self.directed:
                            if masses[j] - masses[i] > 0 and reaction_mass > 0:
                                # i -> j (добавление группы)
                                self._add_edge(indices[i], indices[j], reaction_name, reaction_mass)
                            elif masses[i] - masses[j] > 0 and reaction_mass < 0:
                                # j -> i (добавление группы)
                                self._add_edge(indices[j], indices[i], reaction_name, abs(reaction_mass))
                            else:
                                # Ненаправленное ребро (если разрешено)
                                if not self.directed:
                                    self._add_undirected_edge(indices[i], indices[j], reaction_name, abs(reaction_mass))
                        else:
                            self._add_undirected_edge(indices[i], indices[j], reaction_name, abs(reaction_mass))
                        
                        edges_found += 1
                        break  # Нашли одну реакцию для этой пары
        
        print(f"Найдено ребер: {edges_found}")
        return edges_found
    
    def _matches_mass_diff(self, mass_diff: float, target_mass: float) -> bool:
        """Проверяет, соответствует ли разница масс целевой в пределах допуска"""
        if target_mass == 0:
            return False
        error_ppm = abs(mass_diff - target_mass) / target_mass * 1e6
        return error_ppm <= self.tolerance_ppm
    
    def _add_edge(self, u: int, v: int, reaction: str, mass_shift: float):
        """Добавляет направленное ребро"""
        if self.G.has_edge(u, v):
            # Добавляем дополнительную реакцию к существующему ребру
            existing_reactions = self.G.edges[u, v].get('reactions', [])
            existing_reactions.append(reaction)
            self.G.edges[u, v]['reactions'] = existing_reactions
            self.G.edges[u, v]['mass_shifts'].append(mass_shift)
        else:
            self.G.add_edge(u, v, 
                           reactions=[reaction], 
                           mass_shifts=[mass_shift],
                           reaction=reaction,  # для обратной совместимости
                           mass_shift=mass_shift)
    
    def _add_undirected_edge(self, u: int, v: int, reaction: str, mass_shift: float):
        """Добавляет ненаправленное ребро"""
        self._add_edge(u, v, reaction, mass_shift)
    
    def add_custom_transformation(self, name: str, exact_mass: float):
        """Добавляет пользовательскую трансформацию"""
        self.transformations[name] = exact_mass
    
    def add_transformations_from_series(self, data: pd.DataFrame, 
                                        kmd_col: str = 'KMD',
                                        nom_ke_col: str = 'Nominal_Ke',
                                        max_points_per_series: int = 100):
        """
        Добавляет трансформации, найденные из гомологических рядов
        (метод из статьи Pikovskoi et al. 2025)
        
        Параметры
        ----------
        data : pd.DataFrame
            Данные с рассчитанным KMD
        kmd_col : str
            Колонка с KMD
        nom_ke_col : str
            Колонка с номинальной массой Кендрика
        max_points_per_series : int
            Максимальное количество точек для анализа
        """
        from collections import defaultdict
        
        # Группируем по KMD (как для поиска гомологических рядов)
        epsilon = 1e-6
        data['kmd_group'] = (data[kmd_col] / epsilon).round() * epsilon
        
        groups = defaultdict(list)
        for kmd_group in data['kmd_group'].unique():
            group_data = data[data['kmd_group'] == kmd_group]
            if len(group_data) > 2:
                groups[kmd_group] = group_data.sort_values(nom_ke_col)
        
        # Для каждой группы ищем последовательности
        for kmd_group, group_data in groups.items():
            if len(group_data) > max_points_per_series:
                continue
            
            masses = group_data['calc_mass'].to_numpy() # type: ignore
            for i in range(len(masses) - 1):
                for j in range(i + 1, len(masses)):
                    mass_diff = masses[j] - masses[i]
                    if mass_diff > 0 and mass_diff < 200:  # разумный диапазон
                        # Проверяем, нет ли уже такой трансформации
                        exists = False
                        for name, m in self.transformations.items():
                            if abs(mass_diff - abs(m)) < 0.01:
                                exists = True
                                break
                        
                        if not exists:
                            # Добавляем новую трансформацию
                            name = f"unknown_{mass_diff:.4f}"
                            self.transformations[name] = mass_diff
                            print(f"Добавлена новая трансформация: {name} = {mass_diff:.4f} Da")
    
    def get_node_attributes(self, node: int) -> dict:
        """Возвращает атрибуты узла"""
        return dict(self.G.nodes[node])
    
    def get_edge_attributes(self, u: int, v: int) -> dict:
        """Возвращает атрибуты ребра"""
        return dict(self.G.edges[u, v])
    
    def get_statistics(self) -> dict:
        """Возвращает статистику по графу"""
        stats = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'num_connected_components': nx.number_weakly_connected_components(self.G) if self.directed else nx.number_connected_components(self.G), # type: ignore
            'average_degree': 2 * self.G.number_of_edges() / max(1, self.G.number_of_nodes()),
            'reaction_counts': defaultdict(int)
        }
        
        # Подсчет реакций
        for _, _, edge_data in self.G.edges(data=True):
            reactions = edge_data.get('reactions', [edge_data.get('reaction', 'unknown')])
            for r in reactions:
                stats['reaction_counts'][r] += 1
        
        return stats
    
    def find_hubs(self, top_n: int = 10) -> List[Tuple[int, int]]:
        """
        Находит узлы-хабы (с наибольшим количеством связей)
        
        Возвращает
        ----------
        List[Tuple[int, int]] : Список (узел, степень)
        """
        degrees = dict(self.G.degree())
        sorted_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_hubs[:top_n]
    
    def find_reaction_paths(self, source: int, target: int, max_length: int = 5) -> List[List]:
        """
        Находит все пути реакций между двумя соединениями
        
        Параметры
        ----------
        source : int
            Индекс исходного узла
        target : int
            Индекс целевого узла
        max_length : int
            Максимальная длина пути
            
        Возвращает
        ----------
        List[List] : Список путей (каждый путь - список узлов)
        """
        paths = []
        if self.directed:
            paths = list(nx.all_simple_paths(self.G, source, target, cutoff=max_length))
        else:
            # Для ненаправленного графа
            paths = list(nx.all_simple_paths(self.G.to_undirected(), source, target, cutoff=max_length))
        return paths
    
    def export_to_cytoscape(self, filename: str):
        """
        Экспорт графа в формат для Cytoscape (JSON)
        """
        import json
        
        cytoscape_data = {
            'elements': {
                'nodes': [],
                'edges': []
            }
        }
        
        # Узлы
        for node, attrs in self.G.nodes(data=True):
            cytoscape_data['elements']['nodes'].append({
                'data': {
                    'id': str(node),
                    'label': attrs.get('formula', f'node_{node}'),
                    'mass': attrs.get('mass', 0),
                    'intensity': attrs.get('intensity', 0)
                }
            })
        
        # Ребра
        for u, v, attrs in self.G.edges(data=True):
            cytoscape_data['elements']['edges'].append({
                'data': {
                    'source': str(u),
                    'target': str(v),
                    'label': attrs.get('reaction', attrs.get('reactions', ['unknown'])[0]),
                    'mass_shift': attrs.get('mass_shift', 0)
                }
            })
        
        with open(filename, 'w') as f:
            json.dump(cytoscape_data, f, indent=2)
        
        print(f"Экспортировано в {filename}")
    
    def plot(self, 
             figsize: Tuple[int, int] = (14, 10),
             node_size_by: str = 'intensity',
             edge_color_by: str = 'reaction',
             max_nodes: int = 200,
             show_labels: bool = False):
        """
        Визуализация графа разностей масс
        
        Параметры
        ----------
        figsize : tuple
            Размер фигуры
        node_size_by : str
            Чем определять размер узла ('intensity', 'degree', 'mass')
        edge_color_by : str
            Чем определять цвет ребра ('reaction', 'mass_shift')
        max_nodes : int
            Максимальное количество узлов для отображения
        show_labels : bool
            Показывать ли подписи
        """
        import matplotlib.pyplot as plt
        
        # Если граф слишком большой, берем только хабы
        if self.G.number_of_nodes() > max_nodes:
            hubs = self.find_hubs(max_nodes)
            hub_nodes = [h[0] for h in hubs]
            G_sub = self.G.subgraph(hub_nodes)
        else:
            G_sub = self.G
        
        # Определяем размеры узлов
        node_sizes = []
        for node in G_sub.nodes():
            if node_size_by == 'intensity':
                size = G_sub.nodes[node].get('intensity', 100)
            elif node_size_by == 'degree':
                size = G_sub.degree(node) * 50
            elif node_size_by == 'mass':
                size = G_sub.nodes[node].get('mass', 500) / 10
            else:
                size = 100
            node_sizes.append(min(max(size, 30), 500))  # Ограничиваем
        
        # Определяем цвета ребер
        edge_colors = []
        for _, _, attrs in G_sub.edges(data=True):
            if edge_color_by == 'reaction':
                reaction = attrs.get('reaction', attrs.get('reactions', ['unknown'])[0])
                color_map = {
                    'CH2': 'green', '-CH2': 'lightgreen',
                    'O': 'red', '-O': 'pink',
                    'H2O': 'blue', '-H2O': 'lightblue',
                    'SO3': 'orange', '-SO3': 'gold',
                    'CO2': 'purple', '-CO2': 'violet',
                }
                edge_colors.append(color_map.get(reaction, 'gray'))
            else:
                shift = attrs.get('mass_shift', 0)
                edge_colors.append(plt.colormaps['viridis'](abs(shift) / 100))
        
        # Рисуем граф
        plt.figure(figsize=figsize)
        
        pos = nx.spring_layout(G_sub, k=2, iterations=50)
        
        nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, 
                               node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(G_sub, pos, edge_color=edge_colors, 
                               width=1.5, alpha=0.5, arrows=True if self.directed else False)
        
        if show_labels:
            labels = {node: G_sub.nodes[node].get('formula', str(node)) 
                     for node in G_sub.nodes()}
            nx.draw_networkx_labels(G_sub, pos, labels, font_size=8)
        
        plt.title(f"Difference Network\nNodes: {G_sub.number_of_nodes()}, Edges: {G_sub.number_of_edges()}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_reaction_star(self, 
                        node_id: int|None = None,
                        formula: str|None = None,
                        top_n_neighbors: int = 20,
                        figsize: Tuple[int, int] = (12, 12),
                        show_mass_shifts: bool = True,
                        edge_width_by_intensity: bool = True,
                        colormap: str = 'Set3',
                        save_path: str|None = None):
        """
        Визуализация n-конечной звезды для выбранного узла или для всех узлов (суммарно)
        
        Параметры
        ----------
        node_id : int
            Индекс узла (вершины) для визуализации
            Если None, строится суммарная звезда для всех узлов
        formula : str
            Формула соединения для поиска узла (если node_id не указан)
        top_n_neighbors : int
            Максимальное количество соседей для отображения
        figsize : tuple
            Размер фигуры
        show_mass_shifts : bool
            Показывать ли значения массовых сдвигов
        edge_width_by_intensity : bool
            Толщина ребер пропорциональна интенсивности
        colormap : str
            Цветовая схема для реакций
        save_path : str
            Путь для сохранения графика
        
        Возвращает
        ----------
        dict : Статистика по реакциям для визуализированного узла/всех узлов
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge, Circle
        import numpy as np
        from collections import Counter
        
        # Определяем целевой узел
        target_node = None
        if node_id is not None:
            target_node = node_id
        elif formula is not None:
            # Ищем узел по формуле
            for node, attrs in self.G.nodes(data=True):
                if attrs.get('formula', '') == formula:
                    target_node = node
                    break
            if target_node is None:
                raise ValueError(f"Формула '{formula}' не найдена в графе")
        
        # Собираем статистику по реакциям
        if target_node is None:
            # Суммарная статистика по всем узлам
            reaction_stats = self.get_statistics()['reaction_counts']
            title = f"Reaction Star - All Nodes ({self.G.number_of_nodes()} compounds)"
            center_label = f"Total Network\n{self.G.number_of_edges()} reactions"
        else:
            # Статистика для конкретного узла
            neighbors = list(self.G.neighbors(target_node))
            reaction_stats = Counter()
            
            for neighbor in neighbors:
                edge_data = self.G.get_edge_data(target_node, neighbor)
                if edge_data:
                    reactions = edge_data.get('reactions', [edge_data.get('reaction', 'unknown')])
                    for r in reactions:
                        reaction_stats[r] += 1
            
            # Добавляем информацию об узле
            node_attrs = self.G.nodes[target_node]
            formula_str = node_attrs.get('formula', f'Node {target_node}')
            mass = node_attrs.get('mass', 0)
            intensity = node_attrs.get('intensity', 0)
            
            title = f"Reaction Star: {formula_str}"
            center_label = f"{formula_str}\nm/z: {mass:.4f}\nInt: {intensity:.0f}"
        
        # Ограничиваем количество реакций
        reaction_items = list(reaction_stats.items())
        reaction_items.sort(key=lambda x: x[1], reverse=True)
        
        if len(reaction_items) > top_n_neighbors:
            # Группируем редкие реакции в "Other"
            top_items = reaction_items[:top_n_neighbors - 1]
            other_count = sum(count for _, count in reaction_items[top_n_neighbors - 1:])
            reaction_items = top_items + [('Other', other_count)]
        
        # Цвета для реакций
        colors = plt.cm.get_cmap(colormap, len(reaction_items))
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Параметры для лепестков
        num_reactions = len(reaction_items)
        angles = np.linspace(0, 2 * np.pi, num_reactions, endpoint=False)
        widths = 2 * np.pi / num_reactions  # Ширина каждого лепестка
        
        max_count = max(count for _, count in reaction_items) if reaction_items else 1
        
        # Нормализация радиусов (0-1 для наглядности)
        radii = [count / max_count for _, count in reaction_items]
        
        # Рисуем лепестки (bars в полярных координатах)
        bars = ax.bar(angles, radii, width=widths, bottom=0.1,
                    color=colors(range(num_reactions)), alpha=0.8,
                    edgecolor='white', linewidth=1)
        
        # Добавляем подписи реакций
        for i, (reaction, count) in enumerate(reaction_items):
            angle = angles[i]
            # Позиция для текста на внешнем крае
            text_radius = radii[i] + 0.15
            # Поворот текста
            rotation = np.degrees(angle)
            if rotation > 90 and rotation < 270:
                rotation += 180
            
            # Формируем текст
            if show_mass_shifts:
                mass_shift = self.transformations.get(reaction, 0)
                label = f"{reaction}\n({count})"
                if mass_shift != 0:
                    label += f"\n{mass_shift:.3f} Da"
            else:
                label = f"{reaction}\n({count})"
            
            ax.text(angle, text_radius, label,
                    ha='center', va='center', fontsize=9,
                    rotation=rotation, rotation_mode='anchor',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Рисуем центральный круг с информацией
        center_circle = Circle((0, 0), 0.1, transform=ax.transData._b, # type: ignore
                            facecolor='white', edgecolor='black', linewidth=2, zorder=10)
        ax.add_patch(center_circle)
        ax.text(0, 0, center_label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=11)
        
        # Настройка внешнего вида
        ax.set_ylim(0, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        
        # Заголовок
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Легенда (цветовая шкала не нужна, так как цвета уже подписаны)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено в {save_path}")
        
        
        return dict(reaction_items)


    def plot_reaction_sunburst(self,
                            node_id: int|None = None,
                            depth: int = 2,
                            max_branches: int = 15,
                            figsize: Tuple[int, int] = (14, 14),
                            show_percentages: bool = True,
                            save_path: str|None = None):
        """
        Визуализация иерархической структуры реакций в виде sunburst diagram
        
        Параметры
        ----------
        node_id : int
            Индекс узла-центра (если None, строится глобальный sunburst)
        depth : int
            Глубина иерархии (1 - только прямые реакции, 2 - реакции реакций)
        max_branches : int
            Максимальное количество ветвей на уровне
        figsize : tuple
            Размер фигуры
        show_percentages : bool
            Показывать проценты
        save_path : str
            Путь для сохранения
        """
        import plotly.express as px
        import plotly.graph_objects as go
        from collections import defaultdict
        
        # Собираем иерархические данные
        if node_id is None:
            # Глобальный sunburst: все реакции
            stats = self.get_statistics()['reaction_counts']
            data = []
            total = sum(stats.values())
            
            for reaction, count in stats.items():
                if count > 0:
                    percentage = (count / total) * 100 if show_percentages else count
                    label = f"{reaction}<br>{count} ({percentage:.1f}%)" if show_percentages else f"{reaction}<br>{count}"
                    data.append({
                        'ids': reaction,
                        'labels': reaction,
                        'parents': 'reactions',
                        'values': count,
                        'display_label': label
                    })
            
            # Корневой узел
            data.append({
                'ids': 'reactions',
                'labels': f"All Reactions<br>Total: {total}",
                'parents': '',
                'values': total,
                'display_label': f"All Reactions<br>Total: {total}"
            })
            
            title = "Reaction Sunburst - Global Network"
            
        else:
            # Sunburst для конкретного узла
            node_attrs = self.G.nodes[node_id]
            formula = node_attrs.get('formula', f'Node {node_id}')
            
            data = []
            
            # Уровень 0: Центральный узел
            center_id = f"center_{node_id}"
            data.append({
                'ids': center_id,
                'labels': formula,
                'parents': '',
                'values': 100,
                'display_label': formula
            })
            
            # Уровень 1: Прямые реакции
            neighbors = list(self.G.neighbors(node_id))
            reactions_level1 = defaultdict(int)
            
            for neighbor in neighbors:
                edge_data = self.G.get_edge_data(node_id, neighbor)
                if edge_data:
                    reactions = edge_data.get('reactions', [edge_data.get('reaction', 'unknown')])
                    for r in reactions:
                        reactions_level1[r] += 1
            
            # Ограничиваем количество
            sorted_reactions = sorted(reactions_level1.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_reactions) > max_branches:
                top_reactions = sorted_reactions[:max_branches-1]
                other_count = sum(count for _, count in sorted_reactions[max_branches-1:])
                if other_count > 0:
                    top_reactions.append(('Other', other_count))
            else:
                top_reactions = sorted_reactions
            
            for reaction, count in top_reactions:
                reaction_id = f"{center_id}_{reaction}"
                label = f"{reaction}<br>{count}"
                data.append({
                    'ids': reaction_id,
                    'labels': label,
                    'parents': center_id,
                    'values': count,
                    'display_label': label
                })
                
                # Уровень 2: Реакции второго порядка (если depth >= 2)
                if depth >= 2 and reaction in self.transformations:
                    # Ищем, какие соединения участвуют в этой реакции
                    second_order = self._get_second_order_reactions(node_id, reaction)
                    if second_order:
                        second_items = list(second_order.items())
                        second_items.sort(key=lambda x: x[1], reverse=True)
                        second_items = second_items[:max_branches]
                        
                        for sec_reaction, sec_count in second_items:
                            sec_id = f"{reaction_id}_{sec_reaction}"
                            sec_label = f"{sec_reaction}<br>{sec_count}"
                            data.append({
                                'ids': sec_id,
                                'labels': sec_label,
                                'parents': reaction_id,
                                'values': sec_count,
                                'display_label': sec_label
                            })
            
            title = f"Reaction Sunburst - {formula}"
        
        # Создаем sunburst диаграмму
        fig = px.sunburst(
            data,
            ids='ids',
            labels='labels',
            parents='parents',
            values='values',
            title=title,
            width=figsize[0]*80,
            height=figsize[1]*80
        )
        
        fig.update_traces(
            textinfo='label',
            textfont_size=12,
            marker=dict(colors=px.colors.qualitative.Set3)
        )
        
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Сохранено в {save_path}")
        
        fig.show()


    def _get_second_order_reactions(self, source_node: int, reaction_type: str) -> Counter:
        """
        Вспомогательная функция: собирает реакции второго порядка
        (реакции, в которых участвуют соединения, связанные с source_node через reaction_type)
        """
        
        
        second_order = Counter()
        
        # Находим всех соседей через указанную реакцию
        neighbors_via_reaction = []
        for neighbor in self.G.neighbors(source_node):
            edge_data = self.G.get_edge_data(source_node, neighbor)
            if edge_data:
                reactions = edge_data.get('reactions', [edge_data.get('reaction', 'unknown')])
                if reaction_type in reactions:
                    neighbors_via_reaction.append(neighbor)
        
        # Для каждого соседа собираем его реакции (кроме исходной)
        for neighbor in neighbors_via_reaction:
            for second_neighbor in self.G.neighbors(neighbor):
                if second_neighbor != source_node:
                    edge_data = self.G.get_edge_data(neighbor, second_neighbor)
                    if edge_data:
                        reactions = edge_data.get('reactions', [edge_data.get('reaction', 'unknown')])
                        for r in reactions:
                            if r != reaction_type:
                                second_order[r] += 1
        
        return second_order


    def plot_reaction_hierarchy(self,
                            figsize: Tuple[int, int] = (12, 8),
                            top_n_reactions: int = 15,
                            color_by_type: bool = True,
                            save_path: str|None = None):
        """
        Иерархическая визуализация реакций в виде stacked bar chart
        
        Параметры
        ----------
        figsize : tuple
            Размер фигуры
        top_n_reactions : int
            Количество отображаемых реакций
        color_by_type : bool
            Цвет по типу реакции (окисление, гидрирование, и т.д.)
        save_path : str
            Путь для сохранения
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        stats = self.get_statistics()['reaction_counts']
        
        # Сортируем реакции по частоте
        sorted_reactions = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        top_reactions = sorted_reactions[:top_n_reactions]
        
        # Группировка по типу для цветовой маркировки
        type_colors = {
            'oxidation': '#ff9999',      # O, OH, OCH3
            'reduction': '#99ff99',      # -O, -H2
            'methylation': '#99ccff',    # CH2, CH3, CH2O
            'hydration': '#ffcc99',      # H2O, OH
            'sulfonation': '#ff99cc',    # S, SO2, SO3
            'decarboxylation': '#cc99ff', # CO2, CO
            'other': '#dddddd'
        }
        
        def get_reaction_type(reaction_name):
            if reaction_name in ['O', 'OH', 'OCH3', 'COOH']:
                return 'oxidation'
            elif reaction_name in ['-O', '-H2', '-CO2']:
                return 'reduction'
            elif reaction_name in ['CH2', 'CH3', 'CH2O', 'C2H4O']:
                return 'methylation'
            elif reaction_name in ['H2O', 'OH']:
                return 'hydration'
            elif reaction_name in ['S', 'SO2', 'SO3', 'H2SO3']:
                return 'sulfonation'
            elif reaction_name in ['CO2', 'CO', '-CO2']:
                return 'decarboxylation'
            else:
                return 'other'
        
        # Создаем фигуру
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # График 1: Горизонтальная гистограмма
        reactions = [r[0] for r in top_reactions]
        counts = [r[1] for r in top_reactions]
        
        colors = [type_colors[get_reaction_type(r)] if color_by_type else 'steelblue' 
                for r in reactions]
        
        bars = ax1.barh(reactions, counts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Number of occurrences', fontsize=12)
        ax1.set_title(f'Top {top_n_reactions} Reactions in Network', fontsize=14)
        ax1.invert_yaxis()
        
        # Добавляем значения на концах баров
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{count}', va='center', fontsize=10)
        
        # График 2: Круговая диаграмма по типам
        if color_by_type:
            type_counts = defaultdict(int)
            for reaction, count in stats.items():
                r_type = get_reaction_type(reaction)
                type_counts[r_type] += count
            
            # Фильтруем только типы с ненулевым количеством
            type_counts = {k: v for k, v in type_counts.items() if v > 0}
            
            labels = [t.capitalize() for t in type_counts.keys()]
            sizes = list(type_counts.values())
            colors_pie = [type_colors[t] for t in type_counts.keys()]
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                                autopct='%1.1f%%', startangle=90,
                                                textprops={'fontsize': 11})
            
            # Настройка внешнего вида
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax2.set_title('Reactions by Type', fontsize=14)
        
        plt.suptitle(f'Reaction Network Analysis\nTotal: {self.G.number_of_edges()} edges, {len(stats)} reaction types',
                    fontsize=12, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено в {save_path}")
        


    def compare_reaction_stars(self, 
                            node_ids: List[int],
                            figsize: Tuple[int, int] = (15, 10),
                            save_path: str|None = None):
        """
        Сравнение реакционных звезд для нескольких узлов
        
        Параметры
        ----------
        node_ids : List[int]
            Список индексов узлов для сравнения
        figsize : tuple
            Размер фигуры
        save_path : str
            Путь для сохранения
        """
        import matplotlib.pyplot as plt
        from collections import Counter
        
        n_nodes = len(node_ids)
        cols = min(3, n_nodes)
        rows = (n_nodes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                                subplot_kw=dict(projection='polar'))
        if n_nodes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        all_reactions = set()
        nodes_data = []
        
        # Собираем данные по каждому узлу
        for node_id in node_ids:
            node_attrs = self.G.nodes[node_id]
            formula = node_attrs.get('formula', f'Node {node_id}')
            
            reaction_stats = Counter()
            for neighbor in self.G.neighbors(node_id):
                edge_data = self.G.get_edge_data(node_id, neighbor)
                if edge_data:
                    reactions = edge_data.get('reactions', [edge_data.get('reaction', 'unknown')])
                    for r in reactions:
                        reaction_stats[r] += 1
                        all_reactions.add(r)
            
            nodes_data.append({
                'id': node_id,
                'formula': formula,
                'stats': reaction_stats,
                'mass': node_attrs.get('mass', 0)
            })
        
        # Сортируем реакции для единой шкалы
        all_reactions = sorted(all_reactions)
        
        # Рисуем звезды
        for idx, (ax, node_data) in enumerate(zip(axes, nodes_data)):
            stats = node_data['stats']
            formula = node_data['formula']
            
            # Подготавливаем данные
            reaction_counts = [stats.get(r, 0) for r in all_reactions]
            max_count = max(reaction_counts) if reaction_counts else 1
            radii = [c / max_count for c in reaction_counts]
            
            angles = np.linspace(0, 2 * np.pi, len(all_reactions), endpoint=False)
            widths = 2 * np.pi / len(all_reactions)
            
            # Цвет в зависимости от интенсивности
            colors = plt.colormaps['RdYlGn_r']([r for r in radii])
            
            bars = ax.bar(angles, radii, width=widths, bottom=0.05,
                        color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Центральный круг
            center = pth.Circle((0, 0), 0.08, transform=ax.transData._b,
                            facecolor='white', edgecolor='black', linewidth=1)
            ax.add_patch(center)
            ax.text(0, 0, formula[:15], ha='center', va='center', fontsize=7, fontweight='bold')
            
            # Заголовок
            ax.set_title(f"{formula[:20]}\nm/z: {node_data['mass']:.2f}", fontsize=10)
            
            ax.set_ylim(0, 1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
        
        # Скрываем пустые subplots
        for idx in range(len(nodes_data), len(axes)):
            axes[idx].set_visible(False)
        
        # Общая легенда для реакций
        legend_elements = []
        for i, reaction in enumerate(all_reactions[:15]):  # Ограничиваем легенду
            legend_elements.append(pth.Rectangle((0, 0), 1, 1, facecolor='gray', 
                                                alpha=0.5, label=reaction))
        
        if legend_elements:
            fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                    fontsize=8, bbox_to_anchor=(0.5, -0.05))
        
        plt.suptitle(f'Reaction Stars Comparison ({len(node_ids)} compounds)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено в {save_path}")
        
        plt.show()


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def build_difference_network(data: pd.DataFrame,
                            mass_col: str = 'calc_mass',
                            intensity_col: str = 'intensity',
                            formula_col: str = 'formula',
                            tolerance_ppm: float = 3.0,
                            min_intensity: float = 0.0,
                            directed: bool = True,
                            **kwargs) -> DifferenceNetwork:
    """
    Упрощенная функция для построения графа разностей масс
    
    Пример использования:
    >>> G = build_difference_network(spectrum, tolerance_ppm=2.0, min_intensity=0.1)
    >>> G.plot()
    >>> stats = G.get_statistics()
    >>> print(stats['reaction_counts'])
    """
    network = DifferenceNetwork(
        tolerance_ppm=tolerance_ppm,
        min_intensity=min_intensity,
        directed=directed
    )
    
    # Добавляем пользовательские трансформации из kwargs
    if 'transformations' in kwargs:
        for name, mass in kwargs['transformations'].items():
            network.add_custom_transformation(name, mass)
    
    network.build(data, mass_col, intensity_col, formula_col)
    
    return network

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
        
        spectra_list.append(spectra)
        spectra_list_name.append(spectra_name)

fragment = 'C10H12O4'

for data in spectra_list:

    data = kmd.kendrick(data,fragment_formula=fragment,method='round')

    if data is None:
        raise ValueError('')

    plot_homologous_series(data,fragment=fragment)
    plt.savefig(fr"C:\Users\Kirill\Desktop\Lab\Process_mass_spectra\process_result\kendrik\{data.attrs['name']}.png")
    
    continue
    network = build_difference_network(
        data=data,
        tolerance_ppm=0.5,
        min_intensity=0,
        directed=True,
        formula_col='brutto'
    )
    
    # Статистика
    stats = network.get_statistics()
    print("\nСтатистика графа:")
    for key, value in stats.items():
        if key != 'reaction_counts':
            print(f"  {key}: {value}")
    
    print("\nРеакции:")
    for reaction, count in stats['reaction_counts'].items():
        print(f"  {reaction}: {count}")
            
    network.plot_reaction_hierarchy(top_n_reactions=20,save_path=fr"C:\Users\Kirill\Desktop\Lab\Process_mass_spectra\process_result\plot_reaction_hierarchy\{data.attrs['name']}.png")
    network.plot_reaction_star(top_n_neighbors=20, show_mass_shifts=True, save_path=fr"C:\Users\Kirill\Desktop\Lab\Process_mass_spectra\process_result\plot_reaction_star\{data.attrs['name']}.png")