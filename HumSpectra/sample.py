import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Any, List, Union, Literal, Tuple
import json
import hashlib
from datetime import datetime
import itertools
from collections import defaultdict, Counter
import matplotlib as mpl

@dataclass
class Sample:
    """Класс для хранения всех данных образца"""
    sample_id: str
    sample_subclass: str
    sample_class: str
    
    # Основные спектральные данные
    fluorescence_eem: Optional[pd.DataFrame] = None
    uv_vis_absorption: Optional[pd.DataFrame] = None
    uv_vis_reflection: Optional[pd.DataFrame] = None
    
    cutted_fluo_eem: Optional[pd.DataFrame] = None
    
    uv_vis_absorption_smooth: Optional[pd.DataFrame] = None
    uv_vis_reflection_smooth: Optional[pd.DataFrame] = None
    
    #Дополнительные данные
    org_carbon: Optional[float] = None
    pH: Optional[float] = None
    Eh: Optional[float] = None
    
    #Координаты
    latitude: Optional[float] = None
    lontitude: Optional[float] = None
    
    # Параметры измерений
    measurement_params: Dict[str, Any] = field(default_factory=dict)
    
    # Рассчитанные дескрипторы
    descriptors: Dict[str, float] = field(default_factory=dict)
    
    # Метаданные
    file_path: Optional[Path] = None
    measurement_date: Optional[str] = None
    sample_tags: Set[str] = field(default_factory=set)
    comments: str = ""
    
    # Вспомогательные данные
    _data_hash: Optional[str] = None
    
    def calculate_hash(self) -> str:
        """Вычисляет хеш основных данных для отслеживания изменений"""
        data_string = ""
        if self.fluorescence_eem is not None:
            data_string += pickle.dumps(self.fluorescence_eem).hex()
        if self.uv_vis_absorption is not None:
            data_string += pickle.dumps(self.uv_vis_absorption).hex()
        
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def update_descriptors(self, descriptor_dict: Dict[str, float]):
        """Обновляет дескрипторы"""
        self.descriptors.update(descriptor_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сериализации"""
        return {
            'sample_id': self.sample_id,
            'sample_subclass': self.sample_subclass,
            'sample_class': self.sample_class,
            'fluorescence_eem': self.fluorescence_eem,
            'uv_vis_absorption': self.uv_vis_absorption,
            'uv_vis_reflection': self.uv_vis_reflection,
            'cutted_fluo_eem': self.cutted_fluo_eem,
            'uv_vis_absorption_smooth': self.uv_vis_absorption_smooth,
            'uv_vis_reflection_smooth': self.uv_vis_reflection_smooth,
            'org_carbon': self.org_carbon,
            'pH': self.pH,
            'Eh': self.Eh,
            'latitude': self.latitude,
            'lontitude': self.lontitude,
            'measurement_params': self.measurement_params,
            'descriptors': self.descriptors,
            'file_path': str(self.file_path) if self.file_path else None,
            'measurement_date': self.measurement_date,
            'sample_tags': self.sample_tags,
            'comments': self.comments,
            '_data_hash': self._data_hash
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Sample':
        """Создает объект Sample из словаря (альтернативный конструктор)"""
        # Обрабатываем специальные поля
        file_path = Path(data['file_path']) if data.get('file_path') else None
        
        return cls(
            sample_id=data['sample_id'],
            sample_subclass=data['sample_subclass'],
            sample_class=data['sample_class'],
            fluorescence_eem=data.get('fluorescence_eem'),
            uv_vis_absorption=data.get('uv_vis_absorption'),
            uv_vis_reflection=data.get('uv_vis_reflection'),
            cutted_fluo_eem=data.get('cutted_fluo_eem'),
            uv_vis_absorption_smooth=data.get('uv_vis_absorption_smooth'),
            uv_vis_reflection_smooth=data.get('uv_vis_reflection_smooth'),
            org_carbon=data.get('org_carbon'),
            pH=data.get('pH'),
            Eh=data.get('Eh'),
            latitude=data.get('latitude'),
            lontitude=data.get('lontitude'),
            measurement_params=data.get('measurement_params', {}),
            descriptors=data.get('descriptors', {}),
            file_path=file_path,
            measurement_date=data.get('measurement_date'),
            sample_tags=data.get('sample_tags', []),
            comments=data.get('comments', ''),
            _data_hash=data.get('_data_hash')
        )
        
    def add_tag(self, *tags: str):
        """Добавляет один или несколько тегов"""
        self.sample_tags.update(tags)
    
    def remove_tag(self, tag: str):
        """Удаляет тег"""
        self.sample_tags.discard(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Проверяет наличие тега"""
        return tag in self.sample_tags
    
    def has_all_tags(self, *tags: str) -> bool:
        """Проверяет наличие всех указанных тегов"""
        return all(tag in self.sample_tags for tag in tags)
    
    def has_any_tag(self, *tags: str) -> bool:
        """Проверяет наличие хотя бы одного из тегов"""
        return any(tag in self.sample_tags for tag in tags)
    
    def clear_tags(self):
        """Очищает все теги"""
        self.sample_tags.clear()

class SampleCollection:
    """Коллекция образцов с возможностью сохранения/загрузки"""
    
    def __init__(self, cache_dir: Optional[Path] = None, samples: Optional[List[Sample]] = None):
        self.samples: Dict[str, Sample] = {}
        self.cache_dir = cache_dir or Path("./sample_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._tag_index: Dict[str, Set[str]] = {}
        self._tag_cooccurrence: Optional[Dict[Tuple[str, str], int]] = None
        
        if samples:
            self.add_samples(samples)
        
    def add_sample(self, sample: Sample):
        """Добавляет образец в коллекцию"""
        if sample._data_hash is None:
            sample._data_hash = sample.calculate_hash()
        self.samples[sample.sample_id] = sample
        self._update_tag_index(sample)
        # Сбрасываем кэш совместной встречаемости при добавлении нового образца
        self._tag_cooccurrence = None
    
    def create_sample_from_data(
        self,
        sample_id: str,
        fluorescence_eem: pd.DataFrame|None = None,
        uv_vis_absorption: pd.DataFrame|None = None,
        measurement_params: Dict|None = None,
        **kwargs
    ) -> Sample:
        """Создает образец из исходных данных"""
        sample = Sample(
            sample_id=sample_id,
            fluorescence_eem=fluorescence_eem,
            uv_vis_absorption=uv_vis_absorption,
            measurement_params=measurement_params or {},
            **kwargs
        )
        sample._data_hash = sample.calculate_hash()
        self.add_sample(sample)
        return sample
    
    def save_sample(self, sample_id: str):
        """Сохраняет отдельный образец в файл"""
        if sample_id not in self.samples:
            raise KeyError(f"Sample {sample_id} not found")
        
        sample = self.samples[sample_id]
        file_path = self.cache_dir / f"{sample_id}.pkl"
        
        with open(file_path, 'wb') as f:
            pickle.dump(sample.to_dict(), f)
        
        # Также сохраняем в JSON для читаемости (без массивов)
        json_path = self.cache_dir / f"{sample_id}_meta.json"
        json_data = {
            k: v for k, v in sample.to_dict().items() 
            if not isinstance(v, np.ndarray)
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
            
    def _update_tag_index(self, sample: Sample):
        """Обновляет индекс тегов"""
        for tag in sample.sample_tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(sample.sample_id)
    
    def load_sample(self, sample_id: str) -> Sample:
        """Загружает образец из файла"""
        file_path = self.cache_dir / f"{sample_id}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No cache for sample {sample_id}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Используем фабричный метод
        sample = Sample.from_dict(data)
        self.samples[sample_id] = sample
        return sample
    
    def save_collection(self, filename: str = "sample_collection.pkl"):
        """Сохраняет всю коллекцию"""
        file_path = self.cache_dir / filename
        with open(file_path, 'wb') as f:
            pickle.dump([s.to_dict() for s in self.samples.values()], f)
    
    def load_collection(self, filename: str = "sample_collection.pkl"):
        """Загружает всю коллекцию"""
        file_path = self.cache_dir / filename
        with open(file_path, 'rb') as f:
            samples_data = pickle.load(f)
        
        self.samples.clear()
        self._tag_index.clear()  # Очищаем индекс
        
        for data in samples_data:
            sample = Sample.from_dict(data)
            self.samples[sample.sample_id] = sample
            
        # ВАЖНО: перестраиваем индекс после загрузки всех образцов
        self._rebuild_tag_index()

    def _rebuild_tag_index(self):
        """Перестраивает индекс тегов с нуля"""
        self._tag_index.clear()
        for sample_id, sample in self.samples.items():
            for tag in sample.sample_tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(sample_id)
            
    def filter_by_tag(self, tag: str) -> List[Sample]:
        """Возвращает все образцы с указанным тегом"""
        sample_ids = self._tag_index.get(tag, set())
        return [self.samples[sid] for sid in sample_ids]
    
    def filter_by_tags_all(self, *tags: str) -> List[Sample]:
        """Возвращает образцы, имеющие ВСЕ указанные теги"""
        if not tags:
            return list(self.samples.values())
        
        # Находим пересечение ID образцов по тегам
        common_ids = None
        for tag in tags:
            tag_ids = self._tag_index.get(tag, set())
            if common_ids is None:
                common_ids = tag_ids
            else:
                common_ids = common_ids.intersection(tag_ids)
            
            if not common_ids:  # Нет пересечения
                return []
        
        return [self.samples[sid] for sid in common_ids] if common_ids else []
    
    def filter_by_tags_any(self, *tags: str) -> List[Sample]:
        """Возвращает образцы, имеющие ХОТЯ БЫ ОДИН из указанных тегов"""
        if not tags:
            return list(self.samples.values())
        
        # Объединяем ID образцов по всем тегам
        all_ids = set()
        for tag in tags:
            all_ids.update(self._tag_index.get(tag, set()))
        
        return [self.samples[sid] for sid in all_ids]
    
    def filter_by_tags_none(self, *tags: str) -> List[Sample]:
        """Возвращает образцы, НЕ имеющие ни одного из указанных тегов"""
        if not tags:
            return list(self.samples.values())
        
        # Собираем все ID с любым из тегов
        excluded_ids = set()
        for tag in tags:
            excluded_ids.update(self._tag_index.get(tag, set()))
        
        return [s for sid, s in self.samples.items() 
                if sid not in excluded_ids]
    
    def get_all_tags(self) -> Set[str]:
        """Возвращает все уникальные теги в коллекции"""
        return set(self._tag_index.keys())
    
    def get_samples_without_tags(self) -> List[Sample]:
        """Возвращает образцы без тегов"""
        return [s for s in self.samples.values() if not s.sample_tags]
    
    def add_samples(self, samples: List[Sample]):
        """Добавляет несколько образцов в коллекцию"""
        for sample in samples:
            self.add_sample(sample)
    
    @classmethod
    def from_samples(cls, samples: List[Sample], cache_dir: Optional[Path] = None) -> 'SampleCollection':
        """Создает коллекцию из списка образцов (фабричный метод)"""
        collection = cls(cache_dir=cache_dir)
        collection.add_samples(samples)
        return collection
    
    def create_common_descriptors_table(self, 
                                      fill_na: Any = np.nan,
                                      sample_attributes: Optional[List[str]] = None,
                                      sort_by: Optional[str] = None) -> pd.DataFrame:
        """
        Создает DataFrame с дескрипторами, общими для всех образцов.
        
        Параметры:
            fill_na: значение для заполнения пропусков (по умолчанию NaN)
            sample_attributes: дополнительные атрибуты образцов для включения в таблицу
                              (например, ['sample_type', 'pH', 'org_carbon'])
            sort_by: имя столбца для сортировки (None - без сортировки)
            
        Возвращает:
            DataFrame где:
            - строки: образцы
            - столбцы: sample_id + общие дескрипторы + дополнительные атрибуты
            - только образцы, имеющие ВСЕ общие дескрипторы
        """
        if not self.samples:
            return pd.DataFrame()
        
        # 1. Находим дескрипторы, общие для всех образцов
        all_descriptor_sets = [set(sample.descriptors.keys()) 
                              for sample in self.samples.values()]
        
        # Находим пересечение всех множеств дескрипторов
        common_descriptors = set.intersection(*all_descriptor_sets) if all_descriptor_sets else set()
        
        if not common_descriptors:
            print("Нет дескрипторов, общих для всех образцов")
            return pd.DataFrame()
        
        print(f"Найдено {len(common_descriptors)} общих дескрипторов: {sorted(common_descriptors)}")
        
        # 2. Собираем данные только для образцов, имеющих все общие дескрипторы
        rows = []
        sample_ids_included = []
        sample_ids_excluded = []
        
        for sample_id, sample in self.samples.items():
            # Проверяем, есть ли у образца все общие дескрипторы
            sample_descriptors = set(sample.descriptors.keys())
            if not common_descriptors.issubset(sample_descriptors):
                missing = common_descriptors - sample_descriptors
                sample_ids_excluded.append((sample_id, missing))
                continue
            
            # Создаем строку для DataFrame
            row = {'sample_id': sample_id}
            
            # Добавляем значения общих дескрипторов
            for desc in sorted(common_descriptors):
                row[desc] = sample.descriptors.get(desc, fill_na) # type: ignore
            
            # Добавляем дополнительные атрибуты образца
            if sample_attributes:
                for attr in sample_attributes:
                    if hasattr(sample, attr):
                        row[attr] = getattr(sample, attr, fill_na)
                    else:
                        row[attr] = fill_na
            
            rows.append(row)
            sample_ids_included.append(sample_id)
        
        # 3. Создаем DataFrame
        if not rows:
            print("Нет образцов, содержащих все общие дескрипторы")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Устанавливаем sample_id как индекс, если нужно
        df.set_index('sample_id', inplace=True)
        
        # Сортируем, если указано
        if sort_by and sort_by in df.columns:
            df.sort_values(by=sort_by, inplace=True)
        
        # 4. Выводим статистику
        print(f"\nСтатистика:")
        print(f"  Всего образцов в коллекции: {len(self.samples)}")
        print(f"  Образцов включено в таблицу: {len(sample_ids_included)}")
        print(f"  Образцов исключено: {len(sample_ids_excluded)}")
        
        if sample_ids_excluded:
            print(f"\nИсключенные образцы (отсутствующие дескрипторы):")
            for sample_id, missing in sample_ids_excluded[:5]:  # Показываем первые 5
                print(f"  {sample_id}: отсутствуют {missing}")
            if len(sample_ids_excluded) > 5:
                print(f"  ... и еще {len(sample_ids_excluded) - 5} образцов")
        
        return df
    
    def create_flexible_descriptors_table(self,
                                         min_coverage: float = 1.0,
                                         fill_na: Any = np.nan,
                                         sample_attributes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Создает таблицу с дескрипторами, которые есть хотя бы у min_coverage*100% образцов.
        
        Параметры:
            min_coverage: минимальная доля образцов, имеющих дескриптор (0.0-1.0)
            fill_na: значение для пропусков
            sample_attributes: дополнительные атрибуты образцов
            
        Возвращает:
            DataFrame с дескрипторами, удовлетворяющими условию покрытия
        """
        if not self.samples:
            return pd.DataFrame()
        
        total_samples = len(self.samples)
        min_samples_required = int(total_samples * min_coverage)
        
        # Собираем статистику по дескрипторам
        descriptor_stats = {}
        for sample in self.samples.values():
            for desc_name in sample.descriptors:
                if desc_name not in descriptor_stats:
                    descriptor_stats[desc_name] = 0
                descriptor_stats[desc_name] += 1
        
        # Выбираем дескрипторы с достаточным покрытием
        selected_descriptors = [
            desc for desc, count in descriptor_stats.items()
            if count >= min_samples_required
        ]
        
        print(f"Дескрипторы с покрытием >= {min_coverage*100}%:")
        print(f"  Всего дескрипторов в коллекции: {len(descriptor_stats)}")
        print(f"  Отобранных дескрипторов: {len(selected_descriptors)}")
        
        if not selected_descriptors:
            return pd.DataFrame()
        
        # Создаем таблицу
        rows = []
        for sample_id, sample in self.samples.items():
            row = {'sample_id': sample_id}
            
            # Добавляем значения выбранных дескрипторов
            for desc in sorted(selected_descriptors):
                row[desc] = sample.descriptors.get(desc, fill_na) # type: ignore
            
            # Добавляем дополнительные атрибуты
            if sample_attributes:
                for attr in sample_attributes:
                    if hasattr(sample, attr):
                        row[attr] = getattr(sample, attr, fill_na)
                    else:
                        row[attr] = fill_na
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.set_index('sample_id', inplace=True)
        
        return df
    
    def _calculate_tag_cooccurrence(self) -> Dict[Tuple[str, str], int]:
        """
        Рассчитывает матрицу совместного встречания тегов.
        Возвращает словарь: {(tag1, tag2): count}
        """
        cooccurrence = {}
        tag_pairs = set()
        
        # Собираем все пары тегов для каждого образца
        for sample in self.samples.values():
            tags = list(sample.sample_tags)
            if len(tags) >= 2:
                # Генерируем все возможные пары тегов (без повторений)
                for tag1, tag2 in itertools.combinations(sorted(tags), 2):
                    pair = (tag1, tag2)
                    if pair not in cooccurrence:
                        cooccurrence[pair] = 0
                    cooccurrence[pair] += 1
        
        # Добавляем обратные пары для удобства
        cooccurrence_symmetric = cooccurrence.copy()
        for (tag1, tag2), count in cooccurrence.items():
            cooccurrence_symmetric[(tag2, tag1)] = count
        
        return cooccurrence_symmetric
    
    def visualize_tags_as_graph(self, 
                               min_cooccurrence: int = 1,
                               layout: str = 'spring',
                               node_size_multiplier: float = 100,
                               edge_width_multiplier: float = 2,
                               figsize: tuple = (15, 10),
                               title: Optional[str] = None,
                               show_isolated_nodes: bool = True):
        """
        Визуализирует теги коллекции в виде графа.
        
        Параметры:
        ----------
        min_cooccurrence : int
            Минимальное количество совместных вхождений для отображения связи
        layout : str
            Алгоритм расположения узлов:
            - 'spring' (по умолчанию) - пружинный алгоритм
            - 'circular' - круговая раскладка
            - 'kamada_kawai' - алгоритм Камады-Каваи
            - 'random' - случайное расположение
        node_size_multiplier : float
            Множитель для размера узлов (пропорционален частоте тега)
        edge_width_multiplier : float
            Множитель для толщины связей (пропорционален совместной встречаемости)
        figsize : tuple
            Размер графика
        title : Optional[str]
            Заголовок графика
        show_isolated_nodes : bool
            Показывать ли теги без связей с другими тегами
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import matplotlib.cm as cm
            
            # Собираем статистику по тегам
            tag_counts = Counter()
            for sample in self.samples.values():
                for tag in sample.sample_tags:
                    tag_counts[tag] += 1
            
            if not tag_counts:
                print("В коллекции нет тегов для визуализации")
                return
            
            # Рассчитываем совместную встречаемость
            if self._tag_cooccurrence is None:
                self._tag_cooccurrence = self._calculate_tag_cooccurrence()
            
            # Создаем граф
            G = nx.Graph()
            
            # Добавляем узлы (теги) с атрибутами
            for tag, count in tag_counts.items():
                G.add_node(tag, count=count, size=count * node_size_multiplier)
            
            # Добавляем ребра (связи между тегами)
            edges_to_add = []
            for (tag1, tag2), cooccurrence_count in self._tag_cooccurrence.items():
                if (tag1 in G.nodes() and tag2 in G.nodes() and 
                    tag1 != tag2 and cooccurrence_count >= min_cooccurrence):
                    edges_to_add.append((tag1, tag2, cooccurrence_count))
            
            # Добавляем ребра с весом
            for tag1, tag2, weight in edges_to_add:
                G.add_edge(tag1, tag2, weight=weight, width=weight * edge_width_multiplier)
            
            # Удаляем изолированные узлы если не нужно их показывать
            if not show_isolated_nodes:
                isolated_nodes = list(nx.isolates(G))
                if isolated_nodes:
                    print(f"Скрыто {len(isolated_nodes)} изолированных тегов: {isolated_nodes}")
                    G.remove_nodes_from(isolated_nodes)
            
            # Проверяем, есть ли что рисовать
            if len(G.nodes()) == 0:
                print("Нет узлов для отображения после фильтрации")
                return
            
            # Создаем фигуру
            plt.figure(figsize=figsize)
            
            # Выбираем алгоритм расположения
            if layout == 'spring':
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            elif layout == 'random':
                pos = nx.random_layout(G, seed=42)
            else:
                print(f"Неизвестный тип раскладки: {layout}. Используется spring.")
                pos = nx.spring_layout(G, seed=42)
            
            # Подготавливаем данные для визуализации
            node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
            node_counts = [G.nodes[node]['count'] for node in G.nodes()]
            
            # Цвет узлов по частоте тега
            max_count = max(node_counts) if node_counts else 1
            node_colors = [count / max_count for count in node_counts]
            
            # Толщина ребер
            edge_weights = [G.edges[edge]['width'] for edge in G.edges()]
            
            # Рисуем граф
            # 1. Рисуем ребра
            nx.draw_networkx_edges(
                G, pos,
                width=edge_weights,
                alpha=0.7,
                edge_color='gray'
            )
            
            # 2. Рисуем узлы
            nodes = nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=plt.colormaps['viridis'],
                alpha=0.8,
                edgecolors='black',
                linewidths=1
            )
            
            # 3. Добавляем подписи узлов
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_weight='bold',
                font_color='white'
            )
            
            # 4. Добавляем подписи к ребрам (веса)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            if edge_labels:
                # Нормализуем веса для отображения
                max_weight = max(edge_labels.values()) if edge_labels else 1
                scaled_labels = {k: f"{v:.0f}" for k, v in edge_labels.items()}
                
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels=scaled_labels,
                    font_size=8,
                    font_color='darkred'
                )
            
            # Добавляем цветовую шкалу
            sm = plt.cm.ScalarMappable(cmap=plt.colormaps['viridis'], 
                                      norm=plt.Normalize(vmin=0, vmax=max_count)) # type: ignore
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
            cbar.set_label('Частота тега', fontsize=10)
            
            # Настраиваем заголовок и метки
            if title is None:
                title = f"Граф тегов коллекции ({len(tag_counts)} тегов, {len(edges_to_add)} связей)"
            
            plt.title(title, fontsize=14, pad=20)
            plt.axis('off')
            
            # Добавляем легенду с информацией
            legend_text = f"Всего образцов: {len(self.samples)}\n"
            legend_text += f"Всего уникальных тегов: {len(tag_counts)}\n"
            legend_text += f"Мин. совместная встречаемость: {min_cooccurrence}\n"
            legend_text += f"Размер узла ∝ частоте тега\n"
            legend_text += f"Толщина ребра ∝ совместной встречаемости"
            
            plt.figtext(0.02, 0.02, legend_text, 
                       fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor="lightgray", 
                                alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            # Выводим статистику
            self._print_tag_statistics(tag_counts, edges_to_add)
            
        except ImportError as e:
            print(f"Для визуализации графов требуется установить дополнительные библиотеки:")
            print("pip install matplotlib networkx")
            print(f"Ошибка импорта: {e}")
    
    def _print_tag_statistics(self, tag_counts: Counter, edges: List[tuple]):
        """Выводит статистику по тегам"""
        print("\n" + "="*60)
        print("СТАТИСТИКА ТЕГОВ:")
        print("="*60)
        
        # Статистика по тегам
        print(f"\nВсего образцов: {len(self.samples)}")
        print(f"Всего уникальных тегов: {len(tag_counts)}")
        
        # Топ-10 самых частых тегов
        print(f"\nТоп-10 самых частых тегов:")
        for tag, count in tag_counts.most_common(10):
            percentage = (count / len(self.samples)) * 100
            print(f"  {tag}: {count} образцов ({percentage:.1f}%)")
        
        # Теги, встречающиеся только один раз
        single_tags = [tag for tag, count in tag_counts.items() if count == 1]
        if single_tags:
            print(f"\nТеги, встречающиеся только один раз ({len(single_tags)}):")
            print(f"  {', '.join(single_tags[:10])}" + 
                  ("..." if len(single_tags) > 10 else ""))
        
        # Статистика по связям
        if edges:
            print(f"\nВсего связей между тегами: {len(edges)}")
            
            # Самые сильные связи
            sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
            print(f"\nТоп-5 самых сильных связей:")
            for tag1, tag2, weight in sorted_edges[:5]:
                print(f"  {tag1} ↔ {tag2}: {weight} совместных вхождений")
        else:
            print("\nНет связей между тегами (или все связи отфильтрованы)")
        
        print("="*60)
    
    def get_tag_network_data(self) -> Dict[str, Any]:
        """
        Возвращает данные графа тегов в структурированном виде.
        Полезно для экспорта или дальнейшей обработки.
        
        Returns:
        --------
        Dict с ключами:
            - 'nodes': список словарей с информацией о тегах
            - 'edges': список словарей с информацией о связях
            - 'metadata': общая статистика
        """
        if self._tag_cooccurrence is None:
            self._tag_cooccurrence = self._calculate_tag_cooccurrence()
        
        # Собираем информацию о тегах
        tag_counts = Counter()
        for sample in self.samples.values():
            for tag in sample.sample_tags:
                tag_counts[tag] += 1
        
        # Формируем узлы
        nodes = []
        for tag, count in tag_counts.items():
            nodes.append({
                'id': tag,
                'label': tag,
                'count': count,
                'percentage': (count / len(self.samples)) * 100
            })
        
        # Формируем ребра
        edges = []
        for (tag1, tag2), weight in self._tag_cooccurrence.items():
            if tag1 != tag2:
                edges.append({
                    'source': tag1,
                    'target': tag2,
                    'weight': weight,
                    'strength': weight / min(tag_counts[tag1], tag_counts[tag2])
                })
        
        # Метаданные
        metadata = {
            'total_samples': len(self.samples),
            'total_tags': len(tag_counts),
            'total_edges': len(edges),
            'max_tag_count': max(tag_counts.values()) if tag_counts else 0,
            'min_tag_count': min(tag_counts.values()) if tag_counts else 0
        }
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': metadata
        }
    
    def export_tag_network_to_gexf(self, filepath: Union[str, Path]):
        """
        Экспортирует граф тегов в формате GEXF для Gephi или других инструментов.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Путь для сохранения файла
        """
        try:
            import networkx as nx
            
            # Получаем данные графа
            network_data = self.get_tag_network_data()
            
            # Создаем граф NetworkX
            G = nx.Graph()
            
            # Добавляем узлы
            for node in network_data['nodes']:
                G.add_node(node['id'], 
                          label=node['label'],
                          count=node['count'],
                          percentage=node['percentage'])
            
            # Добавляем ребра
            for edge in network_data['edges']:
                G.add_edge(edge['source'], edge['target'],
                          weight=edge['weight'],
                          strength=edge['strength'])
            
            # Сохраняем в GEXF
            nx.write_gexf(G, filepath)
            print(f"Граф тегов экспортирован в {filepath}")
            
        except ImportError:
            print("Для экспорта в GEXF требуется networkx")
            print("pip install networkx")
        except Exception as e:
            print(f"Ошибка при экспорте: {e}")
    
    def find_related_tags(self, tag: str, min_cooccurrence: int = 1) -> List[Tuple[str, int]]:
        """
        Находит теги, наиболее связанные с заданным тегом.
        
        Parameters:
        -----------
        tag : str
            Целевой тег
        min_cooccurrence : int
            Минимальное количество совместных вхождений
            
        Returns:
        --------
        List[Tuple[str, int]] - список пар (связанный тег, количество совместных вхождений)
        """
        if self._tag_cooccurrence is None:
            self._tag_cooccurrence = self._calculate_tag_cooccurrence()
        
        related_tags = []
        for (tag1, tag2), count in self._tag_cooccurrence.items():
            if count >= min_cooccurrence:
                if tag1 == tag and tag2 != tag:
                    related_tags.append((tag2, count))
                elif tag2 == tag and tag1 != tag:
                    related_tags.append((tag1, count))
        
        # Сортируем по силе связи
        related_tags.sort(key=lambda x: x[1], reverse=True)
        return related_tags
    
    def visualize_tag_communities(self, 
                                 resolution: float = 1.0,
                                 figsize: tuple = (14, 10)):
        """
        Визуализирует сообщества тегов на основе их совместной встречаемости.
        
        Parameters:
        -----------
        resolution : float
            Параметр разрешения для алгоритма обнаружения сообществ.
            Большие значения -> больше сообществ
        figsize : tuple
            Размер графика
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import community as community_louvain  # python-louvain
            
            # Создаем граф для анализа сообществ
            G = nx.Graph()
            
            # Добавляем узлы
            tag_counts = Counter()
            for sample in self.samples.values():
                for tag in sample.sample_tags:
                    tag_counts[tag] += 1
            
            for tag, count in tag_counts.items():
                G.add_node(tag, count=count)
            
            # Добавляем ребра с весом
            if self._tag_cooccurrence is None:
                self._tag_cooccurrence = self._calculate_tag_cooccurrence()
            
            for (tag1, tag2), weight in self._tag_cooccurrence.items():
                if tag1 != tag2 and weight > 0:
                    G.add_edge(tag1, tag2, weight=weight)
            
            # Обнаруживаем сообщества
            partition = community_louvain.best_partition(G, 
                                                        weight='weight',
                                                        resolution=resolution)
            
            # Создаем цветовую карту для сообществ
            num_communities = len(set(partition.values()))
            colors = plt.colormaps['tab20'](np.linspace(0, 1, max(num_communities, 1)))
            
            # Визуализируем
            plt.figure(figsize=figsize)
            
            # Расположение узлов
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Рисуем граф
            for community_id in set(partition.values()):
                # Узлы текущего сообщества
                nodes_in_community = [node for node in G.nodes() 
                                    if partition[node] == community_id]
                
                # Размер узлов пропорционален частоте тега
                node_sizes = [tag_counts[node] * 100 for node in nodes_in_community]
                
                # Рисуем узлы сообщества
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes_in_community,
                    node_size=node_sizes,
                    node_color=[colors[community_id]], # type: ignore
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=1
                )
            
            # Рисуем ребра
            nx.draw_networkx_edges(
                G, pos,
                alpha=0.3,
                edge_color='gray'
            )
            
            # Добавляем подписи
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_weight='bold'
            )
            
            plt.title(f"Сообщества тегов (найдено {num_communities} сообществ)", 
                     fontsize=14, pad=20)
            plt.axis('off')
            
            # Добавляем легенду с информацией о сообществах
            legend_text = "Сообщества тегов:\n"
            for community_id in sorted(set(partition.values())):
                tags_in_community = [tag for tag, cid in partition.items() 
                                   if cid == community_id]
                legend_text += f"\nСообщество {community_id}: {len(tags_in_community)} тегов"
                if tags_in_community:
                    legend_text += f"\n  {', '.join(tags_in_community[:3])}"
                    if len(tags_in_community) > 3:
                        legend_text += "..."
            
            plt.figtext(0.02, 0.02, legend_text, 
                       fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor="lightgray", 
                                alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            # Выводим детальную информацию о сообществах
            print("\n" + "="*60)
            print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О СООБЩЕСТВАХ:")
            print("="*60)
            
            for community_id in sorted(set(partition.values())):
                tags_in_community = [tag for tag, cid in partition.items() 
                                   if cid == community_id]
                
                print(f"\nСообщество {community_id}:")
                print(f"  Количество тегов: {len(tags_in_community)}")
                print(f"  Теги: {', '.join(sorted(tags_in_community))}")
                
                # Суммарная частота тегов в сообществе
                total_count = sum(tag_counts[tag] for tag in tags_in_community)
                print(f"  Общая частота: {total_count} упоминаний")
                
                # Самые частые теги в сообществе
                community_tag_counts = [(tag, tag_counts[tag]) 
                                       for tag in tags_in_community]
                community_tag_counts.sort(key=lambda x: x[1], reverse=True)
                
                if community_tag_counts:
                    print(f"  Самые частые теги:")
                    for tag, count in community_tag_counts[:3]:
                        percentage = (count / len(self.samples)) * 100
                        print(f"    {tag}: {count} ({percentage:.1f}%)")
            
            print("="*60)
            
        except ImportError as e:
            print(f"Для анализа сообществ требуется:")
            print("pip install python-louvain matplotlib networkx")
            print(f"Ошибка импорта: {e}")

# Утилитарные функции для работы с Sample и SampleCollection
def split_by_category(
    samples: Union[List[Sample], Dict[str, Sample], 'SampleCollection'],
    category: Literal['class', 'subclass'] = 'subclass',
    include_empty: bool = False
) -> Dict[str, List[Sample]]:
    """
    Разделяет коллекцию образцов по категориям (классам или подклассам).
    
    Parameters:
    -----------
    samples : Union[List[Sample], Dict[str, Sample], SampleCollection]
        Коллекция образцов. Может быть:
        - Список объектов Sample
        - Словарь {sample_id: Sample}
        - Объект SampleCollection
    category : Literal['class', 'subclass']
        По какой категории разделять:
        - 'class' -> по sample_class
        - 'subclass' -> по sample_subclass
    include_empty : bool
        Включать ли категории без образцов (пустые списки)
        
    Returns:
    --------
    Dict[str, List[Sample]]
        Словарь, где ключ - название категории (класса/подкласса),
        значение - список образцов этой категории.
    """
    # Преобразуем входные данные в список Sample
    if isinstance(samples, dict):
        sample_list = list(samples.values())
    elif isinstance(samples, list):
        sample_list = samples
    elif hasattr(samples, 'samples'):  # Это SampleCollection
        sample_list = list(samples.samples.values())
    else:
        raise TypeError(f"Неподдерживаемый тип данных: {type(samples)}")
    
    # Группируем образцы по выбранной категории
    result = {}
    
    for sample in sample_list:
        if category == 'class':
            key = sample.sample_class
        else:  # category == 'subclass'
            key = sample.sample_subclass
        
        # Если ключ None или пустая строка, обрабатываем специально
        if key is None or key == '':
            key = '_unknown' if include_empty else None
            if key is None:
                continue
        
        if key not in result:
            result[key] = []
        result[key].append(sample)
    
    # Добавляем пустые категории если нужно
    if include_empty:
        # Получаем все уникальные категории из всех образцов
        all_keys = set()
        for sample in sample_list:
            if category == 'class':
                key = sample.sample_class
            else:
                key = sample.sample_subclass
            all_keys.add(key if key else '_unknown')
        
        # Добавляем пустые списки для отсутствующих категорий
        for key in all_keys:
            if key not in result:
                result[key] = []
    
    return result

def split_and_get_stats(
    samples: Union[List[Sample], Dict[str, Sample], 'SampleCollection'],
    category: Literal['class', 'subclass'] = 'subclass'
) -> pd.DataFrame:
    """
    Разделяет коллекцию по категориям и возвращает статистику.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame со статистикой по категориям:
        - category: название категории
        - count: количество образцов
        - percentage: процент от общего количества
        - sample_ids: список ID образцов
    """
    grouped = split_by_category(samples, category, include_empty=True)
    
    total = sum(len(samples_list) for samples_list in grouped.values())
    
    stats_data = []
    for category_name, samples_list in sorted(grouped.items()):
        count = len(samples_list)
        percentage = (count / total * 100) if total > 0 else 0
        sample_ids = [s.sample_id for s in samples_list]
        
        stats_data.append({
            'category': category_name,
            'count': count,
            'percentage': round(percentage, 2),
            'sample_ids': sample_ids,
            'sample_count': len(sample_ids)  # для удобства фильтрации
        })
    
    return pd.DataFrame(stats_data)

def get_samples_by_category(
    samples: Union[List[Sample], Dict[str, Sample], 'SampleCollection'],
    category_value: str,
    category_type: Literal['class', 'subclass'] = 'subclass'
) -> List[Sample]:
    """
    Получает все образцы с определенным значением категории.
    
    Parameters:
    -----------
    category_value : str
        Значение категории для поиска
    category_type : Literal['class', 'subclass']
        Тип категории для поиска
        
    Returns:
    --------
    List[Sample]
        Список образцов с указанной категорией
    """
    if isinstance(samples, dict):
        sample_list = list(samples.values())
    elif isinstance(samples, list):
        sample_list = samples
    elif hasattr(samples, 'samples'):
        sample_list = list(samples.samples.values())
    else:
        raise TypeError(f"Неподдерживаемый тип данных: {type(samples)}")
    
    result = []
    for sample in sample_list:
        if category_type == 'class':
            if sample.sample_class == category_value:
                result.append(sample)
        else:
            if sample.sample_subclass == category_value:
                result.append(sample)
    
    return result

def visualize_category_distribution(
    samples: Union[List[Sample], Dict[str, Sample], 'SampleCollection'],
    category: Literal['class', 'subclass'] = 'subclass',
    top_n: Optional[int] = None,
    figsize: tuple = (12, 6)
):
    """
    Визуализирует распределение образцов по категориям.
    
    Requires:
    ---------
    import matplotlib.pyplot as plt
    import seaborn as sns
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        grouped = split_by_category(samples, category, include_empty=True)
        
        # Подготовка данных
        categories = []
        counts = []
        
        for cat_name, samples_list in grouped.items():
            if samples_list or cat_name != '_unknown':  # Показываем пустые кроме unknown
                categories.append(cat_name if cat_name != '_unknown' else 'Unknown')
                counts.append(len(samples_list))
        
        # Ограничиваем количество категорий если нужно
        if top_n and len(categories) > top_n:
            # Сортируем по количеству
            sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
            categories, counts = zip(*sorted_data[:top_n])
            
            # Добавляем "Остальные"
            other_count = sum(sorted_data[top_n:], key=lambda x: x[1]) # type: ignore
            if other_count > 0:
                categories = list(categories) + ['Other']
                counts = list(counts) + [other_count]
        
        # Создаем график
        plt.figure(figsize=figsize)
        
        # Столбчатая диаграмма
        ax1 = plt.subplot(1, 2, 1)
        bars = ax1.bar(categories, counts)
        ax1.set_xlabel(f'Sample {category}')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Distribution by {category}')
        ax1.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Круговая диаграмма
        ax2 = plt.subplot(1, 2, 2)
        wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%') # type: ignore
        ax2.set_title(f'Percentage by {category}')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Для визуализации требуется matplotlib и seaborn")
        print("Установите: pip install matplotlib seaborn")
        print("Данные изменились, нужно пересчитать дескрипторы!")
    