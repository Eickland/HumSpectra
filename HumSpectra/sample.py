import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Any, List
import json
import hashlib
from datetime import datetime

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
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.samples: Dict[str, Sample] = {}
        self.cache_dir = cache_dir or Path("./sample_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._tag_index: Dict[str, Set[str]] = {}
        
    def add_sample(self, sample: Sample):
        """Добавляет образец в коллекцию"""
        if sample._data_hash is None:
            sample._data_hash = sample.calculate_hash()
        self.samples[sample.sample_id] = sample
        self._update_tag_index(sample)
    
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
        for data in samples_data:
            # Используем фабричный метод
            sample = Sample.from_dict(data)
            self.samples[sample.sample_id] = sample
            
        # Методы фильтрации
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

# Пример использования
if __name__ == "__main__":
    # Создание коллекции
    collection = SampleCollection(cache_dir=Path("./data_cache"))
    
    # Загрузка данных (пример)
    sample_data = None
    uv_data = None # УФ-видимый спектр
    
    # Создание образца
    sample = collection.create_sample_from_data(
        sample_id="sample_001",
        fluorescence_eem=sample_data,
        uv_vis_absorption=uv_data,
        measurement_params={
            "excitation_wavelengths": np.linspace(240, 450, 100),
            "emission_wavelengths": np.linspace(250, 600, 100),
            "integration_time": 0.1
        },
        sample_type="unknown",
        measurement_date=datetime.now(),
        comments="Первый тестовый образец"
    )
    
    
    # Сохранение образца
    collection.save_sample("sample_001")
    
    # Позже можно загрузить
    loaded_sample = collection.load_sample("sample_001")
    
    # Проверка, изменились ли данные
    current_hash = loaded_sample.calculate_hash()
    if current_hash != loaded_sample._data_hash:
        print("Данные изменились, нужно пересчитать дескрипторы!")
    