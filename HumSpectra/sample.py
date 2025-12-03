import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import hashlib
from datetime import datetime

@dataclass
class Sample:
    """Класс для хранения всех данных образца"""
    sample_id: str
    
    # Основные спектральные данные
    fluorescence_eem: Optional[pd.DataFrame] = None
    uv_vis_absorption: Optional[pd.DataFrame] = None
    
    #Дополнительные данные
    org_carbon: Optional[float] = None
    pH: Optional[float] = None
    Eh: Optional[float] = None
    
    #Координаты
    latitude: Optional[float] = None
    lonitude: Optional[float] = None
    
    # Параметры измерений
    measurement_params: Dict[str, Any] = field(default_factory=dict)
    
    # Рассчитанные дескрипторы
    descriptors: Dict[str, float] = field(default_factory=dict)
    
    # Метаданные
    file_path: Optional[Path] = None
    measurement_date: Optional[str] = None
    sample_type: str = ""
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
            'fluorescence_eem': self.fluorescence_eem,
            'uv_vis_absorption': self.uv_vis_absorption,
            'org_carbon': self.org_carbon,
            'pH': self.pH,
            'Eh': self.Eh,
            'latitude': self.latitude,
            'lonitude': self.lonitude,
            'measurement_params': self.measurement_params,
            'descriptors': self.descriptors,
            'file_path': str(self.file_path) if self.file_path else None,
            'measurement_date': self.measurement_date,
            'sample_type': self.sample_type,
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
            fluorescence_eem=data.get('fluorescence_eem'),
            uv_vis_absorption=data.get('uv_vis_absorption'),
            org_carbon=data.get('org_carbon'),
            pH=data.get('pH'),
            Eh=data.get('Eh'),
            latitude=data.get('latitude'),
            lonitude=data.get('lonitude'),
            measurement_params=data.get('measurement_params', {}),
            descriptors=data.get('descriptors', {}),
            file_path=file_path,
            measurement_date=data.get('measurement_date'),
            sample_type=data.get('sample_type', ''),
            comments=data.get('comments', ''),
            _data_hash=data.get('_data_hash')
        )

class SampleCollection:
    """Коллекция образцов с возможностью сохранения/загрузки"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.samples: Dict[str, Sample] = {}
        self.cache_dir = cache_dir or Path("./sample_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def add_sample(self, sample: Sample):
        """Добавляет образец в коллекцию"""
        if sample._data_hash is None:
            sample._data_hash = sample.calculate_hash()
        self.samples[sample.sample_id] = sample
    
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
    