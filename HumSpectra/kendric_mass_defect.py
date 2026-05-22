import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Tuple
from collections import Counter
import re

import HumSpectra.mass_descriptors as md

class KendrickMassCalculator:
    """
    Калькулятор масс Кендрика с поддержкой произвольных молекулярных фрагментов
    """
    
    # Точные атомные массы (в Да)
    ATOMIC_MASSES = {
        'C': 12.0000000,   # Углерод-12 (стандарт)
        'H': 1.00782503223,
        'O': 15.99491461957,
        'N': 14.00307400443,
        'S': 31.9720711744,
        'P': 30.97376199842,  # Фосфор (опционально)
        'Na': 22.98976928,     # Натрий (для аддуктов)
        'K': 38.96370649,      # Калий
    }
    
    def __init__(self):
        pass
    
    @staticmethod
    def parse_formula(formula: str) -> Dict[str, int]:
        """
        Парсит химическую формулу в словарь {атом: количество}
        
        Примеры:
        'CH2' -> {'C': 1, 'H': 2}
        'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}
        '(CH2)2' -> {'C': 2, 'H': 4}
        """
        # Удаляем пробелы
        formula = formula.strip()
        
        # Обработка скобок (простая версия, без вложенных скобок)
        # Для полной поддержки скобок нужен более сложный парсер
        pattern = r'([A-Z][a-z]?)(\d*)|\(([^()]+)\)(\d*)'
        
        atom_counts = Counter()
        
        # Находим все элементы и группы
        for match in re.finditer(pattern, formula):
            if match.group(1):  # Обычный атом
                atom = match.group(1)
                count = int(match.group(2)) if match.group(2) else 1
                atom_counts[atom] += count
            elif match.group(3):  # Группа в скобках
                group_formula = match.group(3)
                multiplier = int(match.group(4)) if match.group(4) else 1
                # Рекурсивно парсим группу
                group_counts = KendrickMassCalculator.parse_formula(group_formula)
                for atom, cnt in group_counts.items():
                    atom_counts[atom] += cnt * multiplier
        
        return dict(atom_counts)
    
    @staticmethod
    def calculate_exact_mass(formula_dict: Dict[str, int]) -> float:
        """
        Рассчитывает точную массу фрагмента по словарю атомов
        """
        mass = 0.0
        for atom, count in formula_dict.items():
            if atom not in KendrickMassCalculator.ATOMIC_MASSES:
                raise ValueError(f"Неизвестный атом: {atom}. Доступны: {list(KendrickMassCalculator.ATOMIC_MASSES.keys())}")
            mass += KendrickMassCalculator.ATOMIC_MASSES[atom] * count
        return mass
    
    @staticmethod
    def get_kendrick_parameters(fragment_formula: str) -> Tuple[float, float, float]:
        """
        Получает параметры для пересчета в шкалу Кендрика
        
        Параметры:
        - fragment_formula: формула фрагмента (например, 'CH2', 'CO2', 'C6H6')
        
        Возвращает:
        - nominal_mass: номинальная масса фрагмента (целое число)
        - exact_mass: точная масса фрагмента
        - scale_factor: коэффициент масштабирования
        """
        atom_counts = KendrickMassCalculator.parse_formula(fragment_formula)
        
        # Номинальная масса = сумма номинальных масс атомов
        nominal_masses = {
            'C': 12, 'H': 1, 'O': 16, 'N': 14, 'S': 32,
            'P': 31, 'Na': 23, 'K': 39
        }
        
        nominal_mass = 0
        for atom, count in atom_counts.items():
            nominal_mass += nominal_masses.get(atom, 0) * count
        
        exact_mass = KendrickMassCalculator.calculate_exact_mass(atom_counts)
        scale_factor = nominal_mass / exact_mass
        
        return nominal_mass, exact_mass, scale_factor
    
    @staticmethod
    def calculate_kendrick_mass(exact_mass: float, fragment_formula: str = 'CH2') -> Tuple[float, float, float]:
        """
        Рассчитывает массу Кендрика и дефект
        
        Параметры:
        - exact_mass: точная масса молекулы
        - fragment_formula: формула фрагмента (по умолчанию 'CH2')
        
        Возвращает:
        - kendrick_mass: масса Кендрика
        - nominal_kendrick_mass: номинальная масса Кендрика
        - kmd: дефект массы Кендрика
        """
        _, _, scale_factor = KendrickMassCalculator.get_kendrick_parameters(fragment_formula)
        
        kendrick_mass = exact_mass * scale_factor
        nominal_kendrick_mass = np.floor(kendrick_mass)
        kmd = nominal_kendrick_mass - kendrick_mass
        
        return kendrick_mass, nominal_kendrick_mass, kmd


def kendrick(self, fragment_formula: str = 'CH2', method: str = 'floor', 
             adjust_negative: bool = True, inplace: bool = True) -> Union[pd.DataFrame, None]:
    """
    Рассчитывает массу Кендрика и дефект массы для произвольного фрагмента
    
    Параметры
    ----------
    fragment_formula : str
        Формула фрагмента (например, 'CH2', 'CO2', 'H2', 'C6H6', '(CH2)2O')
    method : str
        Метод округления: 'floor' (по умолчанию) или 'round'
    adjust_negative : bool
        Корректировать ли отрицательные KMD (прибавлять 1)
    inplace : bool
        Если True, добавляет колонки в существующий DataFrame
        Если False, возвращает новый DataFrame с колонками
        
    Возвращает
    ----------
    Spectrum or None
    """
    import pandas as pd
    import numpy as np
    
    # Проверяем наличие колонки с массами
    if 'calc_mass' not in self:
        self = md.calc_mass(self)
    
    # Получаем параметры для выбранного фрагмента
    nominal_frag, exact_frag, scale_factor = KendrickMassCalculator.get_kendrick_parameters(fragment_formula)
    
    # Создаем копию если нужно
    result = self if inplace else self.copy()
    
    # Рассчитываем массу Кендрика
    result['Ke'] = result['calc_mass'] * scale_factor
    
    # Рассчитываем номинальную массу Кендрика и KMD
    if method == 'floor':
        result['Nominal_Ke'] = np.floor(result['Ke'].to_numpy().astype('float'))
        result['KMD'] = result['Nominal_Ke'] - result['Ke']
    elif method == 'round':
        result['Nominal_Ke'] = np.round(result['Ke'].to_numpy().astype('float'))
        result['KMD'] = result['Nominal_Ke'] - result['Ke']
    else:
        raise ValueError("method должен быть 'floor' или 'round'")
    
    # Корректировка отрицательных KMD (опционально)
    if adjust_negative and method == 'floor':
        # Традиционный подход: KMD всегда в [0, 1)
        result.loc[result['KMD'] <= 0, 'KMD'] = result.loc[result['KMD'] <= 0, 'KMD'] + 1
        result.loc[result['KMD'] >= 1, 'KMD'] = result.loc[result['KMD'] >= 1, 'KMD'] - 1
    
    # Добавляем информацию о фрагменте в атрибуты
    result.attrs['kendrick_fragment'] = fragment_formula
    result.attrs['kendrick_method'] = method
    result.attrs['kendrick_scale_factor'] = scale_factor
    
    if not inplace:
        return result
    else:
        return self


# Пример использования с вашим классом Spectrum
def add_kendrick_methods(Spectrum):
    """Добавляет метод kendrick в класс Spectrum"""
    Spectrum.kendrick = kendrick
    return Spectrum