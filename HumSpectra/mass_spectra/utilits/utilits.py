import copy
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from functools import wraps
from frozendict import frozendict
from collections import defaultdict
from typing import Sequence, Any, Dict, Tuple, List, Optional

import HumSpectra.utilits as ut
from ..brutto import brutto as ms_brutto
from ..calc_process import calc_process as ms_calc

def convert_mass_spectra_batch(source_dir, output_base, program_location=
                               "C:\\Users\\mnbv2\\AppData\\Local\\Apps\\ProteoWizard 3.0.21229.9668f52 64-bit"):
    """
    Конвертирует масс-спектрометрические данные из папки с несколькими подпапками (.d)
    
    Args:
        source_dir (str): Путь к папке, содержащей подпапки с исходными файлами
        output_base (str): Путь для сохранения результатов
        program_location (str): Путь к программе Proteo Wizard, пример:
        "C:\\Users\\mnbv2\\AppData\\Local\\Apps\\ProteoWizard 3.0.21229.9668f52 64-bit"
    """
    
    # PowerShell скрипт для пакетной обработки (рекурсивный поиск .d папок)
    powershell_script = f'''
    $msconvertPath = "{program_location}\\msconvert.exe"
    $sourceDir = "{source_dir}"
    $outputBase = "{output_base}"
    $dFolders = Get-ChildItem -Path $sourceDir -Recurse -Directory | Where-Object {{ $_.Name -like "*.d" }}
    Write-Host "Найдено .d папок: $($dFolders.Count)"
    foreach ($folder in $dFolders) {{
        $inputPath = $folder.FullName
        $relativePath = $folder.FullName.Replace($sourceDir, "").Trim("\\")
        $outputSubDir = Join-Path $outputBase $relativePath
        $outputSubDir = $outputSubDir -replace '\\.d$', ''
        New-Item -ItemType Directory -Path $outputSubDir -Force | Out-Null
        Write-Host "Конвертируем: $($folder.Name) -> $outputSubDir"
        msconvertPath "$inputPath" -o "$outputSubDir" --mzML --filter "zeroSamples removeExtra" --verbose
    }}
    '''
    
    try:
        print(f"🔍 Поиск .d папок в: {source_dir}")
        print(f"💾 Сохранение в: {output_base}")
        print("⏳ Начинаем пакетную конвертацию...")
        
        # Запускаем PowerShell скрипт
        result = subprocess.run([
            'powershell', '-Command', powershell_script
        ], capture_output=True, text=True, check=True, encoding='utf8')
        
        print("✅ Пакетная конвертация завершена успешно!")
        print("📋 Вывод PowerShell:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Предупреждения:")
            print(result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при выполнении PowerShell скрипта: {e}")
        print(f"🔴 Stderr: {e.stderr}")

def convert_single_folder(single_folder, output_dir, program_location):
    """
    Конвертирует масс-спектрометрические данные из одной папки (.d)
    """
    
    folder_path = Path(single_folder)
    if not folder_path.name.endswith('.d'):
        print(f"⚠️ Внимание: выбранная папка '{folder_path.name}' не имеет расширения .d")
        response = input("Продолжить конвертацию? (y/n): ").lower()
        if response != 'y':
            print("❌ Конвертация отменена")
            return
    
    # Экранируем пути с пробелами двойными кавычками
    powershell_script = f'''
    $msconvertPath = "{program_location}\\msconvert.exe"
    $inputPath = "{single_folder}"
    $outputDir = "{output_dir}"
    
    # Создаем целевую папку
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    
    Write-Host "Конвертируем: $(Split-Path $inputPath -Leaf) -> $outputDir"
    
    # Запускаем конвертацию, используя полный путь к msconvert.exe
    & $msconvertPath "$inputPath" -o "$outputDir" --mzML 
    
    Write-Host "Конвертация завершена!"
    '''
    
    try:
        print(f"📁 Конвертируем папку: {folder_path.name}")
        print(f"💾 Сохранение в: {output_dir}")
        print("⏳ Начинаем конвертацию...")
        
        result = subprocess.run([
            'powershell', '-Command', powershell_script
        ], capture_output=True, text=True, check=True)
        
        print("✅ Конвертация завершена успешно!")
        print("📋 Вывод PowerShell:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Предупреждения:")
            print(result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при выполнении PowerShell скрипта: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"🔴 Stderr: {e.stderr}")

def read_mass_list(path: str,
                map_columns: dict|None = {"m/z":"mass","I":"intensity"},
                custom_columns_name: bool = False,
                sep: str|None = "\t",
                **kwargs) -> pd.DataFrame:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.csv").
    :param sep: разделитель в строчном виде (example: ",").
    :return: DataFrame: 
    """
    if sep is None:
        sep = ut.check_sep(path)
    try:
        data = pd.read_csv(path, sep=sep, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Файл пуст: {path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")
    
    if custom_columns_name:
        data.rename(columns=map_columns,inplace=True)

        data = data[["mass","intensity"]]

    data.dropna(inplace=True, axis=1)
    data = data.astype("float64")
    name = ut.extract_name_from_path(path)
    data.attrs['name'] = name

    return data

def find_elements(self) -> Sequence[str]:
    """ 
    Find elements from columns of mass spectrum table.

    For example, column 'C' will be recognised as carbon 12C, column 'C_13" as 13C

    Returns
    -------
    list
    """

    main_elems = ms_brutto.elements_table()['element'].values
    all_elems = ms_brutto.elements_table()['element_isotop'].values

    elems = []
    for col in self.columns:
        if col in main_elems:
            elems.append(col)
        elif col in all_elems:
            elems.append(col)

    if len(elems) == 0:
        elems = ""

    self.attrs['elems'] = elems

    return elems

def _copy(func):
    """
    Decorator for deep copy pd.DataFrame before apllying methods
    
    Parameters
    ----------
    func: method
        function for decoarate
    
    Return
    ------
    function with deepcopyed pd.DataFrame
    """

    @wraps(func)
    def wrapper(dataframe, *args, **kwargs):
        # Создаем глубокую копию DataFrame
        dataframe_copy = copy.deepcopy(dataframe)
        
        # Вызываем оригинальную функцию с копией
        result = func(dataframe_copy, *args, **kwargs)
        
        return result
    
    return wrapper

@_copy
def drop_unassigned(self) -> "pd.DataFrame":
    """
    Drop unassigned by brutto rows

    Return
    ------
    pd.DataFrame

    Caution
    -------
    Danger of lose data - with these operation we exclude data that can be usefull
    """

    if "assign" not in self:
        raise Exception("Spectrum is not assigned")

    self = self.loc[self["assign"] == True].reset_index(drop=True)
    self.attrs['drop_unassigned'] = True

    return self

@_copy
def merge_duplicates(self) -> "pd.DataFrame":
    """
    merge duplicataes with the same calculated mass with sum intensity

    Return
    ------
    pd.DataFrame
    """
    if 'calc_mass' not in self.columns:
        self = ms_calc.calc_mass(self)

    cols = {col: ('sum' if col=='intensity' else 'max') for col in self.columns}
    self = self.groupby(['calc_mass'],as_index = False).agg(cols)
    return self

@_copy
def merge_isotopes(self) -> "pd.DataFrame":
    """
    Merge isotopes.

    For example if specrum list have 'C' and 'C_13' they will be summed in 'C' column.

    Return
    ------
    pd.DataFrame

    Caution
    -------
    Danger of lose data - with these operation we exclude data that can be usefull       
    """

    elems = find_elements(self)
    for el in elems:
        res = el.split('_')
        if len(res) == 2:
            if res[0] not in self:
                self[res[0]] = 0
            self[res[0]] = self[res[0]] + self[el]
            self = self.drop(columns=[el])
    
    self.attrs['merge_isotopes'] = True

    return self

def _freeze(func):
    """
    freeze dict in func
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped

def _process_elems(elems: Any) -> Dict[str, Tuple[int, int]]:
    """
    Преобразует входные данные в рабочий dict
    """

    if elems is None:
        return {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}
    elif isinstance(elems, frozendict):
        return dict(elems)
    elif isinstance(elems, dict):
        return elems
    else:
        raise TypeError(f"Expected dict, frozendict or None, got {type(elems)}")

def _merge_isotopes(gdf: pd.DataFrame) -> pd.DataFrame:
    """
    All isotopes will be merged and title as main.

    Return
    ------
    pandas Dataframe    
    """

    for el in gdf.columns:
        res = el.split('_')
        if len(res) == 2:
            if res[0] not in gdf:
                gdf[res[0]] = 0
            gdf[res[0]] = gdf[res[0]] + gdf[el]
            gdf = gdf.drop(columns=[el]) 

    return gdf

def merge_mass_spectra(
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
