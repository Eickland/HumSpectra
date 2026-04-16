from pyopenms import * # type: ignore
from pathlib import Path
import pandas as pd
import HumSpectra.mass_spectra as ms
import HumSpectra.mass_visualizer as msv
import HumSpectra.utilits as ut
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
from sklearn.mixture import GaussianMixture
from numba import njit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, CrosshairTool
from bokeh.io import output_notebook, show

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

@njit
def fast_diff_window(mass, max_diff=1000.0):
    """
    Вычисляет разности масс в пределах окна max_diff.
    Это превращает O(n^2) почти в O(n * k), где k - среднее число пиков в окне.
    """
    # Примерный размер массива для результатов (можно подстроить)
    # Для 80к пиков лучше использовать динамический список или большой буфер
    res = []
    n = len(mass)
    for i in range(n):
        for j in range(i + 1, n):
            d = mass[j] - mass[i]
            if d > max_diff:
                break
            res.append(round(d, 6))
    return np.array(res)

def calc_by_brutto_new(self, max_diff=1000.0) -> "pd.DataFrame":
    name = self.attrs.get('name', 'Unknown')
    mass = np.sort(drop_unassigned(self)['calc_mass'].to_numpy())
    
    # Вычисляем разности (Numba сделает это молниеносно)
    diffs = fast_diff_window(mass, max_diff)
    
    # Подсчет уникальных
    unique, counts = np.unique(diffs, return_counts=True)
    
    diff_spec = pd.DataFrame({
        'mass': unique,
        'count': counts,
        'intensity': counts / len(mass)
    })
    
    diff_spec.attrs['name'] = name
    return diff_spec

def interactive_spectrum_bokeh(spec: pd.DataFrame,
                                xlim=(None, None),
                                ylim=(None, None),
                                normalize: bool = False,
                                max_points: int = 50000,
                                title=None,
                                height=600,
                                width=1200):
    """
    Интерактивный спектр с использованием Bokeh (оптимизирован для больших данных)
    """
    df = spec.copy()
    mass = df['mass'].to_numpy()
    intensity = df['intensity'].to_numpy()
    
    # Фильтр по xlim
    if xlim[0] is not None and xlim[1] is not None:
        mask = (mass >= xlim[0]) & (mass <= xlim[1])
        mass = mass[mask]
        intensity = intensity[mask]
    
    # Нормализация
    if normalize and len(intensity) > 0:
        intensity = (intensity / intensity.max()) * 100
    
    # ДАУНСЕМПЛИНГ для больших данных
    if len(mass) > max_points:
        step = len(mass) // max_points
        mass = mass[::step]
        intensity = intensity[::step]
        print(f"Downsampled: {len(mass)} points (from {len(df)})")
    
    # Создаем источник данных
    source = ColumnDataSource(data={
        'x': mass,
        'y': intensity,
        'mz': mass,
        'int': intensity
    })
    
    # Создаем фигуру
    p = figure(title=title or f'Spectrum: {len(mass)} peaks',
               height=height, 
               width=width,
               x_axis_label='m/z, Da',
               y_axis_label='Relative Intensity (%)' if normalize else 'Intensity',
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom")
    
    # Рисуем пики как сегменты (более эффективно чем vlines)
    # Создаем данные для сегментов
    x_segments = []
    y_segments = []
    for x, y in zip(mass, intensity):
        x_segments.append(x)
        x_segments.append(x)
        x_segments.append(None)
        y_segments.append(0)
        y_segments.append(y)
        y_segments.append(None)
    
    p.segment(x0=mass, y0=0, x1=mass, y1=intensity,
              color='navy', line_width=1.5, alpha=0.7)
    
    # Добавляем точки на вершинах для hover
    p.circle(x=mass, y=intensity, color='red', alpha=0.3,radius=0.0002)
    
    # Hover tool
    hover = HoverTool(tooltips=[
        ("m/z", "@mz{0.00}"),
        ("Intensity", "@int{0.00}")
    ])
    p.add_tools(hover)
    p.add_tools(CrosshairTool())
    
    # Установка лимитов
    if xlim[0] is not None and xlim[1] is not None:
        p.x_range.start = xlim[0] # type: ignore
        p.x_range.end = xlim[1] # type: ignore
    
    if ylim[0] is not None:
        p.y_range.start = ylim[0] # type: ignore
    if ylim[1] is not None:
        p.y_range.end = ylim[1] # type: ignore
    
    return p
def gmm_noise_filter(df, intensity_col='intensity'):
    """
    Автоматическая очистка шума в масс-спектре на основе GMM.
    Алгоритм из статьи Potemkin et al., Anal. Chem. 2024.
    """
    # 1. Работаем с десятичным логарифмом интенсивности [cite: 796, 892]
    intensities = df[intensity_col].to_numpy()
    intensities = intensities[intensities > 0]
    log_ints = np.log10(intensities).reshape(-1, 1)
    
    # 2. Поиск оптимального количества компонент K (от 1 до 15) [cite: 868]
    k_range = range(1, 16)
    bics = []
    models = []
    
    for k in k_range:
        # Инициализация через k-means++ для стабильности [cite: 855]
        gmm = GaussianMixture(n_components=k, random_state=42, init_params='k-means++')
        gmm.fit(log_ints)
        bics.append(gmm.bic(log_ints)) # Информационный критерий Байеса [cite: 858]
        models.append(gmm)
    
    # 3. Определение границы "крутого спуска" на графике BIC [cite: 867, 884]
    bics = np.array(bics)
    diffs = np.diff(bics)
    # Нормализованные абсолютные значения производной [cite: 879, 880]
    x = np.abs(diffs) / np.max(np.abs(diffs))
    
    # Вычисление отклонения дельта [cite: 881, 882]
    deltas = []
    for i in range(len(x) - 1):
        mean_subsequent = np.mean(x[i+1:])
        deltas.append(np.abs(x[i] - mean_subsequent))
    
    # Поиск точки после спуска (в обратном порядке, порог > 0.1) [cite: 884]
    idx = 0
    for i in range(len(deltas)-1, -1, -1):
        if deltas[i] > 0.1:
            idx = i
            break
            
    # Дополнительное уточнение вперед (порог > 0.04) [cite: 885]
    for k in range(idx, len(deltas)):
        if deltas[k] > 0.04:
            idx = k
        else:
            break
            
    # Выбираем модель с оптимальным K
    best_k = idx + 2 # Коррекция индекса
    best_model = models[best_k - 1]
    
    # 4. Расчет порога интенсивности [cite: 808, 895]
    # Находим две первые гауссианы (по возрастанию средних)
    means = best_model.means_.flatten()
    sorted_indices = np.argsort(means)
    idx1, idx2 = sorted_indices[0], sorted_indices[1]
    
    m1, s1, w1 = means[idx1], np.sqrt(best_model.covariances_[idx1][0][0]), best_model.weights_[idx1]
    m2, s2, w2 = means[idx2], np.sqrt(best_model.covariances_[idx2][0][0]), best_model.weights_[idx2]
    
    # Нахождение точки пересечения двух первых гауссиан [cite: 554, 808]
    # Решаем квадратное уравнение Ax^2 + Bx + C = 0 для ln(f1) = ln(f2)
    A = 1/(2*s2**2) - 1/(2*s1**2)
    B = m1/(s1**2) - m2/(s2**2)
    C = m2**2/(2*s2**2) - m1**2/(2*s1**2) + np.log((w1*s2)/(w2*s1))
    
    roots = np.roots([A, B, C])
    # Выбираем корень, лежащий между средними значениями
    threshold_log = [r for r in roots if m1 < r < m2]
    
    if not threshold_log:
        threshold_log = [m1 + 3*s1] # Альтернативный метод, если пересечение не найдено [cite: 571]
    
    threshold_abs = 10**threshold_log[0]*30 # Перевод из логарифма в абсолютные единицы [cite: 555]
    
    # 5. Фильтрация [cite: 809, 556]
    filtered_df = df[df[intensity_col] >= threshold_abs].copy()
    
    return filtered_df, threshold_abs

def spectrum(spec: 'pd.DataFrame',
             xlim = (None, None),
             ylim = (None, None),
             color: str = 'black',
             ax = None,
             title= None,
             normalize: bool = False,
             peak_width: float = 1.0,
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
    
    df = spec
    
    # Filter data based on xlim
    mass = df['mass'].to_numpy()
    intensity = df['intensity'].to_numpy()
    
    # Auto-determine xlim if None
    x_min = xlim[0] if xlim[0] is not None else mass.min()
    x_max = xlim[1] if xlim[1] is not None else mass.max()
    xlim = (x_min, x_max)
    
    # Apply xlim filter
    mask = (mass >= xlim[0]) & (mass <= xlim[1])
    mass = mass[mask]
    intensity = intensity[mask]
    
    # Normalize if requested
    if normalize and intensity.max() > 0:
        intensity = intensity / intensity.max() * 100
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Direct vlines approach (more efficient and clear)
    ax.vlines(mass, ymin=0, ymax=intensity, 
              color=color, linewidth=peak_width, **kwargs)
    '''
    # Optional: add baseline
    ax.plot([xlim[0], xlim[1]], [0, 0], 
            color=color, linewidth=0.5, alpha=0.5)
    
    # Optional: smoothed baseline curve
    if len(mass) > 1:
        ax.plot(mass, intensity, color=color, 
                linewidth=0.5, alpha=0.3)
    '''
    # Set limits
    ax.set_xlim(xlim)
    # Безопасная проверка ylim
    if ylim[0] is None or ylim[1] is None:
        y_max = intensity.max() if len(intensity) > 0 else 1
        y_padding = y_max * 0.15
        ax.set_ylim((-y_padding * 0.1, y_max + y_padding))
    else:
        ax.set_ylim(ylim)
    
    # Labels
    ax.set_xlabel('m/z, Da', fontsize=12)
    ax.set_ylabel('Relative Intensity (%)' if normalize else 'Intensity', 
                  fontsize=12)
    
    # Title
    if title is None:
        if hasattr(spec, 'attrs') and 'name' in spec.attrs:
            title = f'{spec.attrs["name"]}, {len(df)} peaks'
        else:
            title = f'{len(df)} peaks'
    
    if title:
        ax.set_title(title, fontsize=14, pad=12)
    
    # Improve appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Optional grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    return ax

def convert_mass_spectra_batch(source_dir, output_base):
    """
    Конвертирует масс-спектрометрические данные из папки с несколькими подпапками (.d)
    
    Args:
        source_dir (str): Путь к папке, содержащей подпапки с исходными файлами
        output_base (str): Путь для сохранения результатов
    """
    
    # PowerShell скрипт для пакетной обработки (рекурсивный поиск .d папок)
    powershell_script = f'''
    Set-Location "C:\\Users\\mnbv2\\AppData\\Local\\Apps\\ProteoWizard 3.0.21229.9668f52 64-bit"
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
        .\\msconvert.exe "$inputPath" -o "$outputSubDir" --mzML --filter "zeroSamples removeExtra" --verbose
    }}
    '''
    
    try:
        print(f"🔍 Поиск .d папок в: {source_dir}")
        print(f"💾 Сохранение в: {output_base}")
        print("⏳ Начинаем пакетную конвертацию...")
        
        # Запускаем PowerShell скрипт
        result = subprocess.run([
            'powershell', '-Command', powershell_script
        ], capture_output=True, text=True, check=True)
        
        print("✅ Пакетная конвертация завершена успешно!")
        print("📋 Вывод PowerShell:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Предупреждения:")
            print(result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при выполнении PowerShell скрипта: {e}")
        print(f"🔴 Stderr: {e.stderr}")

def convert_single_folder(single_folder, output_dir):
    """
    Конвертирует масс-спектрометрические данные из одной папки (.d)
    
    Args:
        single_folder (str): Путь к конкретной папке .d с исходными файлами
        output_dir (str): Путь для сохранения результатов
    """
    
    # Проверяем, что это действительно .d папка
    folder_path = Path(single_folder)
    if not folder_path.name.endswith('.d'):
        print(f"⚠️ Внимание: выбранная папка '{folder_path.name}' не имеет расширения .d")
        response = input("Продолжить конвертацию? (y/n): ").lower()
        if response != 'y':
            print("❌ Конвертация отменена")
            return
    
    # PowerShell скрипт для одиночной папки
    powershell_script = f'''
    Set-Location "C:\\Users\\mnbv2\\AppData\\Local\\Apps\\ProteoWizard 3.0.21229.9668f52 64-bit"
    $inputPath = "{single_folder}"
    $outputDir = "{output_dir}"
    
    # Создаем целевую папку
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    
    Write-Host "Конвертируем: $(Split-Path $inputPath -Leaf) -> $outputDir"
    
    # Запускаем конвертацию для одной папки
    .\\msconvert.exe "$inputPath" -o "$outputDir" --mzML 
    
    Write-Host "Конвертация завершена!"
    '''
    
    try:
        print(f"📁 Конвертируем папку: {folder_path.name}")
        print(f"💾 Сохранение в: {output_dir}")
        print("⏳ Начинаем конвертацию...")
        
        # Запускаем PowerShell скрипт
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
        print(f"🔴 Stderr: {e.stderr}")

def extract_mass_list_percentile(mzml_file, ms_level=1, rt_range=None, 
                               low_percentile=99.9,
                               high_percentile=99.95):
    """
    Порог как процентиль от распределения интенсивностей
    """
    exp = MSExperiment()
    MzMLFile().load(mzml_file, exp)
    
    # Сбор всех интенсивностей
    all_intensities = []
    for spectrum in exp:
        if spectrum.getMSLevel() == ms_level:
            if rt_range and not (rt_range[0] <= spectrum.getRT() <= rt_range[1]):
                continue
            mz, intensities = spectrum.get_peaks()
            all_intensities.extend(intensities)
    
    if not all_intensities:
        return pd.DataFrame()
    
    # Порог по процентилю
    low_percentile_threshold = np.percentile(all_intensities, low_percentile)
    high_percentile_threshold = np.percentile(all_intensities, high_percentile)
    
    print(f"{low_percentile}-й процентиль интенсивности: {low_percentile_threshold:.2f}")
    print(f"{high_percentile}-й процентиль интенсивности: {high_percentile_threshold:.2f}")
    print(f"Всего точек до фильтрации: {len(all_intensities)}")
    
    # Извлечение с порогом по процентилю
    all_mz = []
    all_intensities_filtered = []
    
    for spectrum in exp:
        if spectrum.getMSLevel() != ms_level:
            continue
            
        rt = spectrum.getRT()
        if rt_range and not (rt_range[0] <= rt <= rt_range[1]):
            continue
            
        mz, intensities = spectrum.get_peaks()
        
        mask = (intensities >= low_percentile_threshold)
        filtered_mz = mz[mask]
        filtered_intensities = intensities[mask]
        
        all_mz.extend(filtered_mz)
        all_intensities_filtered.extend(filtered_intensities)
    
    print(f"Точек после фильтрации: {len(all_mz)}")
    
    return pd.DataFrame({
        'mass': all_mz,
        'intensity': all_intensities_filtered,
    })
 
def extract_mass_list_optimal_percentile(mzml_file, ms_level=1, rt_range=None,
                                         percentiles_to_try=None,
                                         mass_bin_width=10,
                                         smooth_sigma=2,
                                         remove_saturated=True,
                                         upper_percentile=99.999,
                                         upper_threshold=None,
                                         plot=False):
    """
    Автоматически подбирает оптимальный нижний процентиль интенсивности
    для фильтрации шума в масс-спектре и опционально удаляет зашкаливающие пики.

    Параметры:
    ----------
    mzml_file : str
        Путь к mzML файлу
    ms_level : int
        Уровень MS (1 для MS1)
    rt_range : tuple, optional
        Диапазон времени удерживания (мин, макс) в секундах
    percentiles_to_try : array-like, optional
        Список процентилей для подбора нижнего порога
    mass_bin_width : float
        Ширина бина для гистограммы масс
    smooth_sigma : float
        Параметр сглаживания гистограммы
    remove_saturated : bool
        Удалять ли зашкаливающие пики (верхний порог)
    upper_percentile : float
        Верхний процентиль интенсивности для удаления насыщенных пиков
        (используется, если upper_threshold не задан)
    upper_threshold : float, optional
        Абсолютный порог интенсивности для насыщенных пиков (имеет приоритет)

    Возвращает:
    -----------
    df : pd.DataFrame
        Таблица с колонками 'mass' и 'intensity'
    optimal_percentile : float or None
        Оптимальный нижний процентиль
    """
    if percentiles_to_try is None:
        percentiles_to_try = np.arange(96.0, 99.8, 0.05)

    exp = MSExperiment()
    MzMLFile().load(mzml_file, exp)

    # Первый проход: сбор всех интенсивностей и масс
    all_intensities = []
    all_masses = []

    for spectrum in exp:
        if spectrum.getMSLevel() == ms_level:
            if rt_range and not (rt_range[0] <= spectrum.getRT() <= rt_range[1]):
                continue
            mz, intensities = spectrum.get_peaks()
            all_intensities.extend(intensities)
            all_masses.extend(mz)

    if not all_intensities:
        return pd.DataFrame(), None

    all_intensities = np.array(all_intensities)
    all_masses = np.array(all_masses)

    # Определение верхнего порога для насыщенных пиков (если нужно)
    upper_cutoff = None
    if remove_saturated:
        if upper_threshold is not None:
            upper_cutoff = upper_threshold
        else:
            upper_cutoff = np.percentile(all_intensities, upper_percentile)
        print(f"Верхний порог интенсивности (насыщение): {upper_cutoff:.2f}")

    # Для подбора нижнего порога можно предварительно исключить насыщенные пики,
    # чтобы они не влияли на статистику (опционально)
    if upper_cutoff is not None:
        mask_not_saturated = all_intensities <= upper_cutoff
        intensities_for_percentiles = all_intensities[mask_not_saturated]
        print(f"Пиков после удаления насыщенных: {np.sum(mask_not_saturated)} из {len(all_intensities)}")
    else:
        intensities_for_percentiles = all_intensities

    best_score = -np.inf
    best_percentile = None
    best_distribution = None

    print("Поиск оптимального нижнего процентиля...")

    for percentile in percentiles_to_try:
        threshold = np.percentile(intensities_for_percentiles, percentile)

        # Применяем нижний порог ко всем данным (включая насыщенные, они потом отсекутся отдельно)
        mask_lower = all_intensities >= threshold
        filtered_masses = all_masses[mask_lower]

        # Если также убираем насыщенные, применяем верхний порог
        if upper_cutoff is not None:
            mask_upper = all_intensities <= upper_cutoff
            final_mask = mask_lower & mask_upper
            filtered_masses = all_masses[final_mask]

        if len(filtered_masses) < 10:
            continue

        # Построение распределения масс
        max_mass = min(1500, np.percentile(filtered_masses, 99))
        bins = np.arange(0, max_mass + mass_bin_width, mass_bin_width)
        hist, bin_edges = np.histogram(filtered_masses, bins=bins)

        # Сглаживание
        smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)

        # Оценка качества распределения
        score = evaluate_mass_distribution(smoothed_hist, bin_edges[:-1], mass_bin_width)

        if score > best_score:
            best_score = score
            best_percentile = percentile
            best_distribution = (bin_edges[:-1], smoothed_hist, hist)

    if best_percentile is None:
        print("Не удалось найти подходящий процентиль!")
        return pd.DataFrame(), None

    print(f"\nОптимальный нижний процентиль: {best_percentile:.2f}")
    print(f"Оценка качества распределения: {best_score:.3f}")

    # Применяем оптимальные пороги и собираем итоговые данные
    lower_threshold = np.percentile(intensities_for_percentiles, best_percentile)
    print(f"Нижний порог интенсивности: {lower_threshold:.2f}")

    all_mz_filtered = []
    all_intensities_filtered = []

    for spectrum in exp:
        if spectrum.getMSLevel() != ms_level:
            continue

        rt = spectrum.getRT()
        if rt_range and not (rt_range[0] <= rt <= rt_range[1]):
            continue

        mz, intensities = spectrum.get_peaks()

        mask = intensities >= lower_threshold
        if upper_cutoff is not None:
            mask &= intensities <= upper_cutoff

        filtered_mz = mz[mask]
        filtered_intensities = intensities[mask]

        all_mz_filtered.extend(filtered_mz)
        all_intensities_filtered.extend(filtered_intensities)

    print(f"Точек до фильтрации: {len(all_masses)}")
    print(f"Точек после фильтрации: {len(all_mz_filtered)}")

    # Визуализация (опционально)
    if plot:
        try:
            import matplotlib.pyplot as plt
            bin_edges, smoothed_hist, raw_hist = best_distribution # type: ignore
            bin_centers = bin_edges + mass_bin_width / 2

            plt.figure(figsize=(12, 6))
            plt.bar(bin_centers, raw_hist, width=mass_bin_width, alpha=0.3, label='Исходное')
            plt.plot(bin_centers, smoothed_hist, 'r-', linewidth=2, label='Сглаженное')

            mask_400_600 = (bin_centers >= 400) & (bin_centers <= 600)
            if np.any(mask_400_600):
                max_idx = np.argmax(smoothed_hist[mask_400_600])
                peak_mass = bin_centers[mask_400_600][max_idx]
                plt.axvline(peak_mass, color='g', linestyle='--', label=f'Максимум: {peak_mass:.0f} Да')

            plt.axvline(800, color='r', linestyle=':', label='800 Да')
            plt.xlabel('m/z (Да)')
            plt.ylabel('Количество пиков')
            plt.title(f'Распределение масс пиков (нижний процентиль {best_percentile:.2f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except:
            pass

    return pd.DataFrame({
        'mass': all_mz_filtered,
        'intensity': all_intensities_filtered,
    }), best_percentile


def evaluate_mass_distribution(smoothed_hist, bin_edges, mass_bin_width):
    """
    Оценка качества распределения масс.
    Возвращает скор (чем выше, тем лучше).
    """
    bin_centers = bin_edges + mass_bin_width / 2

    # 1. Наличие выраженного максимума в диапазоне 400-600 Да
    mask_400_600 = (bin_centers >= 400) & (bin_centers <= 600)
    if not np.any(mask_400_600):
        return -np.inf

    max_in_range = np.max(smoothed_hist[mask_400_600])
    global_max = np.max(smoothed_hist)

    peak_score = max_in_range / (global_max + 1e-10)

    # 2. Минимум пиков после 800 Да
    mask_high_mass = bin_centers >= 800
    if np.any(mask_high_mass):
        high_mass_peaks = np.sum(smoothed_hist[mask_high_mass])
        total_peaks = np.sum(smoothed_hist)
        high_mass_ratio = high_mass_peaks / (total_peaks + 1e-10)
        high_mass_penalty = np.exp(-10 * high_mass_ratio)
    else:
        high_mass_penalty = 1.0

    # 3. Монотонность распределения
    max_idx = np.argmax(smoothed_hist)
    max_mass = bin_centers[max_idx]

    if max_idx > 5:
        before_max = smoothed_hist[:max_idx]
        increasing_points = np.sum(np.diff(before_max) >= -0.05 * max(before_max))
        increasing_ratio = increasing_points / (len(before_max) - 1) if len(before_max) > 1 else 1
    else:
        increasing_ratio = 1

    if len(smoothed_hist) - max_idx > 5:
        after_max = smoothed_hist[max_idx:]
        decreasing_points = np.sum(np.diff(after_max) <= 0.05 * max(after_max))
        decreasing_ratio = decreasing_points / (len(after_max) - 1) if len(after_max) > 1 else 1
    else:
        decreasing_ratio = 1

    monotonicity_score = (increasing_ratio + decreasing_ratio) / 2

    # 4. Бонус за положение максимума
    position_bonus = 1.0 if 400 <= max_mass <= 600 else 0.5

    total_score = peak_score * high_mass_penalty * monotonicity_score * position_bonus
    return total_score
      
def replace_except_last(s):
    last_index = s.rfind('_')
    if last_index == -1:  # если нет подчеркиваний
        return s
    
    before_last = s[:last_index].replace('_', '-')
    after_last = s[last_index:]
    
    new_name = before_last + after_last
    split_name = new_name.split(sep='_')[0]
    
    return split_name

def extract_mass_list_gmm_fast(mzml_file, ms_level=1, rt_range=None, max_samples=100000):
    exp = MSExperiment()
    MzMLFile().load(mzml_file, exp)
    
    all_mz = []
    all_ints = []
    
    # 1. Быстрый сбор данных через NumPy
    for spectrum in exp:
        if spectrum.getMSLevel() == ms_level:
            rt = spectrum.getRT()
            if rt_range and not (rt_range[0] <= rt <= rt_range[1]):
                continue
            
            mz, intensities = spectrum.get_peaks()
            all_mz.append(mz)
            all_ints.append(intensities)
    
    if not all_ints:
        return pd.DataFrame()

    # Объединяем массивы один раз в конце (векторизованно)
    mz_array = np.concatenate(all_mz)
    ints_array = np.concatenate(all_ints)
    
    # 2. Подготовка данных для GMM (только подвыборка для скорости)
    # Фильтруем нули сразу, чтобы не было ошибок log10
    valid_ints = ints_array[ints_array > 0]
    
    if len(valid_ints) > max_samples:
        # Случайная выборка для обучения
        training_data = np.random.choice(valid_ints, max_samples, replace=False)
    else:
        training_data = valid_ints

    # Создаем временный DF только для функции фильтрации, чтобы найти порог
    temp_df = pd.DataFrame({'intensity': training_data})
    
    # Вызываем ваш фильтр (он вернет порог)
    _, threshold = gmm_noise_filter(temp_df)
    
    # 3. Фильтрация основного массива через булеву маску (очень быстро)
    mask = ints_array >= threshold
    
    print(f"Порог найден: {threshold:.2f}. Точек сохранено: {np.sum(mask)}")
    
    return pd.DataFrame({
        'mass': mz_array[mask],
        'intensity': ints_array[mask]
    })
  
list_data_folder = ['2026_01_28','2026_02_05','2026_02_18','2026_03_12','2026_03_19','2026_03_26','2026_04_01']
#list_data_folder =['2025_09_17','2025_09_23','2025_10_24']

raw = False

for folder in list_data_folder:

    #convert_mass_spectra_batch(source_dir = fr"D:\lab\MassSpectraNew\Catalog\{folder}",output_base = fr"D:\lab\MassSpectraNew\Converted_Data\{folder}")

    mass_path = Path(fr"D:\lab\MassSpectraNew\Converted_Data\{folder}")
    document_save_path = Path(fr"D:\lab\MassSpectraNew\processed_results")
    #document_save_path = Path(fr"D:\lab\mass-spec-server\processed_results")

    for windows_path in mass_path.rglob('*.mzML'):
        
        path = str(windows_path)
        name = ut.extract_name_from_path(str(path))
        name = replace_except_last(name)
        print(name)
        
        if raw:
            ms_list = extract_mass_list_percentile(str(path), low_percentile=90)
            
            # Создаем интерактивный график
            p = interactive_spectrum_bokeh(ms_list, normalize=False, max_points=30000)
            html_path = Path.joinpath(document_save_path, 'raw', f'{name}__spectrum.html')
            output_file(html_path)
            save(p)
            continue       
        
        ms_list,_ = extract_mass_list_optimal_percentile(str(path))
        print(f'Число пиков: {len(ms_list)}')        

        if 'ADOM' not in name:
            continue

        ms_list.attrs['name'] = name
        
 
        
        ms_list = ms.recallibrate_optimize(ms_list,draw=False)
        #ms_list,_ = gmm_noise_filter(ms_list)
        spectra = ms.assign_optimized(ms_list,brutto_dict={'C':(4,50),'H':(4,80),'O':(0,50),'N':(0,3),'C_13':(0,1),'S':(0,1)},rel_error=1, sulfur_precision_factor=10,nitrogen_precision_factor=4)
        '''
        print(spectra.shape)
        spectra = ms.calc_mass(spectra)
        tmds_spectra = calc_by_brutto_new(spectra)
        
        tmds_spectra = ms.assign(tmds_spectra, brutto_dict={'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(0,1)})
        
        print(tmds_spectra[tmds_spectra['assign']==True])
        
        tmds_spectra = ms.calc_mass(tmds_spectra,debug=False)
        spectra = ms.assign_by_tmds_optimize(ms_assign, tmds_spectra, rel_error=0.5,max_num=100)
        '''
        spectra = spectra.loc[spectra['C'] != 0]
        spectra = spectra.loc[spectra['H'] != 0]
        spectra = spectra.loc[spectra['O'] != 0]
        
        
        spectra = ms.normalize(ms.calc_all_metrics(spectra))
        spectra = spectra.loc[spectra.groupby('brutto')['rel_error'].apply(lambda x: (x.abs() == x.abs().min()).idxmax())
                            ].reset_index(drop=True)
        spectra = ms.mol_class(spectra,how="perminova")
        

        spectra.dropna().to_csv(Path.joinpath(document_save_path,'mzlist',f'{name}__mzlist.csv'),sep=',',index=False)
        spectra.dropna().describe().to_excel(Path.joinpath(document_save_path,'statistic',f'{name}__statistic.xlsx'))
        
        msv.vk(spectra,sizes=(8,50))
        plt.savefig(Path.joinpath(document_save_path,'vk',f'{name}__vk.png'))
        plt.close()
        
        msv.vk(spectra,sizes=(8,50),plot_type='mass_scatter')
        plt.savefig(Path.joinpath(document_save_path,'mass_scatter',f'{name}__mass_scatter.png'))
        plt.close()
        
        msv.vk(spectra,sizes=(8,50),plot_type='density')
        plt.savefig(Path.joinpath(document_save_path,'density',f'{name}__density.png'))  
        plt.close()
        
        msv.plot_mass_intensity_relationship(spectra)
        plt.savefig(Path.joinpath(document_save_path,'mir',f'{name}__mir.png'))
        plt.close()
        
        msv.plot_mol_class_distribution(spectra,mod='bar')
        plt.savefig(Path.joinpath(document_save_path,'bar',f'{name}__bar.png')) 
        plt.close()
                       
        msv.plot_mol_class_distribution(spectra,mod='pie')
        plt.savefig(Path.joinpath(document_save_path,'pie',f'{name}__pie.png'))
        
        plt.close()
        msv.elemental_composition(spectra)
        
        plt.savefig(Path.joinpath(document_save_path,'comp',f'{name}__comp.png'))            
        plt.close()
                     
        msv.spectrum(spectra)
        plt.savefig(Path.joinpath(document_save_path,'spectrum',f'{name}__spectrum.png'))
        plt.close()


    