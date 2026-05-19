from pyopenms import * # type: ignore
from pathlib import Path
import pandas as pd
import HumSpectra.mass_spectra as ms
import HumSpectra.mass_visualizer as msv
import HumSpectra.mass_descriptors as md
import HumSpectra.utilits as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

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
        
        mask = (intensities >= low_percentile_threshold) & (intensities <= high_percentile_threshold)
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
      
list_data_folder =['2026_04_01']

raw = False

for folder in list_data_folder:

    #convert_mass_spectra_batch(source_dir = fr"D:\lab\MassSpectraNew\Catalog\{folder}",output_base = fr"D:\lab\MassSpectraNew\Converted_Data\{folder}")

    mass_path = Path(fr"D:\lab\MassSpectraNew\Converted_Data\{folder}")
    #document_save_path = Path(fr"D:\lab\MassSpectraNew\processed_results")
    document_save_path = Path(fr"D:\lab\mass-spec-server\processed_results")

    for windows_path in mass_path.rglob('*.mzML'):
        
        path = str(windows_path)
        name = ut.extract_name_from_path(str(path))
        name = ut.delete_series_number(name)
        print(name)
        
        if raw:
            ms_list = extract_mass_list_percentile(str(path), low_percentile=98)
            
            # Создаем интерактивный график
            fig = msv.interactive_spectrum_plotly(ms_list, normalize=False, max_points=30000)
            html_path = Path.joinpath(document_save_path, 'raw', f'{name}__spectrum.html')
            fig.write_html(html_path)
            continue       
        
        ms_list = extract_mass_list_percentile(str(path), low_percentile=99.8,high_percentile=99.99)
        print(f'Число пиков: {len(ms_list)}')        
        
        
        ms_list.attrs['name'] = name
        
        ms_list = ms.recallibrate_optimize(ms_list,draw=False)

        spectra = ms.assign_optimized(ms_list,brutto_dict={'C':(4,50),'H':(4,80),'O':(0,50),'N':(0,3),'C_13':(0,1),'S':(0,2)},rel_error=0.5, sulfur_precision_factor=10,nitrogen_precision_factor=4)
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
        
        
        spectra = ms.normalize(md.calc_all_metrics(spectra))
        spectra = spectra.loc[spectra.groupby('brutto')['rel_error'].apply(lambda x: (x.abs() == x.abs().min()).idxmax())
                            ].reset_index(drop=True)
        spectra = md.mol_class(spectra,how="perminova")
        

        paths = {
            'mzlist': Path.joinpath(document_save_path, 'mzlist'),
            'statistic': Path.joinpath(document_save_path, 'statistic'),
            'vk': Path.joinpath(document_save_path, 'vk'),
            'mass_scatter': Path.joinpath(document_save_path, 'mass_scatter'),
            'density': Path.joinpath(document_save_path, 'density'),
            'mir': Path.joinpath(document_save_path, 'mir'),
            'bar': Path.joinpath(document_save_path, 'bar'),
            'pie': Path.joinpath(document_save_path, 'pie'),
            'comp': Path.joinpath(document_save_path, 'comp'),
            'spectrum': Path.joinpath(document_save_path, 'spectrum')
        }

        # Создаем все необходимые директории (если их нет)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # Сохраняем файлы
        spectra.dropna().to_csv(paths['mzlist'] / f'{name}__mzlist.csv', sep=',', index=False)
        spectra.dropna().describe().to_excel(paths['statistic'] / f'{name}__statistic.xlsx')

        msv.vk(spectra, sizes=(8,50))
        plt.savefig(paths['vk'] / f'{name}__vk.png')
        plt.close()

        msv.vk(spectra, sizes=(8,50), plot_type='mass_scatter')
        plt.savefig(paths['mass_scatter'] / f'{name}__mass_scatter.png')
        plt.close()

        msv.vk(spectra, sizes=(8,50), plot_type='density')
        plt.savefig(paths['density'] / f'{name}__density.png')  
        plt.close()

        msv.plot_mass_intensity_relationship(spectra)
        plt.savefig(paths['mir'] / f'{name}__mir.png')
        plt.close()

        msv.plot_mol_class_distribution(spectra, mod='bar')
        plt.savefig(paths['bar'] / f'{name}__bar.png') 
        plt.close()

        msv.plot_mol_class_distribution(spectra, mod='pie')
        plt.savefig(paths['pie'] / f'{name}__pie.png')
        plt.close()

        msv.elemental_composition(spectra)
        plt.savefig(paths['comp'] / f'{name}__comp.png')            
        plt.close()

        msv.spectrum(spectra, xlim=(200,800))
        plt.savefig(paths['spectrum'] / f'{name}__spectrum.png')
        plt.close()


    