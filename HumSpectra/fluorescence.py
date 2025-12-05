import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from typing import Union, List
from matplotlib.axes import Axes
from scipy import interpolate, signal

import HumSpectra.utilits as ut

def calc_asm_350(data: pd.DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: asm 350, показатель асимметрии спектра при длине возбуждения 350 нм
    Функция рассчитывает отношение интеграла длины волны испускания от 420 до 460 нм к интегралу от 550 до 600 нм
    """

    row = data[350].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = cut_raman_spline(EM_wavelengths, row, 350)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 420)[0][0]:np.where(EM_wavelengths == 460)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 550)[0][0]:np.where(EM_wavelengths == 600)[0][0]])
    fluo_param = high / low

    return float(fluo_param)

def calc_asm_280(data: pd.DataFrame) -> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: asm 280, показатель асимметрии спектра при длине возбуждения 280 нм
    Функция рассчитывает отношение интеграла длины волны испускания от 350 до 400 нм к интегралу от 475 до 535 нм
    """

    row = data[280].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = cut_raman_spline(EM_wavelengths, row, 280)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 350)[0][0]:np.where(EM_wavelengths == 400)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 475)[0][0]:np.where(EM_wavelengths == 535)[0][0]])
    fluo_param = high / low

    return float(fluo_param)

def calc_fluo_index(data: pd.DataFrame)-> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: fluo_index, отношение интенсивности при длине волны испускания 450 нм к 500 нм при длине волны возбуждения 370 нм
    """
    row = data[370]
    fluo_param = row.loc[450]/row.loc[500]
    
    return float(fluo_param)

def calc_humin_index(data: pd.DataFrame)-> float:
    """
    :param data: DataFrame, спектр флуоресценции
    :return: fluo_param: humin_index, Функция рассчитывает отношение интеграла длины волны испускания от 435 до 480 нм к интегралу от 300 до 345 нм при длине возбуждения 254 нм.
    """
    row = data[255].to_numpy()
    EM_wavelengths = data.index.to_numpy(dtype="int")
    spline = cut_raman_spline(EM_wavelengths, row, 255)
    high = np.trapezoid(spline[np.where(EM_wavelengths == 435)[0][0]:np.where(EM_wavelengths == 480)[0][0]])
    low = np.trapezoid(spline[np.where(EM_wavelengths == 300)[0][0]:np.where(EM_wavelengths == 345)[0][0]])
    fluo_param = high / low

    return float(fluo_param) 

def cut_spectra(data: pd.DataFrame,
                ex_low_limit: int | None=None,
                ex_high_limit: int | None = None,
                em_low_limit: int | None = None,
                em_high_limit: int | None = None) -> pd.DataFrame:
    """
    :param data: DataFrame, спектр флуоресценции.
    :param ex_low_limit: int, нижний предел значения длины волны возбуждения спектра
    :param ex_high_limit: int, верхний предел значения длины волны возбуждения спектра
    :param em_low_limit: int, нижний предел значения длины волны испускания спектра
    :param em_high_limit: int, верхний предел значения длины волны испускания спектра
    :return: cut_data
    Функция обрезает спектр согласно заданным пределам и возвращает копию спектра
    """
    data = data.copy()
    
    if ex_low_limit is None:
        ex_low_limit = int(data.columns.min())

    if ex_high_limit is None:
        ex_high_limit = int(data.columns.max())
    
    if em_low_limit is None:
        em_low_limit = data.index.min()
    
    if em_high_limit is None:
        em_high_limit = data.index.max()
    
    cut_data = data.loc[em_low_limit:em_high_limit, ex_low_limit:ex_high_limit]

    return cut_data

def plot_heat_map(data: pd.DataFrame,
                  ax: Union[Axes, None] = None,
                  xlabel: bool = True,
                  ylabel: bool = True,
                  title: bool = True,
                  name: str|None = None) -> Axes:
    """
    :param data: DataFrame, спектр флуоресценции
    :param q: float, квантиль, определяющий экстремальные выбросы
    :return: ax: Axes, ось графика matplotlib.pyplot
    Функция маскирует экстремальные выбросы нулями, и рисует 2D тепловой график флуоресценции
    """
    
    data_copy = data.copy()
    
    if name is None:
        
        if "name" in data_copy.attrs:
            name = data_copy.attrs["name"]
        else:
            name = "fluo_spectra"
    
    EM_wavelengths = data_copy.index.to_numpy(dtype="int")
    EX_wavelengths = data_copy.columns.to_numpy(dtype="int")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.7, 4.8))
    ax.pcolormesh(EM_wavelengths, EX_wavelengths, data_copy.T, shading="gouraud",
                  cmap=plt.get_cmap('rainbow'))
    if xlabel:
        ax.set_xlabel("λ испускания, нм")
    if ylabel:
        ax.set_ylabel("λ возбуждения, нм")
    if title:
        ax.set_title(f"{name}")

    return ax

def plot_2d(data: pd.DataFrame,
            ex_wave: int,
            xlabel: bool = True,
            ylabel: bool = True,
            title: bool = True,
            ax: Union[Axes, None] = None,
            norm: bool = False,
            name: str|None = None) -> Axes:
    """
    :param data: DataFrame, спектр флуоресценции
    :param ex_wave: int, длина волны возбуждения, при котором строится график
    :param norm: bool, если установлен True, то нормирует график на максимум
    :return: ax: Axes, ось графика matplotlib.pyplot
    Функция возвращает график 2D флуоресценции при одной длине волны возбуждения
    """
    
    data_copy = data.copy()
    
    if name is None:
        
        if "name" in data_copy.attrs:
            name = data_copy.attrs["name"]
        else:
            name = "fluo_spectra"
    
    row = data[ex_wave]
    if norm:
        row = (row - row.min()) / (row.max() - row.min())
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(data.index, row, label = data.attrs['name'])
    if title:
        ax.set_title(f"{data.attrs['name']}, λ возбуждения: {ex_wave} нм")
    if xlabel:
        ax.set_xlabel("λ испускания, нм")
    if ylabel:
        ax.set_ylabel("Интенсивность")

    return ax

def where_is_raman(exc_wv: float,
                   omega: float = 3430.):
    """
    return the central wavelength of solvent raman scattering band (stokes mode)
    for given excitation wavelength (exc_wv) in nm
    exc_wv(float) - excitation wavelength in nm
    Omega(float) - molecular oscillations frequency in cm^-1
    return(float) central wavelength of raman stokes mode peak
    """
    return 1e7 / (1e7 / exc_wv - omega)

def cut_peak_spline(em_wvs, fl, peak_position, peak_half_width=15., points_in_spline=10):
    """
    em_wvs - 1d numpy array of emission wavelength
    fl - 1d numpy array of fluorescence
    peak_position (float) - position of peak in nanometers
    peak_half_width - half-width at half-maximum of peak to cut
    """
    mask = ((em_wvs < peak_position - peak_half_width) | (em_wvs > peak_position + peak_half_width)) & (~np.isnan(fl))

    knots = em_wvs[mask][1:-1][::points_in_spline]
    spline = scipy.interpolate.LSQUnivariateSpline(em_wvs[mask], fl[mask], knots, k=2, ext=1)
    spline_wvs = em_wvs[mask]

    return knots, mask, spline_wvs, spline

def cut_raman_spline(em_wvs: np.ndarray,
                     fl: np.ndarray,
                     exc_wv: int,
                     raman_freq: float = 3430.,
                     raman_hwhm: int = 15,
                     points_in_spline: int = 10) -> np.ndarray:
    """
    em_wvs - 1d numpy array of emission wavelength
    fl - 1d numpy array of fluorescence
    raman_freq (float) = 3430. - frequency of raman scattering in cm^-1
    raman_hwhm (float) - raman band half width at half maximum to cut in nm
    points_in_spline(int) - number of points in interpolating spline

    return 1d numpy array of fluorescence intensity with cut emission
    """

    res = cut_peak_spline(em_wvs, fl,
                          peak_position=where_is_raman(exc_wv, omega=raman_freq),
                          peak_half_width=raman_hwhm, points_in_spline=points_in_spline)
    
    return np.array(res[-1](em_wvs))

def remove_raman_scatter(eem_df, ex_wavelengths=None, em_wavelengths=None, 
                         method='interpolation', width_nm=15, plot=False, 
                         water_raman_peak=None, return_mask=False):
    """
    Удаление пика комбинационного рассеяния (Raman scatter) из EEM спектра.
    
    Параметры:
    ----------
    eem_df : pandas DataFrame
        EEM спектр, где:
        - индексы: длины волн испускания (emission)
        - столбцы: длины волны возбуждения (excitation)
        
    ex_wavelengths : array-like, optional
        Явное указание длин волн возбуждения. Если None, берутся из столбцов df.
        
    em_wavelengths : array-like, optional
        Явное указание длин волн испускания. Если None, берутся из индексов df.
        
    method : str, default='interpolation'
        Метод удаления:
        - 'interpolation': интерполяция через соседние точки
        - 'savgol': фильтр Savitzky-Golay для сглаживания
        - 'linear': линейная интерполяция между краями
        
    width_nm : float, default=15
        Ширина области вокруг пика КР (в нм) для обработки.
        
    plot : bool, default=False
        Визуализация процесса удаления для одного выбранного возбуждения.
        
    water_raman_peak : float, optional
        Положение пика КР воды (смещение от возбуждения в см⁻¹).
        По умолчанию: ~3400 см⁻¹ для воды (что соответствует ~30-40 нм при typ. возбуждении).
        
    return_mask : bool, default=False
        Возвращать также маску удаленной области.
        
    Возвращает:
    -----------
    eem_corrected : pandas DataFrame
        EEM спектр с удаленным пиком КР.
        
    raman_mask : pandas DataFrame (опционально)
        Маска удаленной области (True - удаленные точки).
    """
    
    # Копируем данные для безопасности
    eem_data = eem_df.copy()
    
    # Получаем длины волн
    if ex_wavelengths is None:
        ex_wavelengths = eem_data.columns.values.astype(float)
    else:
        ex_wavelengths = np.array(ex_wavelengths)
        
    if em_wavelengths is None:
        em_wavelengths = eem_data.index.values.astype(float)
    else:
        em_wavelengths = np.array(em_wavelengths)
    
    # Проверяем монотонность
    if not (np.all(np.diff(ex_wavelengths) > 0) and np.all(np.diff(em_wavelengths) > 0)):
        print("Предупреждение: Длины волн не монотонно возрастают. Возможны проблемы.")
    
    # Создаем массив для исправленного EEM
    eem_corrected = eem_data.values.copy().astype(float)
    
    # Маска удаленных областей
    raman_mask = np.zeros_like(eem_corrected, dtype=bool)
    
    # Для каждого возбуждения находим и удаляем пик КР
    for i, ex_wl in enumerate(ex_wavelengths):
        # 1. Определяем положение пика КР для данного возбуждения
        if water_raman_peak is None:
            # Стандартное смещение для воды: ~3400 см⁻¹
            # Переводим в нанометры: 1/λ_ex - 1/λ_raman = 3400 * 1e-7
            # λ_raman = 1 / (1/ex_wl - 3400e-7)
            ex_wl_cm = 1e7 / ex_wl  # возбуждение в см⁻¹
            raman_wl_cm = ex_wl_cm - 3400  # положение КР в см⁻¹
            raman_wl_nm = 1e7 / raman_wl_cm  # положение КР в нм
        else:
            # Явно заданное смещение в см⁻¹
            ex_wl_cm = 1e7 / ex_wl
            raman_wl_cm = ex_wl_cm - water_raman_peak
            raman_wl_nm = 1e7 / raman_wl_cm
        
        # 2. Определяем индексы для удаления вокруг пика КР
        # Находим ближайшую длину волны испускания
        em_idx = np.argmin(np.abs(em_wavelengths - raman_wl_nm))
        raman_em_wl = em_wavelengths[em_idx]
        
        # Определяем диапазон для удаления
        half_width = width_nm / 2
        lower_bound = raman_em_wl - half_width
        upper_bound = raman_em_wl + half_width
        
        # Находим индексы в диапазоне
        mask_indices = np.where((em_wavelengths >= lower_bound) & 
                                (em_wavelengths <= upper_bound))[0]
        
        if len(mask_indices) == 0:
            # Если нет точек в диапазоне, пропускаем
            continue
        
        # 3. Удаляем пик КР выбранным методом
        em_intensity = eem_corrected[:, i].copy()
        
        if method == 'interpolation':
            # Метод интерполяции через соседние точки
            # Создаем маску для интерполяции (все точки кроме удаляемых)
            valid_mask = np.ones(len(em_wavelengths), dtype=bool)
            valid_mask[mask_indices] = False
            
            if np.sum(valid_mask) >= 2:  # Нужно минимум 2 точки для интерполяции
                # Интерполяция кубическим сплайном
                f_interp = interpolate.interp1d(
                    em_wavelengths[valid_mask],
                    em_intensity[valid_mask],
                    kind='cubic',
                    fill_value='extrapolate' # type: ignore
                )
                # Заменяем удаляемую область интерполированными значениями
                eem_corrected[mask_indices, i] = f_interp(em_wavelengths[mask_indices])
                raman_mask[mask_indices, i] = True
        
        elif method == 'savgol':
            # Метод Savitzky-Golay фильтра
            # Временно заполняем удаляемую область соседними значениями
            temp_intensity = em_intensity.copy()
            
            # Находим индексы до и после удаляемой области
            before_idx = mask_indices[0] - 1
            after_idx = mask_indices[-1] + 1
            
            if before_idx >= 0 and after_idx < len(em_intensity):
                # Линейная интерполяция между краями
                left_val = em_intensity[before_idx]
                right_val = em_intensity[after_idx]
                n_points = len(mask_indices)
                
                for j, idx in enumerate(mask_indices):
                    temp_intensity[idx] = left_val + (right_val - left_val) * (j + 1) / (n_points + 1)
            
            # Применяем фильтр Savitzky-Golay
            window_length = min(11, len(em_intensity) // 3 * 2 + 1)  # нечетное
            if window_length >= 3:
                smoothed = signal.savgol_filter(temp_intensity, 
                                               window_length=window_length,
                                               polyorder=2)
                # Заменяем только удаляемую область
                eem_corrected[mask_indices, i] = smoothed[mask_indices]
                raman_mask[mask_indices, i] = True
        
        elif method == 'linear':
            # Простая линейная интерполяция
            # Находим значения до и после удаляемой области
            before_idx = mask_indices[0] - 1
            after_idx = mask_indices[-1] + 1
            
            if before_idx >= 0 and after_idx < len(em_intensity):
                left_val = em_intensity[before_idx]
                right_val = em_intensity[after_idx]
                n_points = len(mask_indices)
                
                for j, idx in enumerate(mask_indices):
                    eem_corrected[idx, i] = left_val + (right_val - left_val) * (j + 1) / (n_points + 1)
                raman_mask[mask_indices, i] = True
        
        # 4. Визуализация для первого или выбранного возбуждения
        if plot and (i == len(ex_wavelengths) // 2 or i == 0):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Сырые данные
            ax = axes[0, 0]
            ax.plot(em_wavelengths, em_intensity, 'b-', label='Исходный', alpha=0.7)
            ax.axvspan(lower_bound, upper_bound, alpha=0.3, color='red', label='Область КР')
            ax.axvline(raman_em_wl, color='r', linestyle='--', alpha=0.5, label='Пик КР')
            ax.set_xlabel('Длина волны испускания (нм)')
            ax.set_ylabel('Интенсивность')
            ax.set_title(f'Возбуждение {ex_wl:.0f} нм\nПик КР при {raman_em_wl:.1f} нм')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # После коррекции
            ax = axes[0, 1]
            ax.plot(em_wavelengths, em_intensity, 'b-', label='Исходный', alpha=0.5)
            ax.plot(em_wavelengths, eem_corrected[:, i], 'r-', label='Корректированный', linewidth=2)
            ax.axvspan(lower_bound, upper_bound, alpha=0.3, color='red')
            ax.set_xlabel('Длина волны испускания (нм)')
            ax.set_ylabel('Интенсивность')
            ax.set_title(f'Сравнение до/после коррекции')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Разница
            ax = axes[1, 0]
            difference = em_intensity - eem_corrected[:, i]
            ax.plot(em_wavelengths, difference, 'g-')
            ax.fill_between(em_wavelengths, 0, difference, where=(difference>0), 
                           alpha=0.3, color='green', label='Удалено')
            ax.fill_between(em_wavelengths, 0, difference, where=(difference<0), 
                           alpha=0.3, color='red', label='Добавлено')
            ax.set_xlabel('Длина волны испускания (нм)')
            ax.set_ylabel('Разница')
            ax.set_title('Разница между исходным и корректированным')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2D визуализация маски
            ax = axes[1, 1]
            mask_display = np.zeros_like(eem_corrected)
            mask_display[raman_mask] = 1
            im = ax.imshow(mask_display, aspect='auto', 
                          extent=[ex_wavelengths[0], ex_wavelengths[-1], 
                                  em_wavelengths[-1], em_wavelengths[0]],
                          cmap='Reds')
            ax.set_xlabel('Возбуждение (нм)')
            ax.set_ylabel('Испускание (нм)')
            ax.set_title('Маска удаленных областей КР')
            plt.colorbar(im, ax=ax, label='Удалено (1) / Сохранено (0)')
            
            plt.tight_layout()
            plt.show()
    
    # Конвертируем обратно в DataFrame
    eem_corrected_df = pd.DataFrame(eem_corrected, 
                                    index=eem_data.index,
                                    columns=eem_data.columns)
    
    if return_mask:
        raman_mask_df = pd.DataFrame(raman_mask,
                                     index=eem_data.index,
                                     columns=eem_data.columns)
        return eem_corrected_df, raman_mask_df
    
    return eem_corrected_df

def read_fluo_3d(path: str,
                 sep: str | None = None,
                 index_col: int | None = None,  # Изменено на None по умолчанию
                 debug: bool = False,
                 remove_raman=True) -> pd.DataFrame:
    """
    :param path: путь к файлу в строчном виде,
            (example: "C:/Users/mnbv2/Desktop/lab/KNP work directory/Флуоресценция/ADOM-SL2-1.csv").
    :param sep: разделитель в строчном виде (example: ",").
    :param index_col: номер столбца, который считается индексом таблицы. Если None, будет определён автоматически.
    :return: DataFrame: Таблица, в котором индексами строк являются длины волн испускания, имена столбцов - длины волн
            возбуждения. Таблица имеет метаданные - имя образца, класс, подкласс
    """

    extension = path.split(sep=".")[-1]
    
    if sep is None and (extension == "csv" or extension == "txt"):
        sep = ut.check_sep(path)

    try:
        # Сначала читаем небольшой кусок данных для анализа структуры
        if extension == "xlsx":
            sample_data = pd.read_excel(path, nrows=5)
        elif extension == "csv" or extension == "txt":
            sample_data = pd.read_csv(path, sep=sep, nrows=5)
        else:
            raise KeyError("Тип данных не поддерживается")
        
        # Автоматическое определение индексного столбца
        if index_col is None:
            # Проверяем, есть ли столбец без имени в позиции 0 (типичный индекс)
            if sample_data.columns[0] == 'Unnamed: 0' or sample_data.columns[0] == '':
                index_col = 0
                if debug:
                    print(f"Автоматически определён индексный столбец: {index_col}")
            else:
                # Если нет явного индексного столбца, создаём обычный числовой индекс
                index_col = False
                if debug:
                    print("Индексный столбец не найден, используется стандартный индекс")
        
        # Чтение полных данных с определённым индексом
        if extension == "xlsx":
            if index_col is False:
                data = pd.read_excel(path)
                # Удаляем безымянные столбцы если они есть
                unnamed_columns = [col for col in data.columns if str(col).startswith('Unnamed:')]
                if unnamed_columns:
                    data = data.drop(unnamed_columns, axis=1)
            else:
                data = pd.read_excel(path, index_col=index_col)
                # Удаляем безымянные столбцы если они есть
                unnamed_columns = [col for col in data.columns if str(col).startswith('Unnamed:')]
                if unnamed_columns:
                    data = data.drop(unnamed_columns, axis=1)
                    
        elif extension == "csv" or extension == "txt":
            if index_col is False:
                data = pd.read_csv(path, sep=sep)
                # Удаляем безымянные столбцы если они есть
                unnamed_columns = [col for col in data.columns if str(col).startswith('Unnamed:')]
                if unnamed_columns:
                    data = data.drop(unnamed_columns, axis=1)
            else:
                data = pd.read_csv(path, sep=sep, index_col=index_col)
                # Удаляем безымянные столбцы если они есть
                unnamed_columns = [col for col in data.columns if str(col).startswith('Unnamed:')]
                if unnamed_columns:
                    data = data.drop(unnamed_columns, axis=1)
        
        else:
            raise KeyError(f"Недопустимое расширение спектра {path}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Файл пуст: {path}")
    
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")
    
    name = ut.extract_name_from_path(path)

    # Удаляем строку с "nm" если она присутствует в индексе
    if "nm" in data.index:
        data = data.drop("nm", axis=0)

    if debug:
        print(f"Имя файла: {name}")
        print(f"Размер данных: {data.shape}")
        print(f"Первые 5 строк:\n{data.head()}")

    # Преобразование типов с обработкой возможных ошибок
    try:
        data = data.astype("float64")
        data.columns = data.columns.astype(int)
        data.index = data.index.astype(int)
        
    except (ValueError, TypeError) as e:
        if debug:
            print(f"Предупреждение: ошибка преобразования типов: {e}")
        # Пробуем преобразовать только числовые столбцы
        data = data.apply(pd.to_numeric, errors='coerce')
        
    data.dropna(axis=0,inplace=True)
    data.set_index(data.columns[0], inplace=True,drop=True)
    data.index = data.index.astype(int)
    data.columns = data.columns.astype(int)

    data.attrs['name'] = name
    data.attrs['class'] = ut.extract_class_from_name(name)
    data.attrs['subclass'] = ut.extract_subclass_from_name(name)
    data.attrs['path'] = path
    
    if remove_raman:
        
        data = remove_raman_scatter(data)
    
    return data # type: ignore

