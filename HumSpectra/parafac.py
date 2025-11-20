import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import HumSpectra.fluorescence as fl
import HumSpectra.utilits as ut
class EEMDataLoader:
    def __init__(self):
        self.data_folder = None
        self.sample_names = []
        self.excitation_wavelengths = None
        self.emission_wavelengths = None
        
    def load_eem_data(self, data_folders=None, index_col=None):
        """Загрузка всех EEM спектров из одной или нескольких папок"""
        

        if data_folders is None:
            data_folders = [self.data_folder]

        elif isinstance(data_folders, (str, Path)):
            data_folders = [data_folders]
        
        eem_matrices = []
        self.sample_names = []  # Очищаем имена образцов
        
        for data_folder in data_folders:
            data_folder = Path(data_folder)
            
            # Находим все CSV файлы в папке
            csv_files = list(data_folder.glob("*.csv"))
            csv_files.sort()  # Сортируем для воспроизводимости
            
            if not csv_files:
                print(f"Предупреждение: CSV файлы не найдены в папке {data_folder}")
                continue
            
            print(f"Загрузка данных из папки: {data_folder}")
            print(f"Найдено файлов: {len(csv_files)}")
            
            for i, window_path in enumerate(csv_files):
                try:
                    # Загружаем CSV файл
                    file_path = str(window_path)
                    
                    df = fl.read_fluo_3d(file_path, index_col=index_col)
                    df = fl.cut_spectra(df, ex_low_limit=255, ex_high_limit=450, 
                                    em_low_limit=255, em_high_limit=600)
                    
                    # Извлекаем длины волн (только при первой успешной загрузке)
                    if self.excitation_wavelengths is None:
                        self.excitation_wavelengths = np.array(df.columns, dtype=float)
                    if self.emission_wavelengths is None:
                        self.emission_wavelengths = np.array(df.index, dtype=float)
                    
                    # Проверяем совместимость размерностей
                    current_ex_wavelengths = np.array(df.columns, dtype=float)
                    current_em_wavelengths = np.array(df.index, dtype=float)
                    
                    if (not np.array_equal(self.excitation_wavelengths, current_ex_wavelengths) or
                        not np.array_equal(self.emission_wavelengths, current_em_wavelengths)):
                        print(f"Предупреждение: Размерности в файле {file_path.name} не совпадают с предыдущими. Файл пропущен.")
                        continue
                    
                    # Преобразуем в numpy array
                    eem_matrix = df.values.astype(float)
                    eem_matrices.append(eem_matrix)
                    
                    sample_name = ut.extract_name_from_path(file_path)
                    self.sample_names.append(sample_name)
                    
                except Exception as e:
                    print(f"Ошибка при загрузке файла {file_path}: {e}")
                    continue
        
        if not eem_matrices:
            raise ValueError(f"Не удалось загрузить ни одного спектра из папок: {data_folders}")
        
        # Создаем 3D тензор: образцы × испускание × возбуждение
        self.tensor = np.stack(eem_matrices, axis=0)
        
        print(f"\nРезультат загрузки:")
        print(f"Создан тензор размерности: {self.tensor.shape}")
        print(f"Образцы: {len(self.sample_names)}")
        print(f"Длины волн испускания: {len(self.emission_wavelengths)}")
        print(f"Длины волн возбуждения: {len(self.excitation_wavelengths)}")
        print(f"Загружено из папок: {len(data_folders)}")
        
        return (tl.tensor(self.tensor), self.emission_wavelengths, 
                self.excitation_wavelengths, self.sample_names)
    
    def get_data_info(self):
        """Возвращает информацию о загруженных данных"""
        return {
            'tensor': self.tensor,
            'sample_names': self.sample_names,
            'excitation_wavelengths': self.excitation_wavelengths,
            'emission_wavelengths': self.emission_wavelengths,
            'shape': self.tensor.shape
        }
        
    def robust_detect_outliers(self, tensor, rank, threshold_std=3.0, verbose=True):
        """
        Устойчивая версия с несколькими способами реконструкции.
        """
        try:
            tensor = tl.tensor(tensor.copy())
            
            # Выполняем PARAFAC
            parafac_result, errors_hals = non_negative_parafac_hals(
                tensor, 
                rank=rank, 
                n_iter_max=100,
                init='random',
                return_errors=True
            )
            
            reconstructed = tl.cp_to_tensor(parafac_result)
            
            # Обнаружение выбросов
            residuals = tensor - reconstructed
            residual_flat = residuals.flatten()
            
            residual_mean = np.mean(residual_flat)
            residual_std = np.std(residual_flat)
            z_scores = np.abs((residual_flat - residual_mean) / residual_std)
            
            outlier_mask = (z_scores > threshold_std).reshape(tensor.shape)
            cleaned_tensor = tensor.copy()
            cleaned_tensor[outlier_mask] = reconstructed[outlier_mask]
            
            n_outliers = np.sum(outlier_mask)
            
            if verbose:
                print(f"Обнаружено выбросов: {n_outliers}")
            
            return cleaned_tensor, outlier_mask, residuals, {
                'n_outliers': n_outliers,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'parafac_result': parafac_result[1]
            }
            
        except Exception as e:
            print(f"Ошибка при обнаружении выбросов: {e}")
            return tensor, np.zeros_like(tensor, dtype=bool), None, {}
        
class OpticalDataAnalyzer:
    def __init__(self, n_components=3):
        self.n_components = n_components
        
    def fit_parafac(self, X, excitation_wavelengths=None, emission_wavelengths=None, sample_names=None, 
                    init='svd', n_iter_max=2000, tol=1e-8):
        """
        PARAFAC разложение с реальными длинами волн
        
        Args:
            X: 3D тензор (образцы × испускание × возбуждение)
            excitation_wavelengths: массив длин волн возбуждения
            emission_wavelengths: массив длин волн испускания
            sample_names: список имен образцов
            init: метод инициализации ('svd', 'random')
            n_iter_max: максимальное количество итераций
            tol: точность сходимости
        """
        self.excitation_wavelengths = excitation_wavelengths
        self.emission_wavelengths = emission_wavelengths
        self.sample_names = sample_names
        self.X = X
        
        # Предобработка данных для DOM анализа
        X_processed = self._preprocess_data(X)
        
        # PARAFAC разложение
        tensor_hals, errors_hals = non_negative_parafac_hals(
            X_processed, 
            rank=self.n_components,
            init=init,
            tol=tol,
            n_iter_max=n_iter_max,
            verbose=False,  # Вывод прогресса
            return_errors=True
        )
        
        self.parafac_result = tensor_hals
        self.factors = tensor_hals[1]
        self.errors = errors_hals
        
        # Валидация результатов
        self._validate_results()
        
        # Дополнительная информация о разложении
        self._calculate_additional_metrics(X_processed)
            
        return self.factors

    def _preprocess_data(self, X):
        """
        Предобработка данных для DOM анализа
        """
        X_processed = X.copy()
        
        # 1. Убедиться в неотрицательности данных
        if np.any(X_processed < 0):
            #print("⚠️ Обнаружены отрицательные значения. Заменяем на 0.")
            X_processed = np.maximum(X_processed, 0)
        
        # 2. Нормализация данных (опционально)
        # Для неотрицательного PARAFAC можно использовать нормализацию
        max_val = np.max(X_processed)
        if max_val > 1:
            #print(f"Масштабирование данных: максимальное значение = {max_val}")
            X_processed = X_processed / max_val
        
        return X_processed

    def _validate_results(self):
        """
        Валидация результатов PARAFAC разложения
        """
        print("\n=== ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ ===")
        
        # Валидация структуры факторов
        if not isinstance(self.factors, (list, tuple)):
            print(f"❌ self.factors должен быть list/tuple, получен {type(self.factors)}")
            return
        
        print(f"Количество факторов: {len(self.factors)}")
        
        # Проверка неотрицательности с безопасным доступом
        tolerance = 1e-10
        for i, factor in enumerate(self.factors):
            if not hasattr(factor, 'shape'):
                print(f"❌ Фактор {i} не является массивом numpy")
                continue
                
            try:
                min_val = np.min(factor)
                if min_val < -tolerance:
                    print(f"❌ Фактор {i} содержит отрицательные значения: min = {min_val:.2e}")
                else:
                    print(f"✅ Фактор {i}: неотрицательный (min = {min_val:.2e})")
            except Exception as e:
                print(f"❌ Ошибка при анализе фактора {i}: {e}")
        
        # Проверка сходимости с безопасным доступом
        if hasattr(self, 'errors') and self.errors is not None:
            try:
                errors = self.errors
                if len(errors) > 1:
                    final_error = errors[-1]
                    prev_error = errors[-2]
                    convergence = abs(final_error - prev_error) / prev_error
                    print(f"Ошибка реконструкции: {final_error:.2e}")
                    print(f"Сходимость: {convergence:.2e}")
                elif len(errors) == 1:
                    print(f"Финальная ошибка: {errors[0]:.2e}")
            except Exception as e:
                print(f"❌ Ошибка при анализе сходимости: {e}")
        
        # Информация о факторах с проверкой
        for i, factor in enumerate(self.factors):
            try:
                if hasattr(factor, 'shape'):
                    shape_info = f"форма {factor.shape}"
                    range_info = f"диапазон [{np.min(factor):.3f}, {np.max(factor):.3f}]"
                    print(f"Фактор {i}: {shape_info}, {range_info}")
                else:
                    print(f"Фактор {i}: не является массивом (тип: {type(factor)})")
            except Exception as e:
                print(f"❌ Ошибка при анализе фактора {i}: {e}")

    def _calculate_additional_metrics(self, X):
        """
        Расчет дополнительных метрик качества разложения
        """
        try:
            # Реконструкция тензора
            reconstructed = tl.cp_to_tensor(self.parafac_result)
            
            # Объясненная дисперсия
            ss_total = np.sum(X ** 2)
            ss_residual = np.sum((X - reconstructed) ** 2)
            explained_variance = (1 - ss_residual / ss_total) * 100
            
            print(f"\n=== МЕТРИКИ КАЧЕСТВА ===")
            print(f"Объясненная дисперсия: {explained_variance:.2f}%")
            
            # Core Consistency (опционально)
            core_consistency = self._calculate_core_consistency(X, reconstructed)
            if core_consistency is not None:
                print(f"Core Consistency: {core_consistency:.2f}%")
            
            self.explained_variance = explained_variance
            
        except Exception as e:
            print(f"Не удалось рассчитать дополнительные метрики: {e}")

    def _calculate_core_consistency(self, X, reconstructed):
        """
        Расчет Core Consistency для проверки модели
        """
        try:
            # Упрощенная версия расчета core consistency
            correlation = np.corrcoef(X.flatten(), reconstructed.flatten())[0, 1]
            return correlation * 100
        except:
            return None

    def reconstruct_sample(self, sample_idx):
        """Реконструкция EEM для конкретного образца"""
        reconstructed = np.zeros((len(self.emission_wavelengths),  # pyright: ignore[reportArgumentType]
                                len(self.excitation_wavelengths))) # type: ignore
        
        for r in range(self.n_components):
            concentration = self.factors[0][sample_idx, r]
            excitation_profile = self.factors[2][:, r]
            emission_profile = self.factors[1][:, r]
            
            outer_product = np.outer(emission_profile, excitation_profile)
            reconstructed += concentration * outer_product
        
        return reconstructed
    
    def get_component_maxima(self):
        """
        Возвращает максимумы возбуждения и испускания для каждого компонента
        """
        maxima = []
        
        for i in range(self.n_components):
            # Максимум возбуждения
            exc_max_idx = np.argmax(self.factors[2][:, i])
            exc_max_wavelength = self.excitation_wavelengths[exc_max_idx] if self.excitation_wavelengths is not None else exc_max_idx
            exc_max_intensity = self.factors[2][exc_max_idx, i]
            
            # Максимум испускания
            em_max_idx = np.argmax(self.factors[1][:, i])
            em_max_wavelength = self.emission_wavelengths[em_max_idx] if self.emission_wavelengths is not None else em_max_idx
            em_max_intensity = self.factors[1][em_max_idx, i]
            
            maxima.append({
                'component': i + 1,
                'excitation_max': exc_max_wavelength,
                'excitation_intensity': exc_max_intensity,
                'emission_max': em_max_wavelength,
                'emission_intensity': em_max_intensity
            })
        
        return maxima
            
    def get_component_loadings(self, normalization='max'):
        """
        Получить таблицу с долями компонентов для каждого образца
        
        Args:
            normalization: метод нормализации
                - 'max': нормализация по максимуму (0-1)
                - 'sum': нормализация к сумме = 1 (доли)
                - 'none': без нормализации
                - 'zscore': z-score нормализация
        
        Returns:
            pandas.DataFrame: таблица с долями компонентов
        """
        if not hasattr(self, 'factors'):
            raise ValueError("Сначала выполните fit_parafac()")
        
        # Фактор для образцов (предполагаем, что это первый мод)
        sample_factor = self.factors[0]
        
        # Создаем DataFrame
        if self.sample_names is not None:
            index = self.sample_names
        else:
            index = [f"Sample_{i+1}" for i in range(sample_factor.shape[0])]
        
        columns = [f"Component_{i+1}" for i in range(self.n_components)]
        
        loadings_df = pd.DataFrame(sample_factor, index=index, columns=columns)
        
        
        # Применяем нормализацию
        if normalization == 'max':
            # Нормализация по максимуму в каждом компоненте
            loadings_df = loadings_df / loadings_df.abs().max()
        elif normalization == 'sum':
            # Нормализация к сумме = 1 (доли)
            loadings_df = loadings_df.div(loadings_df.sum(axis=1), axis=0)
        elif normalization == 'row_sum':
            # Нормализация по строкам (сумма по образцу = 1)
            loadings_df = loadings_df.div(loadings_df.sum(axis=1), axis=0)
        elif normalization == 'zscore':
            # Z-score нормализация
            loadings_df = (loadings_df - loadings_df.mean()) / loadings_df.std()
        elif normalization == 'percentage':
            loadings_df = loadings_df.abs().div(loadings_df.abs().sum(axis=1), axis=0) * 100
        elif normalization == 'none':
            # Без нормализации
            pass
        else:
            raise ValueError(f"Неизвестный метод нормализации: {normalization}")
        
        loadings_df["Subclass"] = [ut.extract_subclass_from_name(sample) for sample in self.sample_names]
        return loadings_df
    
    def get_relative_contributions(self, method='absolute'):
        """
        Получить относительные вклады компонентов
        
        Args:
            method: метод расчета вкладов
                - 'absolute': абсолютные значения нагрузок
                - 'percentage': проценты от общей суммы
                - 'variance': вклад в объясненную дисперсию
        
        Returns:
            pandas.DataFrame: таблица с относительными вкладами
        """
        loadings_df = self.get_component_loadings(normalization='none')
        numeric_columns = loadings_df.select_dtypes(include=['number']).columns

        if method == 'absolute':
            return loadings_df[numeric_columns]
            
        elif method == 'percentage':
            # Процентный вклад каждого компонента в каждом образце
            percentages = loadings_df[numeric_columns].abs().div(loadings_df[numeric_columns].abs().sum(axis=1), axis=0) * 100
            return percentages.round(4)
            
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def get_component_summary(self):
        """
        Получить сводную таблицу по компонентам
        """
        # Доли образцов
        sample_loadings = self.get_component_loadings(normalization='none')
        
        # Процентные вклады
        percentages = self.get_relative_contributions(method='percentage')
        
        summary_data = {}
        for i in range(self.n_components):
            comp_name = f"Component_{i+1}"
            summary_data[comp_name] = {
                'Max_Loading': sample_loadings[comp_name].max(),
                'Min_Loading': sample_loadings[comp_name].min(),
                'Mean_Loading': sample_loadings[comp_name].mean(),
                'Std_Loading': sample_loadings[comp_name].std(),
                'Max_Percentage': percentages[comp_name].max(),
                'Mean_Percentage': percentages[comp_name].mean(),
            }
        
        return pd.DataFrame(summary_data).T
    
    def get_sample_composition(self, sample_index=None, sample_name=None, top_components=None):
        """
        Получить состав конкретного образца по компонентам
        
        Args:
            sample_index: индекс образца
            sample_name: имя образца
            top_components: количество главных компонентов для отображения
        
        Returns:
            pandas.Series: состав образца
        """
        if sample_name is not None and self.sample_names is not None:
            if sample_name not in self.sample_names:
                raise ValueError(f"Образец {sample_name} не найден")
            sample_index = self.sample_names.index(sample_name)
        
        if sample_index is None:
            raise ValueError("Укажите sample_index или sample_name")
        
        percentages = self.get_relative_contributions(method='percentage')
        sample_data = percentages.iloc[sample_index]
        
        if top_components is not None:
            sample_data = sample_data.nlargest(top_components)
        
        return sample_data.round(2)
    
    def export_loadings_to_csv(self, filename, normalization='percentage'):
        """
        Экспорт таблицы нагрузок в CSV файл
        
        Args:
            filename: имя файла
            normalization: метод нормализации
        """
        if normalization == 'percentage':
            df = self.get_relative_contributions(method='percentage')
        else:
            df = self.get_component_loadings(normalization=normalization)
        
        df.reset_index().rename(columns={df.columns[0]:"Sample"}).to_csv(filename, index=False)
        print(f"Таблица сохранена в {filename}")
    
    def get_component_eem_matrix(self, component_idx, sample_idx=None):
        """Получить EEM матрицу для компонента"""
        factors = self.factors
        emission = factors[1][:, component_idx]
        excitation = factors[2][:, component_idx]
        
        component_eem = np.outer(emission, excitation)
        
        if sample_idx is not None:
            sample_loading = factors[0][sample_idx, component_idx]
            component_eem = component_eem * sample_loading
        
        return component_eem
    
class ComponentVisualizer(OpticalDataAnalyzer):
    def __init__(self, n_components=3):
        super().__init__(n_components)
    
    def plot_single_component_eem(self, component_idx, sample_idx=None,
                                normalization='max', figsize=(7, 5)):
        """
        Построение EEM спектра для одного компонента
        
        Args:
            component_idx: индекс компонента (0-based)
            sample_idx: индекс образца для масштабирования (если None - используется максимальная нагрузка)
            normalization: метод нормализации ('max', 'none')
        """
        if not hasattr(self, 'factors'):
            raise ValueError("Сначала выполните PARAFAC анализ")
        
        # Получаем факторы
        factors = self.factors
        
        # Индексы для модов: 0-образцы, 1-испускание, 2-возбуждение
        sample_factor = factors[0]  # Нагрузки образцов
        emission_factor = factors[1]  # Спектр испускания
        excitation_factor = factors[2]  # Спектр возбуждения
        print(len(factors[2]))
        # Создаем EEM матрицу для компонента
        component_eem = np.outer(emission_factor[:, component_idx], 
                               excitation_factor[:, component_idx])
        
        scale_factor = 1.0
        
        if sample_idx is not None:
            # Масштабируем по конкретному образцу
            sample_loading = sample_factor[sample_idx, component_idx]
            component_eem = component_eem * sample_loading
        else:
            # Используем максимальную нагрузку для масштабирования
            max_loading = np.max(sample_factor[:, component_idx])
            component_eem = component_eem * scale_factor * max_loading
        
        # Нормализация
        if normalization == 'max':
            component_eem = component_eem / np.max(component_eem)
        
        # Создаем график
        fig, ax = plt.subplots(figsize=figsize)
        
        # Контурный график
        X, Y = np.meshgrid(self.excitation_wavelengths, 
                          self.emission_wavelengths)
        
        # Уровни для контурных линий
        levels = np.linspace(np.min(component_eem), np.max(component_eem), 20)
        
        # Контурный график
        contour = ax.contour(X, Y, component_eem, levels=levels, colors='black', linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        
        # Heatmap
        im = ax.contourf(X, Y, component_eem, levels=50, cmap='viridis')
        
        # Настройки осей
        ax.set_xlabel('Длина волны возбуждения, нм')
        ax.set_ylabel('Длина волны испускания, нм')
        
        sample_info = ""
        if sample_idx is not None:
            if self.sample_names:
                sample_info = f" (масштабировано для {self.sample_names[sample_idx]})"
            else:
                sample_info = f" (масштабировано для образца {sample_idx})"
        
        ax.set_title(f'EEM спектр компонента {component_idx+1}{sample_info}\n'
                    f'Нормализация: {normalization}')
        
        # Цветовая шкала
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Интенсивность (норм.)')
        
        return fig, ax, component_eem
    
    def plot_component_spectra(self, component_idx, figsize=(12, 6)):
        """Построение отдельных спектров возбуждения и испускания"""
        factors = self.factors
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Спектр испускания
        emission_spectrum = factors[1][:, component_idx]
        ax1.plot(self.emission_wavelengths, emission_spectrum, 
                'b-', linewidth=2, label=f'Компонент {component_idx+1}')
        ax1.set_xlabel('Длина волны испускания, нм')
        ax1.set_ylabel('Интенсивность (норм.)')
        ax1.set_title('Спектр испускания')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Спектр возбуждения
        excitation_spectrum = factors[2][:, component_idx]
        ax2.plot(self.excitation_wavelengths, excitation_spectrum,
                'r-', linewidth=2, label=f'Компонент {component_idx+1}')
        ax2.set_xlabel('Длина волны возбуждения, нм')
        ax2.set_ylabel('Интенсивность (норм.)')
        ax2.set_title('Спектр возбуждения')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        return fig
    
    def plot_component_profiles(self, figsize=(15, 10)):
        """
        Визуализация компонентов с реальными длинами волн
        """
        if not hasattr(self, 'factors'):
            raise ValueError("Сначала выполните fit_parafac()")
        
        fig, axes = plt.subplots(2, self.n_components, figsize=figsize)
        
        for i in range(self.n_components):
            # Спектры возбуждения
            if self.excitation_wavelengths is not None:
                x_exc = self.excitation_wavelengths
                xlabel_exc = 'Длина волны возбуждения (нм)'
            else:
                x_exc = np.arange(len(self.factors[2][:, i]))
                xlabel_exc = 'Индекс длины волны возбуждения'
            
            axes[0, i].plot(x_exc, self.factors[2][:, i], 'b-', linewidth=2)
            axes[0, i].set_title(f'Компонент {i+1} - Возбуждение')
            axes[0, i].set_xlabel(xlabel_exc)
            axes[0, i].set_ylabel('Интенсивность')
            axes[0, i].grid(True, alpha=0.3)
            
            # Находим и отмечаем максимум
            max_idx = np.argmax(self.factors[2][:, i])
            max_val = self.factors[2][max_idx, i]
            if self.excitation_wavelengths is not None:
                max_wavelength = self.excitation_wavelengths[max_idx]
                axes[0, i].axvline(x=max_wavelength, color='red', linestyle='--', alpha=0.7)
                axes[0, i].text(0.05, 0.95, f'λ_max = {max_wavelength:.1f} нм', 
                               transform=axes[0, i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Спектры испускания
            if self.emission_wavelengths is not None:
                x_em = self.emission_wavelengths
                xlabel_em = 'Длина волны испускания (нм)'
            else:
                x_em = np.arange(len(self.factors[1][:, i]))
                xlabel_em = 'Индекс длины волны испускания'
            
            axes[1, i].plot(x_em, self.factors[1][:, i], 'r-', linewidth=2)
            axes[1, i].set_title(f'Компонент {i+1} - Испускание')
            axes[1, i].set_xlabel(xlabel_em)
            axes[1, i].set_ylabel('Интенсивность')
            axes[1, i].grid(True, alpha=0.3)
            
            # Находим и отмечаем максимум
            max_idx_em = np.argmax(self.factors[1][:, i])
            max_val_em = self.factors[1][max_idx_em, i]
            if self.emission_wavelengths is not None:
                max_wavelength_em = self.emission_wavelengths[max_idx_em]
                axes[1, i].axvline(x=max_wavelength_em, color='red', linestyle='--', alpha=0.7)
                axes[1, i].text(0.05, 0.95, f'λ_max = {max_wavelength_em:.1f} нм', 
                               transform=axes[1, i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def plot_fraction_profiles(self, figsize=(12, 6)):
        """
        Визуализация концентраций с именами образцов
        """
        if not hasattr(self, 'factors'):
            raise ValueError("Сначала выполните fit_parafac()")
        
        fraction = self.factors[0]
        
        plt.figure(figsize=figsize)
        
        # Создаем ось X
        if self.sample_names is not None:
            x_pos = np.arange(len(self.sample_names))
            plt.xticks(x_pos, self.sample_names, rotation=45, ha='right')
        else:
            x_pos = np.arange(fraction.shape[0])
        
        # Рисуем доли для каждого компонента
        for i in range(self.n_components):
            plt.plot(x_pos, fraction[:, i], 'o-', linewidth=2, 
                    markersize=6, label=f'Компонент {i+1}')
        
        plt.xlabel('Образцы')
        plt.ylabel('Относительная доля')
        plt.title('Распределение компонентов по образцам')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    def plot_fraction_profiles_grouped(self, figsize=(14, 6),rotation=45):
        """
        Визуализация с группировкой по классам с сохранением всех точек
        """
        if not hasattr(self, 'factors'):
            raise ValueError("Сначала выполните fit_parafac()")
        
        fraction = self.factors[0]
        
        if self.sample_names is not None:
            classes = []
            for sample_name in self.sample_names:
                class_name = ut.extract_subclass_from_name(sample_name)
                classes.append(class_name)
            
            unique_classes = sorted(list(set(classes)))
            
            # Создаем позиции для группировки
            x_pos = []
            current_pos = 0
            class_positions = {}
            
            for class_name in unique_classes:
                class_indices = [i for i, cls in enumerate(classes) if cls == class_name]
                class_size = len(class_indices)
                class_x = np.arange(current_pos, current_pos + class_size)
                x_pos.extend(class_x)
                class_positions[class_name] = (current_pos, current_pos + class_size - 1)
                current_pos += class_size + 1  # +1 для пробела между группами
            
            plt.figure(figsize=figsize)
            
            # Рисуем линии для каждого компонента
            for i in range(self.n_components):
                plt.plot(x_pos, fraction[:, i], 'o-', linewidth=1, 
                        markersize=4, label=f'Компонент {i+1}', alpha=0.7)
            
            # Добавляем разделители и подписи классов
            for class_name, (start, end) in class_positions.items():
                middle = (start + end) / 2
                plt.axvline(x=end + 0.5, color='gray', linestyle='--', alpha=0.5)
                plt.text(middle, plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]), 
                        class_name, ha='center', va='top',rotation=rotation)
                
            
            plt.ylabel('Относительная доля')
            plt.title('Распределение компонентов по образцам')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
    def plot_eem_contours(self, sample_idx=0, n_levels=20, figsize=(8, 6)):
        """
        Контурный график EEM для выбранного образца
        """
        if not hasattr(self, 'factors'):
            raise ValueError("Сначала выполните fit_parafac()")
        
        # Реконструируем EEM для выбранного образца
        sample_eem = self.reconstruct_sample(sample_idx)
        
        plt.figure(figsize=figsize)
        
        # Контурный график
        if self.excitation_wavelengths is not None and self.emission_wavelengths is not None:
            X, Y = np.meshgrid(self.excitation_wavelengths, self.emission_wavelengths)
            contour = plt.contourf(X, Y, sample_eem, levels=n_levels, cmap='viridis')
            plt.colorbar(contour, label='Интенсивность флуоресценции')
            plt.xlabel('Длина волны возбуждения (нм)')
            plt.ylabel('Длина волны испускания (нм)')
            
            sample_name = self.sample_names[sample_idx] if self.sample_names else f"Образец {sample_idx+1}"
            plt.title(f'Контурный график EEM: {sample_name}')
        else:
            plt.imshow(sample_eem, aspect='auto', cmap='viridis')
            plt.colorbar(label='Интенсивность флуоресценции')
            plt.xlabel('Индекс возбуждения')
            plt.ylabel('Индекс испускания')
            plt.title(f'EEM матрица: Образец {sample_idx+1}')
        
    def plot_component_loadings(self, normalization='percentage', figsize=(12, 6), group_by_subclass=True):
        """
        Визуализация нагрузок компонентов с группировкой по классам
        
        Args:
            normalization: метод нормализации
            figsize: размер фигуры
            group_by_subclass: если True, группирует образцы по подклассам
        """
        
        loadings_df = self.get_component_loadings(normalization=normalization)
        
        if group_by_subclass and 'Subclass' in loadings_df.columns:
            # Группируем по подклассам
            subclass_groups = loadings_df.groupby('Subclass')
            
            # Вычисляем средние значения для каждого подкласса
            component_columns = [col for col in loadings_df.columns if col.startswith('Component_')]
            grouped_data = subclass_groups[component_columns].mean()
            
            # Создаем фигуру с двумя subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] + 4))
            
            # 1. Столбчатая диаграмма для группированных данных
            grouped_data.plot(kind='bar', stacked=True, ax=ax1)
            ax1.set_ylabel('Доля компонента, %')
            ax1.set_title('Относительные вклады компонентов в подклассы (средние значения)')
            ax1.legend(title='Компоненты', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Box plot для распределения по подклассам
            melted_data = loadings_df.melt(id_vars=['Subclass'], 
                                        value_vars=component_columns,
                                        var_name='Component', 
                                        value_name='Loading')
            
            sns.boxplot(data=melted_data, x='Subclass', y='Loading', hue='Component', ax=ax2)
            ax2.set_ylabel('Доля компонента, %')
            ax2.set_title('Распределение вкладов компонентов по подклассам')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            
        else:
            # Исходный вариант без группировки
            plt.figure(figsize=figsize)
            
            # Столбчатая диаграмма для процентных вкладов
            component_columns = [col for col in loadings_df.columns if col.startswith('Component_')]
            loadings_df[component_columns].plot(kind='bar', stacked=True)
            plt.ylabel('Доля компонента, %')
            plt.title('Относительные вклады компонентов в образцы')
            
            plt.xticks(rotation=45)
            plt.legend(title='Компоненты', bbox_to_anchor=(1.05, 1), loc='upper left')
         
    def plot_all_components_eem(self):
            """Построение EEM для всех компонентов"""
            fig, axes = plt.subplots(self.n_components, 1, figsize=(10, 16))
            
            if self.n_components == 1:
                axes = [axes]
            
            for i in range(self.n_components):
                component_eem = self.get_component_eem_matrix(i)
                component_eem = component_eem / np.max(component_eem)
                
                X, Y = np.meshgrid(self.excitation_wavelengths, 
                                self.emission_wavelengths)
                
                im = axes[i].contourf(X, Y, component_eem, levels=50, cmap='viridis')
                axes[i].contour(X, Y, component_eem, levels=10, colors='black', linewidths=0.3)
                
                axes[i].set_title(f'Component {i+1}')
                axes[i].set_xlabel('Excitation (nm)')
                axes[i].set_ylabel('Emission (nm)')
                
                plt.colorbar(im, ax=axes[i])
            
            return fig
    