import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil, sqrt
from pandas import DataFrame
from typing import List

from HumSpectra.ultraviolet import plot_uv

def plot_uv_spectra_by_subclass(spectra_list: List[DataFrame], 
                           figsize_multiplier: int = 4,
                           sharey: bool = True, 
                           sharex: bool = True,
                           norm_by_TOC: bool = False,
                           show_titles: bool = True,
                           show_xlabels: bool = True,
                           show_ylabels: bool = True) -> None:
    """
    Отображает спектры по подклассам на отдельных графиках с использованием plot_uv.
    
    Parameters:
    -----------
    spectra_list : list of DataFrame
        Список спектров (DataFrame) с атрибутами 'name' и 'subclass' в .attrs
    figsize_multiplier : int, default=4
        Множитель для размера фигуры
    sharey : bool, default=True
        Общий масштаб по оси Y для всех подграфиков
    sharex : bool, default=True
        Общий масштаб по оси X для всех подграфиков
    norm_by_TOC : bool, default=False
        Нормализовать по TOC (передается в plot_uv)
    show_titles : bool, default=True
        Показывать заголовки графиков
    show_xlabels : bool, default=True
        Показывать подписи оси X
    show_ylabels : bool, default=True
        Показывать подписи оси Y
    """
    
    # Группируем спектры по подклассам
    spectra_by_subclass = {}
    for spectrum in spectra_list:
        subclass = spectrum.attrs.get('subclass', 'Unknown')
        name = spectrum.attrs.get('name', 'Unnamed')
        
        if subclass not in spectra_by_subclass:
            spectra_by_subclass[subclass] = []
        
        spectra_by_subclass[subclass].append({
            'data': spectrum,
            'name': name
        })
    
    # Получаем список подклассов
    subclasses = list(spectra_by_subclass.keys())
    n_subclasses = len(subclasses)
    
    if n_subclasses == 0:
        print("Нет спектров для отображения")
        return
    
    # Определяем оптимальную размерность subplots
    n_cols = ceil(sqrt(n_subclasses))
    n_rows = ceil(n_subclasses / n_cols)
    
    # Создаем фигуру с оптимальным размером
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * figsize_multiplier, n_rows * figsize_multiplier),
                            sharey=sharey, sharex=sharex,
                            squeeze=False)
    
    # Выравниваем axes в плоский массив для удобства итерации
    axes_flat = axes.flatten()
    
    # Отрисовываем спектры для каждого подкласса с использованием plot_uv
    for idx, subclass in enumerate(subclasses):
        ax = axes_flat[idx]
        spectra_data = spectra_by_subclass[subclass]
        
        # Рисуем все спектры этого подкласса
        for spectrum_info in spectra_data:
            spectrum_df = spectrum_info['data']
            name = spectrum_info['name']
            
            # Используем функцию plot_uv для отрисовки каждого спектра
            plot_uv(data=spectrum_df,
                   xlabel=False,  # Убираем xlabel для отдельных графиков
                   ylabel=False,  # Убираем ylabel для отдельных графиков
                   title=False,   # Убираем title для отдельных графиков
                   norm_by_TOC=norm_by_TOC,
                   ax=ax,
                   name=name)
        
        # Добавляем заголовок подкласса
        if show_titles:
            ax.set_title(f'Подкласс: {subclass}\n(спектров: {len(spectra_data)})', 
                        fontsize=12, fontweight='bold')
        
        # Добавляем подписи осей только если нужно
        if show_xlabels:
            ax.set_xlabel("λ поглощения, нм")
        if show_ylabels:
            if norm_by_TOC:
                ax.set_ylabel("SUVA, $см^{-1}*мг^{-1}*л$")
            else:
                ax.set_ylabel("Интенсивность")
        
        ax.grid(True, alpha=0.3)
        
        # Настраиваем легенду в зависимости от количества спектров
        if len(spectra_data) <= 8:
            ax.legend(fontsize=8)
        else:
            # Для большого количества спектров уменьшаем шрифт или выносим легенду
            ax.legend(fontsize=6, loc='upper right')
    
    # Скрываем пустые subplots
    for idx in range(len(subclasses), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Выводим статистику
    print(f"Всего подклассов: {n_subclasses}")
    for subclass, spectra in spectra_by_subclass.items():
        print(f"  {subclass}: {len(spectra)} спектров")