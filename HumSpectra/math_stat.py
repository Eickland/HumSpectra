import pandas as pd
from pandas import DataFrame
import numpy as np

def delete_eject_iqr(data: DataFrame,
                 iqr_param: float = 1.5,
                 level_index: int = 0,
                 multi_index: bool = False) -> DataFrame:
    """
    :param data: DataFrame
    :param iqr_param: Межквартильный множитель
    :param level_index: Уровень индекса, по которому данные группируются и удаляются выбросы
    :return: Отфильтргованная таблица
    """
    data_copy = data.copy()

    descriptor_list = data_copy.columns

    if multi_index:

        class_list = data_copy.index.unique(level=level_index)

        for descriptor in descriptor_list:

            for gen_class in class_list:

                data_iqr = data_copy.loc[gen_class]

                if data_iqr.shape[0] < 5:
                    continue

                q1 = data_iqr[descriptor].quantile(0.25)
                q3 = data_iqr[descriptor].quantile(0.75)
                iqr = q3 - q1

                data_iqr = data_iqr[(data_iqr[descriptor] < q3 + iqr_param * iqr)]
                data_iqr = data_iqr[(data_iqr[descriptor] > q1 - iqr_param * iqr)]

                data_iqr = pd.concat({gen_class: data_iqr}, names=['Класс'])

                data_copy.loc[gen_class] = data_iqr
                
                data_copy.dropna(inplace=True)

        return data_copy
    
    else:

        for descriptor in descriptor_list:

                data_iqr = data_copy

                if data_iqr.shape[0] < 5:
                    continue

                q1 = data_iqr[descriptor].quantile(0.25)
                q3 = data_iqr[descriptor].quantile(0.75)
                iqr = q3 - q1

                data_iqr = data_iqr[(data_iqr[descriptor] < q3 + iqr_param * iqr)]
                data_iqr = data_iqr[(data_iqr[descriptor] > q1 - iqr_param * iqr)]
                
                data_copy.dropna(inplace=True)

        return data_copy


def delete_eject_quantile(data: DataFrame,
                            quant: float=0.995)-> DataFrame:

    """
    Удаляет экстремальные выбросы из 3D матрицы флуоресценции.

    Args:
        data (pd.DataFrame): 3D спектр флуоресценции.
        quant (float): Квантиль для определения выбросов.

    Returns:
        pd.DataFrame: Спектр с удаленными выбросами
    """
    data = data.copy()

    data[data > np.quantile(data,quant)] = 0

    return data

def normilize_by_max(data: DataFrame)-> DataFrame:

    """
    Нормирует спектр от 0 до 1

    Args:
        data (pd.DataFrame): 3D спектр флуоресценции.

    Returns:
        pd.DataFrame: Отнормированый спектр
    """

    data = data.copy()

    data = data/data.max(axis=None)

    return data


