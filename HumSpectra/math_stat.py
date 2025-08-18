import pandas as pd
from pandas import DataFrame

def delete_eject(data: DataFrame,
                 iqr_param: int = 1.5,
                 level_index: int = 0) -> DataFrame:
    """
    :param data: DataFrame
    :param iqr_param: Межквартильный множитель
    :param level_index: Уровень индекса, по которому данные группируются и удаляются выбросы
    :return: Отфильтргованная таблица
    Функция приписывает имя, класс и подкласс (если не игнорируется) спектру
    """
    data_copy = data.copy()

    descriptor_list = data_copy.columns
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