import pandas as pd
import numpy as np

def analyze_geographical(data):
    """
    Анализирует географические координаты и определяет крайние точки.
    
    Parameters:
    -----------
    data : pandas.DataFrame или pandas.Series
        Если DataFrame, должен содержать столбцы с координатами.
        Если Series, должен содержать координаты в формате (lat, lon) или два столбца.
    
    Returns:
    --------
    dict: Словарь с информацией о крайних точках
    """
    

    df = data.copy()

    
    # Проверяем наличие столбцов с координатами
    lat_col = None
    lon_col = None
    
    # Ищем столбцы с координатами по разным возможным названиям
    possible_lat_names = ['latitude', 'lat', 'y', 'широта']
    possible_lon_names = ['longitude', 'lon', 'lng', 'x', 'долгота']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_lat_names):
            lat_col = col
        if any(name in col_lower for name in possible_lon_names):
            lon_col = col
    
    # Если не нашли стандартные названия, используем первые два столбца
    if lat_col is None or lon_col is None:
        if len(df.columns) >= 2:
            lat_col, lon_col = df.columns[0], df.columns[1]
        else:
            return {"error": "Не удалось определить столбцы с координатами"}
    
    # Проверяем на наличие пропущенных значений
    if df[[lat_col, lon_col]].isnull().any().any():
        df = df.dropna(subset=[lat_col, lon_col])
    
    if len(df) == 0:
        return {"error": "Нет данных для анализа"}
    
    # Находим крайние точки
    northernmost = df.loc[df[lat_col].idxmax()]
    southernmost = df.loc[df[lat_col].idxmin()]
    easternmost = df.loc[df[lon_col].idxmax()]
    westernmost = df.loc[df[lon_col].idxmin()]

    # Находим среднее значение широты и от него опредляем северные и южные

    mean_lat = df[lat_col].mean()
    df["North"] = df[lat_col].apply(lambda x: True if x > mean_lat else False)
    df["South"] = df[lat_col].apply(lambda x: False if x > mean_lat else True)

    mean_lon = df[lon_col].mean()
    df["West"] = df[lon_col].apply(lambda x: True if x < mean_lon else False)
    df["East"] = df[lon_col].apply(lambda x: True if x > mean_lon else False)   
    
    # Находим угловые точки (северо-запад, северо-восток, юго-запад, юго-восток)
    # Для этого используем комбинации экстремальных значений
    
    # Северо-запад: максимальная широта, минимальная долгота
    northwest_mask = (df[lat_col] == df[lat_col].max()) & (df[lon_col] == df[lon_col].min())
    if northwest_mask.any():
        northwest = df[northwest_mask].iloc[0]
    else:
        # Если нет точного совпадения, находим ближайшую точку
        df['nw_distance'] = (df[lat_col] - df[lat_col].max())**2 + (df[lon_col] - df[lon_col].min())**2
        northwest = df.loc[df['nw_distance'].idxmin()]
        df = df.drop('nw_distance', axis=1)
    
    # Северо-восток: максимальная широта, максимальная долгота
    northeast_mask = (df[lat_col] == df[lat_col].max()) & (df[lon_col] == df[lon_col].max())
    if northeast_mask.any():
        northeast = df[northeast_mask].iloc[0]
    else:
        df['ne_distance'] = (df[lat_col] - df[lat_col].max())**2 + (df[lon_col] - df[lon_col].max())**2
        northeast = df.loc[df['ne_distance'].idxmin()]
        df = df.drop('ne_distance', axis=1)
    
    # Юго-запад: минимальная широта, минимальная долгота
    southwest_mask = (df[lat_col] == df[lat_col].min()) & (df[lon_col] == df[lon_col].min())
    if southwest_mask.any():
        southwest = df[southwest_mask].iloc[0]
    else:
        df['sw_distance'] = (df[lat_col] - df[lat_col].min())**2 + (df[lon_col] - df[lon_col].min())**2
        southwest = df.loc[df['sw_distance'].idxmin()]
        df = df.drop('sw_distance', axis=1)
    
    # Юго-восток: минимальная широта, максимальная долгота
    southeast_mask = (df[lat_col] == df[lat_col].min()) & (df[lon_col] == df[lon_col].max())
    if southeast_mask.any():
        southeast = df[southeast_mask].iloc[0]
    else:
        df['se_distance'] = (df[lat_col] - df[lat_col].min())**2 + (df[lon_col] - df[lon_col].max())**2
        southeast = df.loc[df['se_distance'].idxmin()]
        df = df.drop('se_distance', axis=1)

    return df
    
