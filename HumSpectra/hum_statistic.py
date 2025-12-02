import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
from typing import Optional, Union, List, Tuple
import base64
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import HumSpectra.utilits as ut

def kmeans_analysis_with_subclasses(df, n_clusters=None, random_state=42, output_html_path=None, save=False):
    """
    Кластеризация данных с многоуровневым индексом и анализ подклассов
    с выводом результатов в HTML файл
    
    Parameters:
    df - DataFrame с многоуровневым индексом
    n_clusters - количество кластеров (если None, подбирается автоматически)
    random_state - для воспроизводимости
    output_html_path - путь для сохранения HTML отчета
    """
    
    # Перехватываем вывод в консоль
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Сохраняем оригинальные индексы
        original_index = df.index
        subclasses = df.index.get_level_values(1).unique()  # Все уникальные подклассы
        
        print("=" * 60)
        print("АНАЛИЗ КЛАСТЕРИЗАЦИИ С ПОДКЛАССАМИ")
        print("=" * 60)
        
        # Шаг 1: Предобработка данных
        print("1. Предобработка данных...")
        
        # Удаляем нечисловые колонки если есть
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].copy()
        
        # Проверяем наличие пропущенных значений
        if df_numeric.isnull().sum().sum() > 0:
            print(f"   Обнаружено пропущенных значений: {df_numeric.isnull().sum().sum()}")
            df_numeric = df_numeric.fillna(df_numeric.mean())
        
        # Масштабирование признаков
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_numeric)
        
        print(f"   Размерность данных: {data_scaled.shape}")
        print(f"   Количество признаков: {data_scaled.shape[1]}")
        
        # Шаг 2: Определение оптимального числа кластеров (если не задано)
        if n_clusters is None:
            print("2. Подбор оптимального числа кластеров...")
            best_k = 2
            best_silhouette = -1
            
            for k in range(2, min(11, len(df) // 2)):
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(data_scaled)
                silhouette_avg = silhouette_score(data_scaled, cluster_labels)
                
                print(f"   k={k}: Silhouette Score = {silhouette_avg:.4f}")
                
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_k = k
            
            n_clusters = best_k
            print(f"   Оптимальное число кластеров: {n_clusters} (silhouette: {best_silhouette:.4f})")
        else:
            print(f"2. Используется заданное число кластеров: {n_clusters}")
        
        # Шаг 3: Кластеризация K-Means
        print("3. Выполнение кластеризации K-Means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        # Добавляем метки кластеров к данным
        result_df = df.copy()
        result_df['cluster'] = cluster_labels
        
        # Шаг 4: Анализ распределения подклассов по кластерам
        print("4. Анализ распределения подклассов по кластерам...")
        
        # Создаем таблицу сопряженности: подклассы vs кластеры
        contingency_table = pd.crosstab(
            index=result_df.index.get_level_values(1),  # подклассы
            columns=result_df['cluster'],
            margins=True,
            margins_name='Всего'
        )
        
        # Процентное распределение по строкам (какой % подкласса попал в каждый кластер)
        percentage_table = pd.crosstab(
            index=result_df.index.get_level_values(1),
            columns=result_df['cluster'],
            normalize='index'
        ) * 100
        
        # Шаг 5: Вывод результатов
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ")
        print("=" * 60)
        
        print(f"\nОбщая информация:")
        print(f"- Всего наблюдений: {len(result_df)}")
        print(f"- Количество кластеров: {n_clusters}")
        print(f"- Количество подклассов: {len(subclasses)}")
        print(f"- Silhouette Score: {silhouette_score(data_scaled, cluster_labels):.4f}")
        
        print(f"\nРаспределение по кластерам:")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"  Кластер {cluster}: {count} наблюдений ({count/len(result_df)*100:.1f}%)")
        
        print(f"\nТАБЛИЦА СОПРЯЖЕННОСТИ: Подклассы × Кластеры")
        print("-" * 50)
        print(contingency_table)
        
        print(f"\nПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ (% по строкам)")
        print("-" * 50)
        print(percentage_table.round(1))
        
        # Анализ "чистоты" кластеров
        print(f"\nАНАЛИЗ 'ЧИСТОТЫ' КЛАСТЕРОВ:")
        print("-" * 50)
        
        cluster_summary = []
        for cluster in range(n_clusters):
            cluster_data = result_df[result_df['cluster'] == cluster]
            cluster_subclasses = cluster_data.index.get_level_values(1)
            
            if len(cluster_data) > 0:
                # Самый частый подкласс в кластере
                subclass_counts = cluster_subclasses.value_counts()
                top_subclass = subclass_counts.index[0]
                top_count = subclass_counts.iloc[0]
                top_percentage = (top_count / len(cluster_data)) * 100
                
                # Количество уникальных подклассов в кластере
                unique_subclasses = cluster_subclasses.nunique()
                
                cluster_summary.append({
                    'Кластер': cluster,
                    'Наблюдений': len(cluster_data),
                    'Уникальных подклассов': unique_subclasses,
                    'Доминирующий подкласс': top_subclass,
                    'Доля доминирующего': f"{top_percentage:.1f}%"
                })
        
        cluster_summary_df = pd.DataFrame(cluster_summary)
        print(cluster_summary_df.to_string(index=False))
        
        # Дополнительный анализ характеристик кластеров
        print(f"\nХАРАКТЕРИСТИКИ КЛАСТЕРОВ (средние значения):")
        print("-" * 60)
        
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns[numeric_columns != 'cluster']  # Исключаем колонку с кластерами
        
        cluster_means = result_df.groupby('cluster')[numeric_columns].mean()
        print(cluster_means.round(3))
        
        # Маппинг подклассов на основные кластеры
        print(f"\nМАППИНГ ПОДКЛАССОВ НА ОСНОВНЫЕ КЛАСТЕРЫ:")
        print("-" * 50)
        
        subclass_cluster_map = {}
        for subclass in result_df.index.get_level_values(1).unique():
            subclass_data = result_df.xs(subclass, level=1)
            if len(subclass_data) > 0:
                # Определяем основной кластер для подкласса
                main_cluster = subclass_data['cluster'].mode()
                if len(main_cluster) > 0:
                    subclass_cluster_map[subclass] = main_cluster[0]
                    print(f"  {subclass} → Кластер {main_cluster[0]}")
        
        # Получаем весь вывод
        console_output = captured_output.getvalue()
        
        # Восстанавливаем stdout
        sys.stdout = old_stdout
        
        if save:
            # Создаем HTML отчет
            create_html_report(console_output, result_df, contingency_table, 
                            percentage_table, cluster_summary_df, cluster_means, 
                            subclass_cluster_map, n_clusters, output_html_path)
            
            print(f"\n✅ HTML отчет сохранен в файл: {output_html_path}")
        
        return result_df, kmeans, scaler, contingency_table, percentage_table, subclass_cluster_map
        
    except Exception as e:
        # Восстанавливаем stdout в случае ошибки
        sys.stdout = old_stdout
        print(f"Ошибка при выполнении анализа: {e}")
        raise

def create_html_report(console_output, result_df, contingency_table, percentage_table, 
                      cluster_summary_df, cluster_means, subclass_cluster_map, 
                      n_clusters, output_html_path):
    """Создание HTML отчета с результатами анализа"""
    
    # Создаем стилизованный HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Анализ кластеризации с подклассами</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #ffffff; /* БЕЛЫЙ ФОН */
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.05); /* Более легкая тень */
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background: #ffffff; /* БЕЛЫЙ ФОН СЕКЦИЙ */
                border-radius: 8px;
                border-left: 4px solid #667eea;
                border: 1px solid #e0e0e0; /* Добавляем границу */
            }}
            .section h2 {{
                color: #333;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .console-output {{
                background: #f8f9fa; /* Светлый фон вместо темного */
                color: #333; /* Темный текст */
                padding: 20px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                overflow-x: auto;
                font-size: 14px;
                border: 1px solid #ddd;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
                border: 1px solid #ddd;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                background: white;
            }}
            tr:nth-child(even) {{
                background: #f8f9fa;
            }}
            .highlight {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .summary-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }}
            .summary-card h3 {{
                margin: 0;
                color: #667eea;
                font-size: 24px;
            }}
            .summary-card p {{
                margin: 5px 0 0 0;
                color: #666;
            }}
            .timestamp {{
                text-align: right;
                color: #888;
                font-style: italic;
                margin-top: 30px;
            }}
            /* Улучшения для таблиц */
            .dataframe {{
                width: 100%;
                border-collapse: collapse;
            }}
            .dataframe th {{
                background: #667eea;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Анализ кластеризации с подклассами</h1>
                <p>Автоматический отчет по результатам кластеризации K-Means</p>
            </div>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>{len(result_df)}</h3>
                    <p>Всего наблюдений</p>
                </div>
                <div class="summary-card">
                    <h3>{n_clusters}</h3>
                    <p>Кластеров</p>
                </div>
                <div class="summary-card">
                    <h3>{len(result_df.index.get_level_values(1).unique())}</h3>
                    <p>Уникальных подклассов</p>
                </div>
                <div class="summary-card">
                    <h3>{result_df.select_dtypes(include=[np.number]).columns.nunique()}</h3>
                    <p>Признаков</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Полный вывод анализа</h2>
                <div class="console-output">{console_output}</div>
            </div>
            
            <div class="section">
                <h2>🔍 Таблица сопряженности: Подклассы × Кластеры</h2>
                {contingency_table.to_html(classes='dataframe', border=0, index=True)}
            </div>
            
            <div class="section">
                <h2>📈 Процентное распределение подклассов</h2>
                <p><em>Проценты показывают распределение каждого подкласса по кластерам</em></p>
                {percentage_table.round(1).to_html(classes='dataframe', border=0, index=True)}
            </div>
            
            <div class="section">
                <h2>🎯 Анализ чистоты кластеров</h2>
                {cluster_summary_df.to_html(classes='dataframe', border=0, index=False)}
            </div>
            
            <div class="section">
                <h2>📊 Характеристики кластеров</h2>
                <p><em>Средние значения признаков по кластерам</em></p>
                {cluster_means.round(3).to_html(classes='dataframe', border=0, index=True)}
            </div>
            
            <div class="section">
                <h2>🗺️ Маппинг подклассов на кластеры</h2>
                <div class="highlight">
    """
    
    # Добавляем маппинг подклассов
    for subclass, cluster in subclass_cluster_map.items():
        html_content += f"<p><strong>{subclass}</strong> → Основной кластер <strong>{cluster}</strong></p>\n"
    
    html_content += f"""
                </div>
            </div>
            
            <div class="timestamp">
                Отчет сгенерирован: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем HTML файл
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def analyze_cluster_characteristics(result_df, n_clusters):
    """Анализ характеристик кластеров"""
    print(f"\nХАРАКТЕРИСТИКИ КЛАСТЕРОВ (средние значения):")
    print("-" * 60)
    
    numeric_columns = result_df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns[numeric_columns != 'cluster']  # Исключаем колонку с кластерами
    
    cluster_means = result_df.groupby('cluster')[numeric_columns].mean()
    print(cluster_means.round(3))

def get_subclass_cluster_mapping(result_df):
    """Получить маппинг подклассов на основные кластеры"""
    subclass_cluster_map = {}
    
    for subclass in result_df.index.get_level_values(1).unique():
        subclass_data = result_df.xs(subclass, level=1)
        if len(subclass_data) > 0:
            # Определяем основной кластер для подкласса
            main_cluster = subclass_data['cluster'].mode()
            if len(main_cluster) > 0:
                subclass_cluster_map[subclass] = main_cluster[0]
    
    return subclass_cluster_map

def random_forest_classification(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    output_html_path: str|None = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    index_level: Optional[int] = None,
    **kwargs
) -> Tuple[pd.DataFrame, RandomForestClassifier, StandardScaler, pd.DataFrame,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, float]:
    """
    Анализ данных с помощью Random Forest для задач классификации.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Входной DataFrame с данными
    target_column : str, optional
        Название целевой колонки. Если None, используется index_level индекс
    test_size : float
        Доля тестовой выборки
    random_state : int
        Seed для воспроизводимости
    output_html_path : str
        Путь для сохранения HTML отчета
    n_estimators : int
        Количество деревьев в Random Forest
    max_depth : int, optional
        Максимальная глубина деревьев
    save_report : bool
        Сохранять ли HTML отчет
    index_level : int
        Уровень индекса для использования в качестве цели (если target_column=None)
    
    Returns:
    --------
    Tuple containing:
    - results_df: DataFrame с результатами предсказаний
    - rf_model: обученная модель Random Forest
    - scaler: обученный скейлер
    - feature_importance: важность признаков
    - X_train, X_test, y_train, y_test: разделенные данные
    - label_encoder: обученный LabelEncoder
    - accuracy: точность модели
    """
    
    # Сохраняем оригинальный stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        print("=" * 70)
        print("RANDOM FOREST АНАЛИЗ КЛАССИФИКАЦИИ")
        print("=" * 70)
        
        # 1. ПОДГОТОВКА ДАННЫХ
        print("\n1. ПОДГОТОВКА ДАННЫХ")
        print("-" * 40)
        
        # Определяем целевую переменную
        # Определяем целевую переменную
        if target_column is None and index_level:
            
            # Используем уровень индекса как целевую переменную
            target = data.index.get_level_values(index_level)
            
            features_df = data.reset_index(drop=True)
            target_name = f"{index_level} уровень индекса"
            
        else:
            
            target = data[target_column]
            features_df = data.drop(columns=[target_column])
            target_name = target_column
        
        print(f"   Целевая переменная: {target_name}")
        print(f"   Уникальных значений в целевой переменной: {target.nunique()}")
        
        # Проверяем, что задача действительно классификация
        unique_values = target.nunique()
        print(f"   Уникальных классов: {unique_values}")
        
        if unique_values < 2:
            raise ValueError("Для классификации необходимо минимум 2 класса")
        
        # Кодируем целевую переменную
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(target)
        class_names = label_encoder.classes_
        
        print(f"   Классы: {list(class_names)}")
        print(f"   Размер датасета: {len(data)} наблюдений")
        
        # 2. ПРЕДОБРАБОТКА ПРИЗНАКОВ
        print("\n2. ПРЕДОБРАБОТКА ПРИЗНАКОВ")
        print("-" * 40)
        
        # Выбираем только числовые признаки
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            raise ValueError("Не найдено числовых признаков для обучения")
        
        features_numeric = features_df[numeric_columns].copy()
        print(f"   Числовых признаков: {len(numeric_columns)}")
        
        # Обработка пропущенных значений
        missing_values = features_numeric.isnull().sum().sum()
        
        if missing_values > 0:
            
            print(f"   Заполняем {missing_values} пропущенных значений средними")
            features_numeric = features_numeric.fillna(features_numeric.mean())
        
        # Масштабирование признаков
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_numeric)
        print(f"   Размерность данных: {X_scaled.shape}")
        
        # 3. РАЗДЕЛЕНИЕ НА ВЫБОРКИ
        print("\n3. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
        print("-" * 40)
        
        # Проверяем распределение классов для стратификации
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        
        print("   Распределение классов:")
        for cls, count, name in zip(unique_classes, class_counts, class_names):
            print(f"      {name} (класс {cls}): {count} наблюдений")
        
        # Определяем стратификацию
        min_samples = class_counts.min()
        
        if min_samples < 2:
            
            print("   ⚠️  Некоторые классы имеют <2 наблюдений, стратификация отключена")
            stratify = None
            
        else:
            stratify = y_encoded
        
        # Разделение данных
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X_scaled, y_encoded, features_df.index,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        print(f"   Обучающая выборка: {X_train.shape[0]} наблюдений")
        print(f"   Тестовая выборка: {X_test.shape[0]} наблюдений")
        
        # 4. ОБУЧЕНИЕ МОДЕЛИ
        print("\n4. ОБУЧЕНИЕ RANDOM FOREST")
        print("-" * 40)
        
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        
        print(f"   Модель обучена: {n_estimators} деревьев")
        
        # 5. ОЦЕНКА МОДЕЛИ
        print("\n5. ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
        print("-" * 40)
        
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Точность (Accuracy): {accuracy:.4f}")
        
        # Детальный отчет по классам
        print("\n   Отчет по классификации:")
        print("   " + "-" * 35)
        
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=False,
            zero_division=0
        )
        
        # Выводим отчет построчно для лучшего форматирования
        if isinstance(report, str):
            for line in report.split('\n'):
                print(f"   {line}")
        
        # 6. ВАЖНОСТЬ ПРИЗНАКОВ
        print("\n6. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("-" * 40)
        
        feature_importance = pd.DataFrame({
            'Признак': numeric_columns,
            'Важность': rf_model.feature_importances_
        }).sort_values('Важность', ascending=False)
        
        print("   Топ-10 важнейших признаков:")
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"      {i:2d}. {row['Признак']:30s}: {row['Важность']:.4f}")
            
        
        # Создаем DataFrame с результатами
        results_df = features_df.copy()
        
        results_df['Истинный_класс'] = y_encoded
        
        results_df['Предсказанный_класс'] = rf_model.predict(X_scaled)
        
        results_df['Верно_предсказано'] = (
            results_df['Истинный_класс'] == results_df['Предсказанный_класс']
        )
        
        # Общая точность на всем датасете
        overall_accuracy = results_df['Верно_предсказано'].mean()
        print(f"   Общая точность на всем датасете: {overall_accuracy:.4f}")
        
        # 7. СВОДКА
        print("\n" + "=" * 70)
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 70)
        print(f"   Классов: {len(class_names)}")
        print(f"   Признаков: {len(numeric_columns)}")
        print(f"   Точность на тесте: {accuracy:.4f}")
        print(f"   Самый важный признак: {feature_importance.iloc[0]['Признак']}")
        print("=" * 70)
        
        # Получаем вывод консоли
        console_output = captured_output.getvalue()
        
        # Восстанавливаем stdout
        sys.stdout = old_stdout
        
        # Выводим результаты в консоль
        print(console_output)
        
        # Создание HTML отчета (опционально)
        if output_html_path:
            try:
                create_rf_classification_html_report(
                    console_output=console_output,
                    results_df=results_df,
                    feature_importance=feature_importance,
                    rf_model=rf_model,
                    class_names=class_names,
                    label_encoder=label_encoder,
                    X_test=X_test,
                    y_test=y_test,
                    y_pred=y_pred,
                    output_html_path=output_html_path
                )
                print(f"\n✅ HTML отчет сохранен: {output_html_path}")
            except Exception as e:
                print(f"⚠️  Не удалось создать HTML отчет: {e}")
        
        return (
            results_df,
            rf_model,
            scaler,
            feature_importance,
            X_train, X_test, y_train, y_test,
            label_encoder,
            float(accuracy)
        )
        
    except Exception as e:
        # Восстанавливаем stdout в случае ошибки
        sys.stdout = old_stdout
        print(f"❌ Ошибка при выполнении анализа: {e}")
        raise

def create_rf_classification_html_report(console_output, results_df, feature_importance, 
                         rf_model, class_names, label_encoder,
                         X_test, y_test, y_pred, output_html_path,problem_type = 'classification'):
    """Создание HTML отчета с результатами анализа Random Forest"""
    
    # Создаем визуализации
    plt.style.use('default')
    
    # График важности признаков
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(15)
    ax1.barh(top_features['Признак'], top_features['Важность'])
    ax1.set_xlabel('Важность признака')
    ax1.set_title('Топ самых важных признаков')
    ax1.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем график в base64
    from io import BytesIO
    import base64
    
    buffer1 = BytesIO()
    fig1.savefig(buffer1, format='png', dpi=100, bbox_inches='tight')
    buffer1.seek(0)
    feature_importance_plot = base64.b64encode(buffer1.getvalue()).decode()
    plt.close(fig1)
    
    # Матрица ошибок для классификации
    confusion_matrix_plot = ""
    
    if problem_type == 'classification':
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        ax2.set_xlabel('Предсказанные')
        ax2.set_ylabel('Фактические')
        ax2.set_title('Матрица ошибок')
        
        plt.tight_layout()
        
        buffer2 = BytesIO()
        fig2.savefig(buffer2, format='png', dpi=100, bbox_inches='tight')
        buffer2.seek(0)
        confusion_matrix_plot = base64.b64encode(buffer2.getvalue()).decode()
        plt.close(fig2)
    
    # Создаем стилизованный HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Анализ Random Forest</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #ffffff;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.05);
            }}
            .header {{
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background: #ffffff;
                border-radius: 8px;
                border-left: 4px solid #28a745;
                border: 1px solid #e0e0e0;
            }}
            .section h2 {{
                color: #333;
                border-bottom: 2px solid #28a745;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .console-output {{
                background: #f8f9fa;
                color: #333;
                padding: 20px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                overflow-x: auto;
                font-size: 14px;
                border: 1px solid #ddd;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
                border: 1px solid #ddd;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th {{
                background: #28a745;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                background: white;
            }}
            tr:nth-child(even) {{
                background: #f8f9fa;
            }}
            .highlight {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .summary-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }}
            .summary-card h3 {{
                margin: 0;
                color: #28a745;
                font-size: 24px;
            }}
            .summary-card p {{
                margin: 5px 0 0 0;
                color: #666;
            }}
            .plot-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .timestamp {{
                text-align: right;
                color: #888;
                font-style: italic;
                margin-top: 30px;
            }}
            .model-info {{
                background: #e8f5e8;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌲 Анализ Random Forest</h1>
                <p>Автоматический отчет по результатам анализа с помощью Random Forest</p>
            </div>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>{len(results_df)}</h3>
                    <p>Всего наблюдений</p>
                </div>
                <div class="summary-card">
                    <h3>{len(feature_importance)}</h3>
                    <p>Признаков</p>
                </div>
                <div class="summary-card">
                    <h3>{rf_model.n_estimators}</h3>
                    <p>Деревьев в лесу</p>
                </div>
                <div class="summary-card">
                    <h3>{problem_type.upper()}</h3>
                    <p>Тип задачи</p>
                </div>
            </div>
            
            <div class="model-info">
                <h3>📊 Информация о модели</h3>
                <p><strong>Тип модели:</strong> Random Forest {'Classifier' if problem_type == 'classification' else 'Regressor'}</p>
                <p><strong>Количество деревьев:</strong> {rf_model.n_estimators}</p>
                <p><strong>Максимальная глубина:</strong> {rf_model.max_depth if rf_model.max_depth else 'Не ограничена'}</p>
                <p><strong>Количество признаков:</strong> {rf_model.n_features_in_}</p>
            </div>
            
            <div class="section">
                <h2>📋 Полный вывод анализа</h2>
                <div class="console-output">{console_output}</div>
            </div>
            
            <div class="section">
                <h2>🔍 Важность признаков</h2>
                <p><em>Топ-15 самых важных признаков по версии Random Forest</em></p>
                <div class="plot-container">
                    <img src="data:image/png;base64,{feature_importance_plot}" alt="Feature Importance">
                </div>
                {feature_importance.head(20).to_html(classes='dataframe', border=0, index=False)}
            </div>
    """
    
    # Добавляем матрицу ошибок для классификации
    if problem_type == 'classification' and confusion_matrix_plot:
        html_content += f"""
            <div class="section">
                <h2>🎯 Матрица ошибок</h2>
                <p><em>Визуализация правильных и неправильных предсказаний</em></p>
                <div class="plot-container">
                    <img src="data:image/png;base64,{confusion_matrix_plot}" alt="Confusion Matrix">
                </div>
            </div>
        """
    
    html_content += f"""
            <div class="section">
                <h2>📈 Детальная информация о признаках</h2>
                <p><em>Полная таблица важности всех признаков</em></p>
                {feature_importance.to_html(classes='dataframe', border=0, index=False)}
            </div>
            
            <div class="timestamp">
                Отчет сгенерирован: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем HTML файл
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def lda_classifaction(data:pd.DataFrame,
                                target_column:str|None = None,
                                index_level:int|None = None,
                                n_components:int|None = None,
                               test_size:float = 0.2,
                               random_state:int = 42, 
                               output_html_path:str|None = None,
                               **kwargs
                               ) -> Tuple[pd.DataFrame, LinearDiscriminantAnalysis, StandardScaler, pd.DataFrame, LabelEncoder,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Анализ данных с помощью Linear Discriminant Analysis (LDA)
    с выводом результатов в HTML файл и проверкой на мультиколлинеарность
    """
    
    # Перехватываем вывод в консоль
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        print("=" * 70)
        print("АНАЛИЗ С ПОМОЩЬЮ LINEAR DISCRIMINANT ANALYSIS (LDA)")
        print("=" * 70)
        
        # Шаг 1: Подготовка данных
        print("1. Подготовка данных...")
        
        # Определяем целевую переменную
        if target_column is None and index_level:
            
            # Используем уровень индекса как целевую переменную
            target = data.index.get_level_values(index_level)
            
            features_df = data.reset_index(drop=True)
            target_name = f"{index_level} уровень индекса"
            
        else:
            
            target = data[target_column]
            features_df = data.drop(columns=[target_column])
            target_name = target_column
        
        print(f"   Целевая переменная: {target_name}")
        print(f"   Уникальных значений в целевой переменной: {target.nunique()}")
        
        # Проверяем, что задача классификации
        if target.dtype == 'object' or target.nunique() < 20:
            problem_type = 'classification'
        else:
            raise ValueError("LDA предназначен только для классификации. Используйте другие методы для регрессии.")
        
        # Кодируем целевую переменную
        le = LabelEncoder()
        target_encoded = le.fit_transform(target)
        class_names = le.classes_
        n_classes = len(class_names)
        
        print(f"   Тип задачи: {problem_type.upper()}")
        print(f"   Классы: {list(class_names)}")
        print(f"   Количество классов: {n_classes}")
        
        # Определяем количество компонентов для LDA
        if n_components is None:
            n_components = min(n_classes - 1, features_df.shape[1])
        print(f"   Количество компонентов LDA: {n_components}")
        
        # Шаг 2: Предобработка признаков
        print("2. Предобработка признаков...")
        
        # Удаляем нечисловые колонки если есть
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_numeric = features_df[numeric_columns].copy()
        
        # Проверяем наличие пропущенных значений
        if features_numeric.isnull().sum().sum() > 0:
            missing_count = features_numeric.isnull().sum().sum()
            print(f"   Обнаружено пропущенных значений: {missing_count}")
            features_numeric = features_numeric.fillna(features_numeric.mean())
        
        # Шаг 2.1: Проверка на мультиколлинеарность с помощью VIF
        print("2.1. Проверка на мультиколлинеарность (VIF анализ)...")
        
        # Проверяем, достаточно ли признаков для VIF анализа
        if len(features_numeric.columns) > 1:
            features_after_vif, vif_results, vif_threshold, _ = ut.calculate_vif(features_numeric, **kwargs)
            
            # Выводим результаты VIF анализа
            print(f"   Исходное количество признаков: {len(features_numeric.columns)}")
            print(f"   Количество признаков после VIF фильтрации: {len(features_after_vif.columns)}")
            print(f"   Удалено признаков с VIF > {vif_threshold}: {len(features_numeric.columns) - len(features_after_vif.columns)}")
            
            if len(vif_results) > 0:
                print(f"\n   Топ признаков по VIF (после фильтрации):")
                for i, row in vif_results.head(10).iterrows():
                    status = "⚠️ ВЫСОКИЙ" if row["VIF"] > vif_threshold else "✅ нормальный"
                    print(f"      {row['feature']}: {row['VIF']:.2f} ({status})")
            
            # Используем отфильтрованные признаки
            features_numeric = features_after_vif
            
            # Обновляем список числовых колонок
            numeric_columns = features_numeric.columns
            
            if len(numeric_columns) == 0:
                raise ValueError("После фильтрации VIF не осталось признаков. Уменьшите порог VIF.")
        else:
            print("   Недостаточно признаков для VIF анализа (требуется > 1)")
            vif_results = pd.DataFrame()
        
        # Масштабирование признаков (важно для LDA)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_numeric)
        
        print(f"   Размерность данных после VIF фильтрации: {features_scaled.shape}")
        print(f"   Количество признаков: {features_scaled.shape[1]}")
        
        # Шаг 3: Разделение на train/test с контролем стратификации
        print("3. Разделение на обучающую и тестовую выборки...")
        
        # Проверяем распределение классов перед разделением
        unique_classes, class_counts = np.unique(target_encoded, return_counts=True)
        print(f"   Распределение классов перед разделением:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"      Класс {cls} ({class_names[cls]}): {count} samples")
        
        # Используем стратификацию только если все классы имеют достаточное количество образцов
        min_samples_per_class = class_counts.min()
        
        if min_samples_per_class < 2:
            
            print("   Предупреждение: некоторые классы имеют менее 2 образцов, стратификация отключена")
            stratify = None
            
        else:
            stratify = target_encoded
        
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            features_scaled, target_encoded, features_df.index, 
            test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        print(f"   Обучающая выборка: {X_train.shape[0]} наблюдений")
        print(f"   Тестовая выборка: {X_test.shape[0]} наблюдений")
        
        # Проверяем наличие всех классов в обучающей и тестовой выборках
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        print(f"   Классы в обучающей выборке: {len(train_classes)}")
        print(f"   Классы в тестовой выборке: {len(test_classes)}")
        
        # Если в тестовой выборке не все классы, используем только присутствующие
        if len(test_classes) < len(class_names):
            
            print("   Предупреждение: не все классы присутствуют в тестовой выборке")
            present_classes_mask = np.isin(np.arange(len(class_names)), test_classes)
            present_class_names = class_names[present_classes_mask]
            print(f"   Используемые классы для отчета: {list(present_class_names)}")
            
        else:
            present_class_names = class_names
        
        # Шаг 4: Обучение LDA
        print("4. Обучение Linear Discriminant Analysis...")
        
        lda_model = LinearDiscriminantAnalysis(n_components=n_components)
        lda_model.fit(X_train, y_train)
        
        # Преобразование данных в LDA пространство
        X_train_lda = lda_model.transform(X_train)
        X_test_lda = lda_model.transform(X_test)
        
        print(f"   Размерность после LDA (обучение): {X_train_lda.shape}")
        print(f"   Размерность после LDA (тест): {X_test_lda.shape}")
        print(f"   Объясненная дисперсия: {lda_model.explained_variance_ratio_.sum():.4f}")
        
        # Шаг 4.1: Вывод уравнений LDA
        print("\n4.1. Уравнения классификации LDA:")
        print("   " + "-" * 50)
        
        def print_lda_equations(lda_model, feature_names, class_names):
            """Выводит уравнения дискриминантных функций LDA"""
            n_classes = len(class_names)
            n_features = len(feature_names)
            
            # Для первых K-1 классов используем coef_ и intercept_
            for i in range(n_classes - 1):
                equation = f"   δ_{class_names[i]}(x) = "
                parts = []
                for j in range(n_features):
                    coef_val = lda_model.coef_[i, j]
                    parts.append(f"{coef_val:+.4f}*{feature_names[j]}")
                
                equation += " ".join(parts)
                equation += f" {lda_model.intercept_[i]:+.4f}"
                print(equation)
            
            # Для последнего класса коэффициенты - это отрицательная сумма всех остальных
            last_coef = -np.sum(lda_model.coef_, axis=0)
            last_intercept = -np.sum(lda_model.intercept_[:-1])
            
            equation = f"   δ_{class_names[-1]}(x) = "
            parts = []
            for j in range(n_features):
                parts.append(f"{last_coef[j]:+.4f}*{feature_names[j]}")
            
            equation += " ".join(parts)
            equation += f" {last_intercept:+.4f}"
            print(equation)
            
            print("\n   📝 Примечание: Объект относится к классу с НАИБОЛЬШИМ значением δₖ(x)")
        
        # Выводим уравнения
        print_lda_equations(lda_model, numeric_columns, class_names)
        
        # Шаг 5: Предсказания и оценка модели
        print("5. Оценка модели...")
        
        y_pred = lda_model.predict(X_test)
        y_pred_proba = lda_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Accuracy: {accuracy:.4f}")
        
        print(f"\n   Classification Report:")
        print("   " + "-" * 50)
        
        try:
            clf_report = classification_report(
                y_test, y_pred, 
                target_names=present_class_names, 
                output_dict=True,
                zero_division=0
            )
            clf_report_df = pd.DataFrame(clf_report).transpose()
            print(clf_report_df.to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
            
            weighted_avg_accuracy = np.float64(clf_report_df.loc['weighted avg', 'precision']) # type: ignore
            macro_avg_accuracy = np.float64(clf_report_df.loc['macro avg', 'precision']) # type: ignore
            
        except ValueError as e:
            
            print(f"   Ошибка при создании classification report: {e}")
            print("   Используем числовые метки классов...")
            
            clf_report = classification_report(
                y_test, y_pred, 
                output_dict=True,
                zero_division=0
            )
            clf_report_df = pd.DataFrame(clf_report).transpose()
            print(clf_report_df.to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
            
            weighted_avg_accuracy = np.float64(clf_report_df.loc['weighted avg', 'precision']) # type: ignore
            macro_avg_accuracy = np.float64(clf_report_df.loc['macro avg', 'precision']) # type: ignore
            
        # Шаг 6: Анализ важности признаков через коэффициенты LDA
        print("\n6. Анализ важности признаков...")
        
        # Анализ коэффициентов LDA
        if n_components == 1:
            feature_importance = pd.DataFrame({
                'feature': numeric_columns,
                'coefficient': lda_model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
        else:
            # Для многокомпонентного случая используем сумму абсолютных значений по компонентам
            feature_importance = pd.DataFrame({
                'feature': numeric_columns,
                'coefficient_sum': np.sum(np.abs(lda_model.coef_), axis=0)
            }).sort_values('coefficient_sum', ascending=False)
        
        print("   Топ-10 самых важных признаков (по абсолютным коэффициентам LDA):")
        importance_column = 'coefficient' if n_components == 1 else 'coefficient_sum'
        for i, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row[importance_column]:.4f}")
        
        
        # Создаем DataFrame с результатами
        results_df = features_df.copy()
        results_df['actual'] = target_encoded
        results_df['predicted'] = lda_model.predict(features_scaled)
        results_df['is_correct'] = (results_df['actual'] == results_df['predicted'])
        
        results_df['Class'] = data.index.get_level_values('Class')
        results_df['Subclass'] = data.index.get_level_values('Subclass')
        
        # Добавляем LDA компоненты
        lda_components = lda_model.transform(features_scaled)
        for i in range(lda_components.shape[1]):
            results_df[f'LDA_Component_{i+1}'] = lda_components[:, i]
        
        # Получаем весь вывод
        console_output = captured_output.getvalue()
        
        # Восстанавливаем stdout
        sys.stdout = old_stdout
        
        # Создаем HTML отчет
        if output_html_path:
            create_lda_classifaction_html_report(console_output, results_df, feature_importance, 
                                 lda_model, class_names, le, X_test, y_test, y_pred, 
                                 X_train_lda, y_train, output_html_path, vif_results, numeric_columns)
            
            print(f"\n✅ HTML отчет сохранен в файл: {output_html_path}")
        
        return (results_df, lda_model, scaler, feature_importance,le,
                X_train, X_test, y_train, y_test, X_train_lda, X_test_lda,
                float(accuracy), weighted_avg_accuracy, macro_avg_accuracy,
                )
        
    except Exception as e:
        # Восстанавливаем stdout в случае ошибки
        sys.stdout = old_stdout
        print(f"Ошибка при выполнении анализа: {e}")
        raise

def create_lda_classifaction_html_report(console_output, results_df, feature_importance, 
                          lda_model, class_names, label_encoder,
                          X_test, y_test, y_pred, X_lda, y_train, output_html_path, vif_results=None, feature_names=None):
    """Создание HTML отчета с результатами анализа LDA"""
    
    # Создаем визуализации
    plt.style.use('default')
    
    # График важности признаков
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(15)
    importance_column = 'coefficient' if 'coefficient' in top_features.columns else 'coefficient_sum'
    
    colors = ['red' if x < 0 else 'blue' for x in top_features[importance_column]]
    ax1.barh(top_features['feature'], top_features[importance_column], color=colors)
    ax1.set_xlabel('Коэффициент LDA')
    ax1.set_title('Топ-15 самых важных признаков (LDA коэффициенты)')
    ax1.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем график в base64
    buffer1 = BytesIO()
    fig1.savefig(buffer1, format='png', dpi=100, bbox_inches='tight')
    buffer1.seek(0)
    feature_importance_plot = base64.b64encode(buffer1.getvalue()).decode()
    plt.close(fig1)
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    ax2.set_xlabel('Предсказанные')
    ax2.set_ylabel('Фактические')
    ax2.set_title('Матрица ошибок')
    plt.tight_layout()
    
    buffer2 = BytesIO()
    fig2.savefig(buffer2, format='png', dpi=100, bbox_inches='tight')
    buffer2.seek(0)
    confusion_matrix_plot = base64.b64encode(buffer2.getvalue()).decode()
    plt.close(fig2)
    
    # Визуализация LDA компонент
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    if X_lda.shape[1] >= 2:
        # 2D scatter plot для первых двух компонент
        scatter = ax3.scatter(X_lda[:, 0], X_lda[:, 1], c=y_train, cmap='viridis', alpha=0.7)
        ax3.set_xlabel(f'LDA Component 1 ({lda_model.explained_variance_ratio_[0]:.2%})')
        ax3.set_ylabel(f'LDA Component 2 ({lda_model.explained_variance_ratio_[1]:.2%})')
        ax3.set_title('LDA Projection (первые 2 компоненты)')
        plt.colorbar(scatter, ax=ax3, label='Класс')
    else:
        # 1D гистограмма если только одна компонента
        for class_idx in np.unique(y_train):
            class_mask = y_train == class_idx
            ax3.hist(X_lda[class_mask, 0], alpha=0.7, label=f'Класс {class_names[class_idx]}', bins=20)
        ax3.set_xlabel('LDA Component 1')
        ax3.set_ylabel('Частота')
        ax3.set_title('Распределение по LDA Component 1')
        ax3.legend()
    
    plt.tight_layout()
    buffer3 = BytesIO()
    fig3.savefig(buffer3, format='png', dpi=100, bbox_inches='tight')
    buffer3.seek(0)
    lda_projection_plot = base64.b64encode(buffer3.getvalue()).decode()
    plt.close(fig3)
    
    # Создаем секцию уравнений LDA
    equations_section = ""
    if feature_names is not None:
        equations_html = []
        n_classes = len(class_names)
        n_features = len(feature_names)
        
        # Для первых K-1 классов
        for i in range(n_classes - 1):
            equation_parts = []
            for j in range(n_features):
                coef_val = lda_model.coef_[i, j]
                equation_parts.append(f"{coef_val:+.4f}×{feature_names[j]}")
            equation = " + ".join(equation_parts)
            equation += f" {lda_model.intercept_[i]:+.4f}"
            equations_html.append(f"<p><strong>δ<sub>{class_names[i]}</sub>(x)</strong> = {equation}</p>")
        
        # Для последнего класса
        last_coef = -np.sum(lda_model.coef_, axis=0)
        last_intercept = -np.sum(lda_model.intercept_[:-1])
        
        equation_parts = []
        for j in range(n_features):
            equation_parts.append(f"{last_coef[j]:+.4f}×{feature_names[j]}")
        equation = " + ".join(equation_parts)
        equation += f" {last_intercept:+.4f}"
        equations_html.append(f"<p><strong>δ<sub>{class_names[-1]}</sub>(x)</strong> = {equation}</p>")
        
        equations_section = f"""
        <div class="section">
            <h2>🧮 Уравнения классификации LDA</h2>
            <p><em>Дискриминантные функции для каждого класса</em></p>
            <div class="coefficient-info">
                {"".join(equations_html)}
                <p><strong>📝 Правило классификации:</strong> Объект относится к классу с НАИБОЛЬШИМ значением δₖ(x)</p>
            </div>
        </div>
        """
    
    # Создаем секцию VIF если есть результаты
    vif_section = ""
    if vif_results is not None and len(vif_results) > 0:
        vif_table = vif_results.to_html(classes='dataframe', border=0, index=False)
        vif_section = f"""
        <div class="section">
            <h2>📊 Анализ мультиколлинеарности (VIF)</h2>
            <p><em>Variance Inflation Factor для признаков после фильтрации</em></p>
            {vif_table}
            <div class="coefficient-info">
                <h3>ℹ️ Интерпретация VIF</h3>
                <p><strong>VIF < 5</strong>: Нет мультиколлинеарности</p>
                <p><strong>5 ≤ VIF < 10</strong>: Умеренная мультиколлинеарность</p>
                <p><strong>VIF ≥ 10</strong>: Высокая мультиколлинеарность (требует внимания)</p>
            </div>
        </div>
        """
    
    # Создаем стилизованный HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Анализ Linear Discriminant Analysis (LDA)</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #ffffff;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.05);
            }}
            .header {{
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background: #ffffff;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                border: 1px solid #e0e0e0;
            }}
            .section h2 {{
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .console-output {{
                background: #f8f9fa;
                color: #333;
                padding: 20px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                overflow-x: auto;
                font-size: 14px;
                border: 1px solid #ddd;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
                border: 1px solid #ddd;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th {{
                background: #007bff;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                background: white;
            }}
            tr:nth-child(even) {{
                background: #f8f9fa;
            }}
            .highlight {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .summary-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }}
            .summary-card h3 {{
                margin: 0;
                color: #007bff;
                font-size: 24px;
            }}
            .summary-card p {{
                margin: 5px 0 0 0;
                color: #666;
            }}
            .plot-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .timestamp {{
                text-align: right;
                color: #888;
                font-style: italic;
                margin-top: 30px;
            }}
            .model-info {{
                background: #e8f4ff;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }}
            .coefficient-info {{
                background: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 4px solid #ffc107;
            }}
            .equations {{
                font-family: 'Courier New', monospace;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Анализ Linear Discriminant Analysis (LDA)</h1>
                <p>Автоматический отчет по результатам анализа с помощью LDA</p>
            </div>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>{len(results_df)}</h3>
                    <p>Всего наблюдений</p>
                </div>
                <div class="summary-card">
                    <h3>{len(feature_importance)}</h3>
                    <p>Признаков</p>
                </div>
                <div class="summary-card">
                    <h3>{len(class_names)}</h3>
                    <p>Классов</p>
                </div>
                <div class="summary-card">
                    <h3>{lda_model.n_components}</h3>
                    <p>LDA компонент</p>
                </div>
            </div>
            
            <div class="model-info">
                <h3>📊 Информация о модели LDA</h3>
                <p><strong>Тип модели:</strong> Linear Discriminant Analysis</p>
                <p><strong>Количество компонентов:</strong> {lda_model.n_components}</p>
                <p><strong>Количество классов:</strong> {len(class_names)}</p>
            </div>
            
            {equations_section}
            
            <div class="coefficient-info">
                <h3>ℹ️ Интерпретация коэффициентов LDA</h3>
                <p><strong>Положительные коэффициенты</strong> (синие) увеличивают вероятность принадлежности к определенным классам</p>
                <p><strong>Отрицательные коэффициенты</strong> (красные) уменьшают вероятность принадлежности к определенным классам</p>
                <p>Чем больше абсолютное значение коэффициента, тем сильнее влияние признака на разделение классов</p>
            </div>
            
            {vif_section}
            
            <div class="section">
                <h2>📋 Полный вывод анализа</h2>
                <div class="console-output">{console_output}</div>
            </div>
            
            <div class="section">
                <h2>🔍 Важность признаков (LDA коэффициенты)</h2>
                <p><em>Топ-15 самых важных признаков по версии LDA</em></p>
                <div class="plot-container">
                    <img src="data:image/png;base64,{feature_importance_plot}" alt="Feature Importance">
                </div>
                {feature_importance.head(20).to_html(classes='dataframe', border=0, index=False)}
            </div>
            
            <div class="section">
                <h2>🎯 Матрица ошибок</h2>
                <p><em>Визуализация правильных и неправильных предсказаний</em></p>
                <div class="plot-container">
                    <img src="data:image/png;base64,{confusion_matrix_plot}" alt="Confusion Matrix">
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Проекция LDA</h2>
                <p><em>Визуализация данных в пространстве LDA компонент</em></p>
                <div class="plot-container">
                    <img src="data:image/png;base64,{lda_projection_plot}" alt="LDA Projection">
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Детальная информация о признаках</h2>
                <p><em>Полная таблица коэффициентов LDA для всех признаков</em></p>
                {feature_importance.to_html(classes='dataframe', border=0, index=False)}
            </div>
            
            <div class="timestamp">
                Отчет сгенерирован: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем HTML файл
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

