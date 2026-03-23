import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

import HumSpectra.utilits as ut

def kmeans_clustering(df, n_clusters=None, random_state=42, output_html_path=None, save=False):
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
            create_kmeans_html_report(console_output, result_df, contingency_table, 
                            percentage_table, cluster_summary_df, cluster_means, 
                            subclass_cluster_map, n_clusters, output_html_path)
            
            print(f"\n✅ HTML отчет сохранен в файл: {output_html_path}")
        
        return result_df, kmeans, scaler, contingency_table, percentage_table, subclass_cluster_map
        
    except Exception as e:
        # Восстанавливаем stdout в случае ошибки
        sys.stdout = old_stdout
        print(f"Ошибка при выполнении анализа: {e}")
        raise

def create_kmeans_html_report(console_output, result_df, contingency_table, percentage_table, 
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

def random_forest_classification_batch(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    index_level: Optional[int] = None,
    vif_filter: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Минимум операций ввода-вывода, максимальная скорость.
    Без HTML отчета, без print, только вычисления.
    
    Возвращает словарь с результатами для быстрого доступа.
    """
    
    # 1. ПОДГОТОВКА ДАННЫХ (векторизованная)
    if (target_column is None) and (index_level is not None):
        target = data.index.get_level_values(index_level).values
        features_df = data.reset_index(drop=True)
        target_name = f"{index_level}_level_index"
    else:
        target = data[target_column].values
        features_df = data.drop(columns=[target_column])
        target_name = target_column
    
    # 2. КОДИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target) # type: ignore
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    
    # 3. ПРЕДОБРАБОТКА ПРИЗНАКОВ (векторизованная)
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        raise ValueError("Не найдено числовых признаков")
    
    # Преобразуем сразу в numpy array для скорости
    X = features_df[numeric_columns].values
    
    # Обработка пропусков (быстрая, векторизованная)
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    # VIF фильтрация (опционально, векторизованная)
    if vif_filter and X.shape[1] > 1:
        X, vif_results = _fast_vif_filter(X, numeric_columns, **kwargs)
        numeric_columns = numeric_columns[X.shape[1] == X.shape[1]]  # Обновляем
        vif_results_dict = vif_results
    else:
        vif_results_dict = None
    
    # 4. МАСШТАБИРОВАНИЕ (один раз для всех)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. РАЗДЕЛЕНИЕ ВЫБОРОК
    # Определяем стратификацию
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    stratify = y_encoded if class_counts.min() >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # 6. ОБУЧЕНИЕ МОДЕЛИ
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # 7. ПРЕДСКАЗАНИЯ И МЕТРИКИ
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Расчет метрик по классам
    clf_report = classification_report(
        y_test, y_pred,
        output_dict=True,
        zero_division=0
    )
    
    weighted_avg_accuracy = clf_report['weighted avg']['precision'] # type: ignore
    macro_avg_accuracy = clf_report['macro avg']['precision'] # type: ignore
    
    # 8. ВАЖНОСТЬ ПРИЗНАКОВ
    feature_importance = pd.DataFrame({
        'feature': numeric_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 9. РЕЗУЛЬТАТЫ
    results_df = pd.DataFrame({
        'true_class': y_encoded,
        'predicted_class': rf_model.predict(X_scaled),
        'is_correct': y_encoded == rf_model.predict(X_scaled)
    })
    
    return {
        'results_df': results_df,
        'model': rf_model,
        'scaler': scaler,
        'feature_importance': feature_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'weighted_avg_accuracy': weighted_avg_accuracy,
        'macro_avg_accuracy': macro_avg_accuracy,
        'classification_report': clf_report,
        'vif_results': vif_results_dict,
        'n_classes': n_classes,
        'class_names': class_names,
        'n_features': len(numeric_columns)
    }


def _fast_vif_filter(X: np.ndarray, feature_names: pd.Index, threshold: float = 10.0, **kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Быстрая фильтрация мультиколлинеарности с помощью VIF.
    Векторизованная версия для массовых вычислений.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    n_features = X.shape[1]
    vif_results = []
    
    # Стандартизируем для VIF
    X_scaled_vif = StandardScaler().fit_transform(X)
    
    # Список признаков для удаления
    to_remove = set()
    
    # Итеративно удаляем признаки с высоким VIF
    remaining_features = list(range(n_features))
    max_iterations = n_features
    
    for _ in range(max_iterations):
        if len(remaining_features) <= 1:
            break
            
        # Пересчитываем VIF для оставшихся признаков
        current_X = X_scaled_vif[:, remaining_features]
        current_vif = []
        
        for idx in range(len(remaining_features)):
            # Векторизованный расчет VIF
            y = current_X[:, idx]
            X_others = np.delete(current_X, idx, axis=1)
            
            # Проверяем, что есть другие признаки
            if X_others.shape[1] > 0:
                model = LinearRegression()
                model.fit(X_others, y)
                r2 = model.score(X_others, y)
                vif = 1.0 / (1.0 - r2) if r2 < 0.999 else 1000.0
            else:
                vif = 1.0
            
            current_vif.append((remaining_features[idx], vif))
        
        # Находим признак с максимальным VIF
        max_vif_feature, max_vif_value = max(current_vif, key=lambda x: x[1])
        
        # Сохраняем результаты
        vif_results.append({
            'feature': feature_names[max_vif_feature],
            'VIF': max_vif_value
        })
        
        # Если VIF превышает порог, удаляем признак
        if max_vif_value > threshold:
            to_remove.add(max_vif_feature)
            remaining_features.remove(max_vif_feature)
        else:
            break
    
    # Фильтруем X
    keep_indices = [i for i in range(n_features) if i not in to_remove]
    X_filtered = X[:, keep_indices]
    
    vif_df = pd.DataFrame(vif_results)
    
    return X_filtered, vif_df


# ============================================================================
# ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ ОДИНОЧНЫХ ВЫЧИСЛЕНИЙ С HTML ОТЧЕТОМ
# ============================================================================

def random_forest_classification(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    output_html_path: str|None = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    index_level: Optional[int] = None,
    external_validation: bool = False,
    cv_dataset: pd.DataFrame | None = None,
    vif_filter: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, RandomForestClassifier, StandardScaler, pd.DataFrame,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, float, float, float, pd.DataFrame, dict | None]:
    """
    ИСПРАВЛЕННАЯ ВЕРСИЯ: Для одиночных вычислений с полным HTML отчетом.
    Использует оптимизированные вычисления внутри.
    """
    
    import sys
    from io import StringIO
    
    # Сохраняем оригинальный stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    external_results = None
    
    try:
        print("=" * 70)
        print("RANDOM FOREST АНАЛИЗ КЛАССИФИКАЦИИ")
        if external_validation and cv_dataset is not None:
            print("С ИСПОЛЬЗОВАНИЕМ КРОСС-ВАЛИДАЦИИ")
        print("=" * 70)
        
        # ИСПОЛЬЗУЕМ ОПТИМИЗИРОВАННУЮ ВЕРСИЮ ДЛЯ ВЫЧИСЛЕНИЙ
        batch_results = random_forest_classification_batch(
            data=data,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            index_level=index_level,
            vif_filter=vif_filter,
            **kwargs
        )
        
        # Извлекаем результаты
        results_df = batch_results['results_df']
        rf_model = batch_results['model']
        scaler = batch_results['scaler']
        feature_importance = batch_results['feature_importance']
        X_train = batch_results['X_train']
        X_test = batch_results['X_test']
        y_train = batch_results['y_train']
        y_test = batch_results['y_test']
        label_encoder = batch_results['label_encoder']
        accuracy = batch_results['accuracy']
        weighted_avg_accuracy = batch_results['weighted_avg_accuracy']
        macro_avg_accuracy = batch_results['macro_avg_accuracy']
        clf_report_df = pd.DataFrame(batch_results['classification_report']).transpose()
        class_names = batch_results['class_names']
        numeric_columns = feature_importance['feature'].values
        
        # Выводим информацию в консоль
        print("\n1. ПОДГОТОВКА ДАННЫХ")
        print("-" * 40)
        print(f"   Целевая переменная: {target_column if target_column else f'{index_level} уровень индекса'}")
        print(f"   Уникальных значений в целевой переменной: {batch_results['n_classes']}")
        print(f"   Классы: {list(class_names)}")
        print(f"   Размер датасета: {len(data)} наблюдений")
        
        print("\n2. ПРЕДОБРАБОТКА ПРИЗНАКОВ")
        print("-" * 40)
        print(f"   Числовых признаков: {batch_results['n_features']}")
        print(f"   Размерность данных: {batch_results['results_df'].shape}")
        
        print("\n3. РАЗДЕЛЕНИЕ НА ВЫБОРКИ")
        print("-" * 40)
        print("   Распределение классов:")
        for cls, name in enumerate(class_names):
            count = np.sum(batch_results['y_train'] == cls) + np.sum(batch_results['y_test'] == cls)
            print(f"      {name}: {count} наблюдений")
        print(f"   Обучающая выборка: {X_train.shape[0]} наблюдений")
        print(f"   Тестовая выборка: {X_test.shape[0]} наблюдений")
        
        print("\n4. ОБУЧЕНИЕ RANDOM FOREST")
        print("-" * 40)
        print(f"   Модель обучена: {n_estimators} деревьев")
        
        print("\n5. ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
        print("-" * 40)
        print(f"   Точность (Accuracy): {accuracy:.4f}")
        print("\n   Отчет по классификации:")
        print("   " + "-" * 35)
        print(clf_report_df.to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
        
        # Внешняя валидация
        if external_validation and cv_dataset is not None:
            external_results = _perform_external_validation(
                rf_model, scaler, label_encoder, numeric_columns,
                cv_dataset, target_column, index_level
            )
        
        print("\n6. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("-" * 40)
        print("   Топ-10 важнейших признаков:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"      {i:2d}. {row['feature']:30s}: {row['importance']:.4f}")
        
        print(f"\n   Общая точность на всем датасете: {results_df['is_correct'].mean():.4f}")
        
        print("\n" + "=" * 70)
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 70)
        print(f"   Классов: {batch_results['n_classes']}")
        print(f"   Признаков: {batch_results['n_features']}")
        print(f"   Точность на тесте: {accuracy:.4f}")
        print(f"   Самый важный признак: {feature_importance.iloc[0]['feature']}")
        print("=" * 70)
        
        # Получаем вывод консоли
        console_output = captured_output.getvalue()
        
        # Восстанавливаем stdout
        sys.stdout = old_stdout
        
        # Создание HTML отчета
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
                    y_pred=rf_model.predict(X_test),
                    output_html_path=output_html_path,
                    vif_results=batch_results.get('vif_results'),
                    external_results=external_results
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
            accuracy,
            weighted_avg_accuracy,
            macro_avg_accuracy,
            clf_report_df,
            external_results
        )
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"❌ Ошибка при выполнении анализа: {e}")
        raise


def _perform_external_validation(model, scaler, label_encoder, numeric_columns,
                                 cv_dataset, target_column, index_level):
    """Вспомогательная функция для внешней валидации"""
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    
    # Определяем целевую переменную
    if (target_column is None) and (index_level is not None):
        cv_target = cv_dataset.index.get_level_values(index_level)
        cv_features_df = cv_dataset.reset_index(drop=True)
    else:
        cv_target = cv_dataset[target_column]
        cv_features_df = cv_dataset.drop(columns=[target_column])
    
    # Кодируем целевую переменную
    known_mask = cv_target.isin(label_encoder.classes_)
    cv_target_filtered = cv_target[known_mask]
    cv_features_filtered = cv_features_df[known_mask]
    
    if len(cv_target_filtered) == 0:
        return None
    
    cv_target_encoded = label_encoder.transform(cv_target_filtered)
    
    # Выбираем общие признаки
    cv_numeric = cv_features_filtered.select_dtypes(include=[np.number])
    common_cols = list(set(numeric_columns).intersection(set(cv_numeric.columns)))
    
    if not common_cols:
        return None
    
    cv_features_numeric = cv_numeric[common_cols].values
    
    # Масштабируем
    cv_features_scaled = scaler.transform(cv_features_numeric)
    
    # Предсказываем
    cv_predictions = model.predict(cv_features_scaled)
    cv_predictions_proba = model.predict_proba(cv_features_scaled)
    
    # Метрики
    cv_accuracy = accuracy_score(cv_target_encoded, cv_predictions)
    cv_unique_classes = np.unique(cv_target_encoded)
    cv_present_names = label_encoder.classes_[cv_unique_classes]
    
    cv_clf_report = classification_report(
        cv_target_encoded, cv_predictions,
        target_names=cv_present_names,
        output_dict=True,
        labels=cv_unique_classes,
        zero_division=0
    )
    
    return {
        'features_scaled': cv_features_scaled,
        'target_encoded': cv_target_encoded,
        'predictions': cv_predictions,
        'predictions_proba': cv_predictions_proba,
        'accuracy': cv_accuracy,
        'classification_report': cv_clf_report,
        'present_classes': cv_present_names
    }
def create_rf_classification_html_report(console_output, results_df, feature_importance, 
                         rf_model, class_names, label_encoder,
                         X_test, y_test, y_pred, output_html_path, vif_results,
                         external_results=None, problem_type='classification'):
    """Создание HTML отчета с результатами анализа Random Forest с поддержкой кросс-валидации"""
    
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
            .coefficient-info {{
                background: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 4px solid #ffc107;
            }}
            .cv-badge {{
                display: inline-block;
                background: #17a2b8;
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 14px;
                margin-left: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌲 Анализ Random Forest</h1>
                <p>Автоматический отчет по результатам анализа с помощью Random Forest</p>
                {f'<p class="cv-badge">С КРОСС-ВАЛИДАЦИЕЙ</p>' if external_results else ''}
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
            
            {vif_section}
            
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
        
def lda_classification_batch(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    index_level: Optional[int] = None,
    n_components: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    vif_filter: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ LDA: Для массовых вычислений (тысячи моделей)
    Минимум операций ввода-вывода, максимальная скорость.
    
    Returns:
        Dict с результатами для быстрого доступа
    """
    
    # 1. ПОДГОТОВКА ДАННЫХ (векторизованная)
    if (target_column is None) and (index_level is not None):
        target = data.index.get_level_values(index_level).values
        features_df = data.reset_index(drop=True)
    else:
        target = data[target_column].values
        features_df = data.drop(columns=[target_column])
    
    # 2. КОДИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target) # type: ignore
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    
    # Проверка на задачу классификации
    if n_classes < 2:
        raise ValueError("Для LDA необходимо минимум 2 класса")
    
    # 3. ПРЕДОБРАБОТКА ПРИЗНАКОВ
    # Выбираем только числовые признаки
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        raise ValueError("Не найдено числовых признаков")
    
    # Преобразуем в numpy array для скорости
    X = features_df[numeric_columns].values
    
    # Быстрая обработка пропусков
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    # VIF фильтрация (опционально)
    if vif_filter and X.shape[1] > 1:
        X, vif_results = _fast_vif_filter_lda(X, numeric_columns, **kwargs)
        numeric_columns = numeric_columns[:X.shape[1]]
        vif_results_dict = vif_results
    else:
        vif_results_dict = None
    
    # 4. МАСШТАБИРОВАНИЕ
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. РАЗДЕЛЕНИЕ ВЫБОРОК
    # Определяем стратификацию
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    stratify = y_encoded if class_counts.min() >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # 6. ОПРЕДЕЛЯЕМ КОЛИЧЕСТВО КОМПОНЕНТОВ
    if n_components is None:
        n_components = min(n_classes - 1, X_train.shape[1])
    n_components = min(n_components, n_classes - 1, X_train.shape[1])
    
    # 7. ОБУЧЕНИЕ LDA
    lda_model = LinearDiscriminantAnalysis(n_components=n_components)
    lda_model.fit(X_train, y_train)
    
    # 8. ПРЕОБРАЗОВАНИЕ В LDA ПРОСТРАНСТВО
    X_train_lda = lda_model.transform(X_train) if n_components > 0 else np.array([])
    X_test_lda = lda_model.transform(X_test) if n_components > 0 else np.array([])
    
    # 9. ПРЕДСКАЗАНИЯ И МЕТРИКИ
    y_pred = lda_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Расчет метрик по классам
    # Определяем какие классы присутствуют в тестовой выборке
    test_classes = np.unique(y_test)
    present_class_indices = test_classes
    present_class_names = class_names[present_class_indices]
    
    clf_report = classification_report(
        y_test, y_pred,
        target_names=present_class_names,
        labels=present_class_indices,
        output_dict=True,
        zero_division=0
    )
    
    weighted_avg_accuracy = clf_report['weighted avg']['precision'] # type: ignore
    macro_avg_accuracy = clf_report['macro avg']['precision'] # type: ignore
    
    # 10. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
    if n_components == 1:
        feature_importance = pd.DataFrame({
            'feature': numeric_columns,
            'importance': np.abs(lda_model.coef_[0])
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'feature': numeric_columns,
            'importance': np.sum(np.abs(lda_model.coef_), axis=0)
        }).sort_values('importance', ascending=False)
    
    # 11. LDA КОМПОНЕНТЫ ДЛЯ ВСЕХ ДАННЫХ
    X_all_lda = lda_model.transform(X_scaled) if n_components > 0 else np.array([])
    
    # 12. РЕЗУЛЬТАТЫ
    results_df = pd.DataFrame({
        'true_class': y_encoded,
        'predicted_class': lda_model.predict(X_scaled),
        'is_correct': y_encoded == lda_model.predict(X_scaled)
    })
    
    # Добавляем LDA компоненты
    if X_all_lda.size > 0:
        for i in range(X_all_lda.shape[1]):
            results_df[f'LDA_Component_{i+1}'] = X_all_lda[:, i]
    
    return {
        'results_df': results_df,
        'model': lda_model,
        'scaler': scaler,
        'feature_importance': feature_importance,
        'label_encoder': label_encoder,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_lda': X_train_lda,
        'X_test_lda': X_test_lda,
        'accuracy': accuracy,
        'weighted_avg_accuracy': weighted_avg_accuracy,
        'macro_avg_accuracy': macro_avg_accuracy,
        'classification_report': clf_report,
        'vif_results': vif_results_dict,
        'n_classes': n_classes,
        'class_names': class_names,
        'n_features': len(numeric_columns),
        'n_components': n_components,
        'explained_variance_ratio': lda_model.explained_variance_ratio_ if hasattr(lda_model, 'explained_variance_ratio_') else None,
        'coef_': lda_model.coef_ if hasattr(lda_model, 'coef_') else None,
        'intercept_': lda_model.intercept_ if hasattr(lda_model, 'intercept_') else None
    }


def _fast_vif_filter_lda(X: np.ndarray, feature_names: pd.Index, threshold: float = 10.0, **kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Быстрая фильтрация мультиколлинеарности для LDA.
    Векторизованная версия для массовых вычислений.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    n_features = X.shape[1]
    vif_results = []
    
    # Стандартизируем для VIF
    X_scaled_vif = StandardScaler().fit_transform(X)
    
    # Список признаков для удаления
    to_remove = set()
    
    # Итеративно удаляем признаки с высоким VIF
    remaining_features = list(range(n_features))
    max_iterations = n_features
    
    for _ in range(max_iterations):
        if len(remaining_features) <= 1:
            break
            
        # Пересчитываем VIF для оставшихся признаков
        current_X = X_scaled_vif[:, remaining_features]
        current_vif = []
        
        for idx in range(len(remaining_features)):
            y = current_X[:, idx]
            X_others = np.delete(current_X, idx, axis=1)
            
            if X_others.shape[1] > 0:
                model = LinearRegression()
                model.fit(X_others, y)
                r2 = model.score(X_others, y)
                vif = 1.0 / (1.0 - r2) if r2 < 0.999 else 1000.0
            else:
                vif = 1.0
            
            current_vif.append((remaining_features[idx], vif))
        
        # Находим признак с максимальным VIF
        max_vif_feature, max_vif_value = max(current_vif, key=lambda x: x[1])
        
        # Сохраняем результаты
        vif_results.append({
            'feature': feature_names[max_vif_feature],
            'VIF': max_vif_value
        })
        
        # Если VIF превышает порог, удаляем признак
        if max_vif_value > threshold:
            to_remove.add(max_vif_feature)
            remaining_features.remove(max_vif_feature)
        else:
            break
    
    # Фильтруем X
    keep_indices = [i for i in range(n_features) if i not in to_remove]
    X_filtered = X[:, keep_indices]
    
    vif_df = pd.DataFrame(vif_results)
    
    return X_filtered, vif_df


def _get_lda_equations(lda_model, feature_names, class_names) -> List[str]:
    """
    Генерирует уравнения LDA для вывода (только для отчета)
    """
    equations = []
    n_classes = len(class_names)
    n_features = len(feature_names)
    
    # Для первых K-1 классов
    for i in range(min(n_classes - 1, lda_model.coef_.shape[0])):
        equation = f"δ_{class_names[i]}(x) = "
        parts = []
        for j in range(n_features):
            coef_val = lda_model.coef_[i, j]
            parts.append(f"{coef_val:+.4f}*{feature_names[j]}")
        equation += " ".join(parts)
        if hasattr(lda_model, 'intercept_') and len(lda_model.intercept_) > i:
            equation += f" {lda_model.intercept_[i]:+.4f}"
        equations.append(equation)
    
    # Для последнего класса
    if n_classes > 1 and lda_model.coef_.shape[0] == n_classes - 1:
        last_coef = -np.sum(lda_model.coef_, axis=0)
        last_intercept = -np.sum(lda_model.intercept_)
        equation = f"δ_{class_names[-1]}(x) = "
        parts = []
        for j in range(n_features):
            parts.append(f"{last_coef[j]:+.4f}*{feature_names[j]}")
        equation += " ".join(parts)
        equation += f" {last_intercept:+.4f}"
        equations.append(equation)
    
    return equations


# ============================================================================
# ИСПРАВЛЕННАЯ ВЕРСИЯ LDA ДЛЯ ОДИНОЧНЫХ ВЫЧИСЛЕНИЙ С HTML ОТЧЕТОМ
# ============================================================================

def lda_classification(
    data: pd.DataFrame,
    target_column: str | None = None,
    index_level: int | None = None,
    n_components: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    output_html_path: str | None = None,
    external_validation: bool = False,
    cv_dataset: pd.DataFrame | None = None,
    vif_filter: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, LinearDiscriminantAnalysis, StandardScaler, 
           pd.DataFrame, LabelEncoder, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           float, float, float, pd.DataFrame, dict | None]:
    """
    ИСПРАВЛЕННАЯ ВЕРСИЯ LDA: Для одиночных вычислений с полным HTML отчетом.
    Использует оптимизированные вычисления внутри.
    """
    
    import sys
    from io import StringIO
    
    # Сохраняем оригинальный stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    external_results = None
    
    try:
        print("=" * 70)
        print("АНАЛИЗ С ПОМОЩЬЮ LINEAR DISCRIMINANT ANALYSIS (LDA)")
        if external_validation and cv_dataset is not None:
            print("С ИСПОЛЬЗОВАНИЕМ КРОСС-ВАЛИДАЦИИ")
        print("=" * 70)
        
        # ИСПОЛЬЗУЕМ ОПТИМИЗИРОВАННУЮ ВЕРСИЮ ДЛЯ ВЫЧИСЛЕНИЙ
        batch_results = lda_classification_batch(
            data=data,
            target_column=target_column,
            index_level=index_level,
            n_components=n_components,
            test_size=test_size,
            random_state=random_state,
            vif_filter=vif_filter,
            **kwargs
        )
        
        # Извлекаем результаты
        results_df = batch_results['results_df']
        lda_model = batch_results['model']
        scaler = batch_results['scaler']
        feature_importance = batch_results['feature_importance']
        label_encoder = batch_results['label_encoder']
        X_train = batch_results['X_train']
        X_test = batch_results['X_test']
        y_train = batch_results['y_train']
        y_test = batch_results['y_test']
        X_train_lda = batch_results['X_train_lda']
        X_test_lda = batch_results['X_test_lda']
        accuracy = batch_results['accuracy']
        weighted_avg_accuracy = batch_results['weighted_avg_accuracy']
        macro_avg_accuracy = batch_results['macro_avg_accuracy']
        clf_report_df = pd.DataFrame(batch_results['classification_report']).transpose()
        class_names = batch_results['class_names']
        n_components_actual = batch_results['n_components']
        numeric_columns = feature_importance['feature'].values
        
        # Шаг 1: Информация о данных
        print("\n1. ПОДГОТОВКА ДАННЫХ")
        print("-" * 40)
        print(f"   Целевая переменная: {target_column if target_column else f'{index_level} уровень индекса'}")
        print(f"   Уникальных значений: {batch_results['n_classes']}")
        print(f"   Классы: {list(class_names)}")
        print(f"   Размер датасета: {len(data)} наблюдений")
        
        # Шаг 2: Предобработка
        print("\n2. ПРЕДОБРАБОТКА ПРИЗНАКОВ")
        print("-" * 40)
        print(f"   Числовых признаков: {batch_results['n_features']}")
        print(f"   Количество компонентов LDA: {n_components_actual}")
        
        # Шаг 3: Разделение данных
        print("\n3. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
        print("-" * 40)
        print("   Распределение классов:")
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            print(f"      {class_names[cls]}: {count} наблюдений (обучение)")
        print(f"   Обучающая выборка: {X_train.shape[0]} наблюдений")
        print(f"   Тестовая выборка: {X_test.shape[0]} наблюдений")
        
        # Шаг 4: Обучение LDA
        print("\n4. ОБУЧЕНИЕ LINEAR DISCRIMINANT ANALYSIS")
        print("-" * 40)
        print(f"   Размерность после LDA (обучение): {X_train_lda.shape if X_train_lda.size > 0 else '0'}")
        print(f"   Размерность после LDA (тест): {X_test_lda.shape if X_test_lda.size > 0 else '0'}")
        if batch_results['explained_variance_ratio'] is not None:
            print(f"   Объясненная дисперсия: {batch_results['explained_variance_ratio'].sum():.4f}")
        
        # Выводим уравнения LDA
        print("\n4.1. УРАВНЕНИЯ КЛАССИФИКАЦИИ LDA:")
        print("   " + "-" * 50)
        
        if hasattr(lda_model, 'coef_') and len(numeric_columns) > 0:
            equations = _get_lda_equations(lda_model, numeric_columns, class_names)
            for eq in equations:
                print(eq)
            print("\n   📝 Примечание: Объект относится к классу с НАИБОЛЬШИМ значением δₖ(x)")
        
        # Шаг 5: Оценка модели
        print("\n5. ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
        print("-" * 40)
        print(f"   Точность (Accuracy): {accuracy:.4f}")
        
        print("\n   ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
        print("   " + "-" * 35)
        print(clf_report_df.to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
        
        # Внешняя валидация
        if external_validation and cv_dataset is not None:
            external_results = _perform_external_validation_lda(
                lda_model, scaler, label_encoder, numeric_columns,
                cv_dataset, target_column, index_level
            )
            
            if external_results:
                print(f"\n   Accuracy на валидационном датасете: {external_results['accuracy']:.4f}")
        
        # Шаг 6: Анализ важности признаков
        print("\n6. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("-" * 40)
        print("   Топ-10 самых важных признаков (по абсолютным коэффициентам LDA):")
        importance_col = 'importance'
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"      {i:2d}. {row['feature']:30s}: {row[importance_col]:.4f}")
        
        # Общая точность
        overall_accuracy = results_df['is_correct'].mean()
        print(f"\n   Общая точность на всем датасете: {overall_accuracy:.4f}")
        
        # Сводка
        print("\n" + "=" * 70)
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 70)
        print(f"   Классов: {batch_results['n_classes']}")
        print(f"   Признаков: {batch_results['n_features']}")
        print(f"   LDA компонентов: {n_components_actual}")
        print(f"   Точность на тесте: {accuracy:.4f}")
        print(f"   Самый важный признак: {feature_importance.iloc[0]['feature']}")
        print("=" * 70)
        
        # Получаем вывод консоли
        console_output = captured_output.getvalue()
        
        # Восстанавливаем stdout
        sys.stdout = old_stdout
        
        # Создание HTML отчета
        if output_html_path:
            try:
                create_lda_classification_html_report(
                    console_output=console_output,
                    results_df=results_df,
                    feature_importance=feature_importance,
                    lda_model=lda_model,
                    class_names=class_names,
                    label_encoder=label_encoder,
                    X_test=X_test,
                    y_test=y_test,
                    y_pred=lda_model.predict(X_test),
                    X_lda=X_train_lda,
                    y_train=y_train,
                    output_html_path=output_html_path,
                    vif_results=batch_results.get('vif_results'),
                    feature_names=numeric_columns,
                    cv_results=external_results
                )
                print(f"\n✅ HTML отчет сохранен: {output_html_path}")
            except Exception as e:
                print(f"⚠️  Не удалось создать HTML отчет: {e}")
        
        return (results_df, lda_model, scaler, feature_importance, label_encoder,
                X_train, X_test, y_train, y_test, X_train_lda, X_test_lda,
                accuracy, weighted_avg_accuracy, macro_avg_accuracy,
                clf_report_df, external_results)
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"❌ Ошибка при выполнении анализа: {e}")
        raise


def _perform_external_validation_lda(model, scaler, label_encoder, numeric_columns,
                                     cv_dataset, target_column, index_level):
    """
    Вспомогательная функция для внешней валидации LDA
    """
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    
    # Определяем целевую переменную
    if (target_column is None) and (index_level is not None):
        cv_target = cv_dataset.index.get_level_values(index_level)
        cv_features_df = cv_dataset.reset_index(drop=True)
    else:
        cv_target = cv_dataset[target_column]
        cv_features_df = cv_dataset.drop(columns=[target_column])
    
    # Кодируем целевую переменную
    known_mask = cv_target.isin(label_encoder.classes_)
    cv_target_filtered = cv_target[known_mask]
    cv_features_filtered = cv_features_df[known_mask]
    
    if len(cv_target_filtered) == 0:
        return None
    
    cv_target_encoded = label_encoder.transform(cv_target_filtered)
    
    # Выбираем общие признаки
    cv_numeric = cv_features_filtered.select_dtypes(include=[np.number])
    common_cols = list(set(numeric_columns).intersection(set(cv_numeric.columns)))
    
    if not common_cols:
        return None
    
    cv_features_numeric = cv_numeric[common_cols].values
    
    # Масштабируем
    cv_features_scaled = scaler.transform(cv_features_numeric)
    
    # Предсказываем
    cv_predictions = model.predict(cv_features_scaled)
    cv_predictions_proba = model.predict_proba(cv_features_scaled)
    
    # Метрики
    cv_accuracy = accuracy_score(cv_target_encoded, cv_predictions)
    cv_unique_classes = np.unique(cv_target_encoded)
    cv_present_names = label_encoder.classes_[cv_unique_classes]
    
    cv_clf_report = classification_report(
        cv_target_encoded, cv_predictions,
        target_names=cv_present_names,
        output_dict=True,
        labels=cv_unique_classes,
        zero_division=0
    )
    
    return {
        'features_scaled': cv_features_scaled,
        'target_encoded': cv_target_encoded,
        'predictions': cv_predictions,
        'predictions_proba': cv_predictions_proba,
        'accuracy': cv_accuracy,
        'classification_report': cv_clf_report,
        'present_classes': cv_present_names
    }
    
def create_lda_classification_html_report(console_output, results_df, feature_importance, 
                          lda_model, class_names, label_encoder,
                          X_test, y_test, y_pred, X_lda, y_train, output_html_path, 
                          vif_results=None, feature_names=None, cv_results=None):
    """Создание HTML отчета с результатами анализа LDA с поддержкой кросс-валидации"""
    
    # Создаем визуализации
    plt.style.use('default')
    
    # График важности признаков
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(15)
    print(top_features)
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
            .cv-badge {{
                display: inline-block;
                background: #28a745;
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 14px;
                margin-left: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Анализ Linear Discriminant Analysis (LDA)</h1>
                <p>Автоматический отчет по результатам анализа с помощью LDA</p>
                {f'<p class="cv-badge">С КРОСС-ВАЛИДАЦИЕЙ</p>' if cv_results else ''}
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
                <p><strong>Объясненная дисперсия:</strong> {lda_model.explained_variance_ratio_.sum():.2%}</p>
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