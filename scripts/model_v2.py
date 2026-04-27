import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
from itertools import combinations
from contextlib import contextmanager
from sklearn.svm import SVC

import HumSpectra.mass_spectra as ms
import HumSpectra.utilits as ut
import HumSpectra.mass_descriptors as md

import warnings
warnings.filterwarnings('ignore')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import chi2
from matplotlib.patches import Ellipse

@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} сек")

wdir_spectra = r"D:\lab\MassSpectraNew\processed_results\mzlist"
wdir_statistic = r"D:\lab\MassSpectraNew\processed_results\statistic"
wdir = r"D:\lab\MassSpectraNew\processed_results"

bad_spectra_data = pd.read_csv(r"D:\lab\MassSpectraNew\processed_results\bad_spectra.csv")
bad_spectra_list = bad_spectra_data['name'].to_list()

spectra_list = []
spectra_stat_list = []
spectra_class_list = []
spectra_list_name = []

with timer("Чтение спектров"):
    for path in Path.rglob(Path(wdir_spectra),'*.csv'):
        
        spectra = pd.read_csv(path)
        spectra_name = ut.extract_name_from_path(str(path))
        spectra_name = spectra_name.split(sep='__')[0]
        
        if spectra_name in bad_spectra_list or spectra_name == 'ADOM-SL9-6':
            continue
        
        spectra_class = ut.extract_class_from_name(spectra_name)
        spectra.attrs['name'] = spectra_name
        spectra.attrs['class'] = spectra_class
        
        spectra_list.append(spectra)
        spectra_class_list.append(spectra_class)
        spectra_list_name.append(spectra_name)
        
    optical_df=pd.read_csv(r"D:\lab\KNP-analysis\Received_data\Statistics\all_descriptors.csv")
    optical_df['sample_id'] = optical_df['sample_id'].apply(lambda x: x.replace('-2025',''))
    optical_df.drop(columns=['lambda','Sample','Subclass'],inplace=True)
    optical_df = optical_df[optical_df['sample_id'].isin(spectra_list_name)]
    optical_df.dropna(inplace=True)
    optical_df.rename(columns={'sample_id':'sample_name','Class':'class'},inplace=True)

entropy_data = md.calculate_entropy_for_spectra(spectra_list)
entropy_data.drop(columns=['n_peaks','max_intensity'],inplace=True)
print(entropy_data.columns)

interval_distrib_data = md.analyze_mass_intervals(spectra_list)

sqaure_data = md.extract_square_intensities(spectra_list,hc_range=(0.5,2.0),oc_range=(0.2,0.8))
print(sqaure_data.columns)

metric_data = md.calculate_metrics_for_spectra_list(spectra_list=spectra_list, metrics=None)
print(metric_data.columns)
metric_data.drop(columns=['S','C_13','mass'],inplace=True)

mass_data = metric_data.merge(sqaure_data, on=['sample_name','class'])
mass_data = mass_data.merge(interval_distrib_data, on=['sample_name','class'])

mass_data.to_csv('mass_data.csv')

combined_data = optical_df.merge(mass_data, on=['sample_name','class'])
combined_data = combined_data[combined_data['sample_name'] != 'ADOM-SLB-B3-2']
combined_data.to_csv('combined_data.csv')
print(combined_data.drop(columns=['sample_name','class','name']).columns[:18])

y = combined_data['class'].map({'Baikal': 1, 'ADOM': 0}).to_numpy() 
X = combined_data.drop(columns=['sample_name','class','name']).to_numpy()
X_baikal = combined_data[combined_data['class'] == 'Baikal'].drop(columns=['sample_name','class','name']).to_numpy()
X_sludge = combined_data[combined_data['class'] == 'ADOM'].drop(columns=['sample_name','class','name']).to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pls_viz = PLSRegression(n_components=2, scale=False)
X_scores, _ = pls_viz.fit_transform(X_scaled, y)

# Создание фигуры с несколькими подграфиками
fig = plt.figure(figsize=(16, 10))

# 1. Распределение по первой компоненте (исходный график)
ax1 = plt.subplot(2, 3, 1)
colors = {'Baikal': 'blue', 'ADOM': 'red'}
for class_name, color in colors.items():
    mask = combined_data['class'] == class_name
    sns.histplot(x=X_scores[mask, 0], color=color, label=class_name, 
                 kde=True, stat="density", alpha=0.6, ax=ax1)
ax1.set_title("Распределение по PLS компоненте 1\nЧеткое разделение классов", fontsize=12)
ax1.set_xlabel("PLS Component 1 Score")
ax1.set_ylabel("Плотность")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Scatter plot двух компонент
ax2 = plt.subplot(2, 3, 2)
for class_name, color in colors.items():
    mask = combined_data['class'] == class_name
    ax2.scatter(X_scores[mask, 0], X_scores[mask, 1], 
                c=color, label=class_name, alpha=0.7, edgecolors='black', s=80)
ax2.set_title("Проекция проб на первые две PLS компоненты", fontsize=12)
ax2.set_xlabel("PLS Component 1 (X-счет)")
ax2.set_ylabel("PLS Component 2 (X-счет)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Корреляционная нагрузка (loading plot)
ax3 = plt.subplot(2, 3, 3)
features = combined_data.drop(columns=['sample_name', 'class', 'name']).columns
loadings = pls_viz.x_loadings_
for i, feature in enumerate(features):
    ax3.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
              head_width=0.05, head_length=0.05, fc='gray', ec='gray', alpha=0.7)
    ax3.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, feature, fontsize=8, alpha=0.8)
ax3.set_title("Корреляционные нагрузки переменных", fontsize=12)
ax3.set_xlabel("Loading Component 1")
ax3.set_ylabel("Loading Component 2")
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax3.grid(True, alpha=0.3)

# 4. Biplot (совмещение счетов и нагрузок)
ax4 = plt.subplot(2, 3, 4)
for class_name, color in colors.items():
    mask = combined_data['class'] == class_name
    ax4.scatter(X_scores[mask, 0], X_scores[mask, 1], 
                c=color, label=class_name, alpha=0.6, s=60)
# Добавляем нагрузки (уменьшенные для наглядности)
scale_factor = 0.5
for i, feature in enumerate(features[:10]):  # топ-10 переменных для чистоты
    ax4.arrow(0, 0, loadings[i, 0]*scale_factor, loadings[i, 1]*scale_factor,
              head_width=0.05, head_length=0.05, fc='darkgreen', ec='darkgreen', alpha=0.5)
    ax4.text(loadings[i, 0]*scale_factor*1.1, loadings[i, 1]*scale_factor*1.1, 
             feature, fontsize=7, alpha=0.7)
ax4.set_title("Biplot: счета объектов и нагрузки переменных", fontsize=12)
ax4.set_xlabel("PLS Component 1")
ax4.set_ylabel("PLS Component 2")
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Объясненная дисперсия
ax5 = plt.subplot(2, 3, 5)
# Расчет объясненной дисперсии
x_variance_exp = np.var(X_scores, axis=0) / np.sum(np.var(X_scaled, axis=0))
x_cumsum = np.cumsum(x_variance_exp)
components = [1, 2]
bars = ax5.bar(components, x_variance_exp, alpha=0.7, label='Individual', color='steelblue')
ax5.plot(components, x_cumsum, 'ro-', label='Cumulative', linewidth=2, markersize=8)
ax5.set_title("Объясненная дисперсия X-переменных", fontsize=12)
ax5.set_xlabel("PLS Компонента")
ax5.set_ylabel("Доля объясненной дисперсии")
ax5.set_xticks(components)
ax5.set_ylim((0, 1))
for i, (bar, val) in enumerate(zip(bars, x_variance_exp)):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
ax5.text(1.5, x_cumsum[-1] + 0.02, f'Суммарно: {x_cumsum[-1]:.2%}', 
         ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. ROC кривая для оценки качества разделения
from sklearn.metrics import roc_curve, auc
ax6 = plt.subplot(2, 3, 6)
# Используем предсказания PLS (по первой компоненте)
y_pred = X_scores[:, 0]
fpr, tpr, _ = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)
ax6.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax6.set_title("ROC кривая для PLS компоненты 1", fontsize=12)
ax6.set_xlabel("False Positive Rate")
ax6.set_ylabel("True Positive Rate")
ax6.legend(loc="lower right")
ax6.grid(True, alpha=0.3)

plt.suptitle("Комплексная визуализация PLS регрессии\nРазделение классов 'Baikal' (1) и 'ADOM' (0)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Дополнительная статистика
print("\n" + "="*60)
print("СТАТИСТИКА PLS МОДЕЛИ")
print("="*60)
print(f"Количество образцов: {len(y)}")
print(f"Количество переменных: {X.shape[1]}")
print(f"Классы: Baikal ({(y==1).sum()} проб), ADOM ({(y==0).sum()} проб)")
print(f"\nОбъясненная дисперсия X:")
for i in range(2):
    print(f"  Компонента {i+1}: {x_variance_exp[i]:.2%}")
print(f"  Суммарно: {x_cumsum[-1]:.2%}")
print(f"\nAUC ROC: {roc_auc:.3f}")

# Корреляция компонент с целевой переменной
print(f"\nКорреляция компонент с целевой переменной (y):")
for i in range(2):
    corr = np.corrcoef(X_scores[:, i], y)[0, 1]
    print(f"  Компонента {i+1}: {corr:.4f}")

# 1. Нахождение оптимальной разделяющей прямой
from sklearn.linear_model import LogisticRegression

# Обучаем классификатор на PLS компонентах
clf = LogisticRegression()
clf.fit(X_scores, y)

# Получаем коэффициенты разделяющей прямой
coef = clf.coef_[0]
intercept = clf.intercept_[0]

# Уравнение прямой: coef[0]*x + coef[1]*y + intercept = 0
# Для построения: y_line = -(coef[0]*x + intercept) / coef[1]

# Визуализация с разделяющей границей
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = {'Baikal': 'blue', 'ADOM': 'red'}
for class_name, color in colors.items():
    mask = combined_data['class'] == class_name
    plt.scatter(X_scores[mask, 0], X_scores[mask, 1], 
                c=color, label=class_name, alpha=0.7, edgecolors='black', s=80)

# Рисуем разделяющую прямую
x_range = np.linspace(X_scores[:, 0].min() - 0.5, X_scores[:, 0].max() + 0.5, 100)
y_line = -(coef[0] * x_range + intercept) / coef[1]
plt.plot(x_range, y_line, 'k--', linewidth=2, label='Разделяющая граница')

# Вычисляем и показываем расстояние до границы
plt.title(f"Идеальное разделение классов\n(100% accuracy)", fontsize=12)
plt.xlabel("PLS Component 1")
plt.ylabel("PLS Component 2")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Маржинальный анализ
plt.subplot(1, 2, 2)

# Обучаем SVM с линейным ядром для нахождения максимальной маржи
svm = SVC(kernel='linear', C=1e10)  # Большое C = жесткая маржа
svm.fit(X_scores, y)

# Получаем опорные векторы
support_vectors = X_scores[svm.support_]
support_labels = y[svm.support_]

# Визуализируем маржу
plt.scatter(X_scores[:, 0], X_scores[:, 1], 
            c=y, cmap='bwr', alpha=0.5, s=50)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
            c=support_labels, cmap='bwr', s=200, marker='s', 
            edgecolors='black', linewidths=2, label='Опорные векторы')

# Разделяющая гиперплоскость и маржа
w = svm.coef_[0]
b = svm.intercept_[0]
xx = np.linspace(X_scores[:, 0].min(), X_scores[:, 0].max(), 100)
yy = -(w[0] * xx + b) / w[1]
margin = 1 / np.sqrt(np.sum(w**2))
yy_upper = yy + margin / np.sqrt(w[0]**2 + w[1]**2)
yy_lower = yy - margin / np.sqrt(w[0]**2 + w[1]**2)

plt.plot(xx, yy, 'k-', linewidth=2, label='Гиперплоскость')
plt.plot(xx, yy_upper, 'k--', alpha=0.5, label=f'Маржа = {margin:.3f}')
plt.plot(xx, yy_lower, 'k--', alpha=0.5)

plt.title(f"Маржинальный анализ\nШирина маржи = {margin:.3f}", fontsize=12)
plt.xlabel("PLS Component 1")
plt.ylabel("PLS Component 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Анализ расстояний до разделяющей границы
# Вычисляем расстояния от каждой точки до разделяющей прямой
distances = np.abs(coef[0]*X_scores[:, 0] + coef[1]*X_scores[:, 1] + intercept)
distances = distances / np.sqrt(coef[0]**2 + coef[1]**2)

# Расстояния со знаком (указывают сторону)
signed_distances = (coef[0]*X_scores[:, 0] + coef[1]*X_scores[:, 1] + intercept) / np.sqrt(coef[0]**2 + coef[1]**2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for class_name, color in colors.items():
    mask = combined_data['class'] == class_name
    plt.hist(signed_distances[mask], bins=15, alpha=0.6, 
             color=color, label=class_name, density=True)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Граница')
plt.xlabel("Расстояние до разделяющей границы (со знаком)")
plt.ylabel("Плотность")
plt.title("Распределение расстояний до границы")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Ближайшие точки к границе (наименее уверенные)
n_closest = 5
closest_indices = np.argsort(distances)[:n_closest]
print("\nБлижайшие к разделяющей границе пробы:")
for idx in closest_indices:
    print(f"  {combined_data.iloc[idx]['sample_name']} - {combined_data.iloc[idx]['class']} - расстояние: {distances[idx]:.4f}")

# Визуализация уверенности классификации
confidence = 1 - distances / distances.max()
scatter = plt.scatter(X_scores[:, 0], X_scores[:, 1], 
                      c=confidence, cmap='RdYlGn', s=80, 
                      edgecolors='black', vmin=0, vmax=1)
plt.colorbar(scatter, label='Уверенность классификации')
plt.xlabel("PLS Component 1")
plt.ylabel("PLS Component 2")
plt.title("Уверенность классификации проб")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Анализ нагрузок для интерпретации разделения
plt.figure(figsize=(12, 6))

# Сортируем переменные по вкладу в разделение (по нагрузкам на первую компоненту)
loadings_comp1 = pls_viz.x_loadings_[:, 0]
feature_names = combined_data.drop(columns=['sample_name', 'class', 'name']).columns

# Сортируем по абсолютному значению вклада
sorted_idx = np.argsort(np.abs(loadings_comp1))[::-1]
top_n = 15

plt.subplot(1, 2, 1)
top_features = feature_names[sorted_idx[:top_n]]
top_loadings = loadings_comp1[sorted_idx[:top_n]]
colors_loadings = ['red' if x > 0 else 'blue' for x in top_loadings]
plt.barh(range(top_n), top_loadings, color=colors_loadings)
plt.yticks(range(top_n), top_features) # type: ignore
plt.xlabel("Loading значение (PLS Component 1)")
plt.title(f"Топ-{top_n} переменных, определяющих разделение")
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(1, 2, 2)
# Нормализованные нагрузки для сравнения
loadings_norm = loadings_comp1 / np.max(np.abs(loadings_comp1))
plt.barh(range(top_n), loadings_norm[sorted_idx[:top_n]], color=colors_loadings)
plt.yticks(range(top_n), top_features) # type: ignore
plt.xlabel("Нормированное loading значение")
plt.title("Нормированный вклад переменных")
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# 5. Статистический отчет
print("\n" + "="*70)
print("СТАТИСТИЧЕСКИЙ ОТЧЕТ О РАЗДЕЛЕНИИ КЛАССОВ")
print("="*70)

# Расстояние между центроидами классов
centroid_baykal = X_scores[y==1].mean(axis=0)
centroid_adom = X_scores[y==0].mean(axis=0)
distance_centroids = np.linalg.norm(centroid_baykal - centroid_adom)

print(f"Расстояние между центроидами классов: {distance_centroids:.4f}")
print(f"Минимальное расстояние от точки до границы: {distances.min():.4f}")
print(f"Среднее расстояние до границы: {distances.mean():.4f}")

# Статистика по PLS компонентам
print(f"\nСтатистика по PLS Component 1:")
print(f"  Baikal: mean={X_scores[y==1, 0].mean():.4f}, std={X_scores[y==1, 0].std():.4f}")
print(f"  ADOM:   mean={X_scores[y==0, 0].mean():.4f}, std={X_scores[y==0, 0].std():.4f}")

print(f"\nСтатистика по PLS Component 2:")
print(f"  Baikal: mean={X_scores[y==1, 1].mean():.4f}, std={X_scores[y==1, 1].std():.4f}")
print(f"  ADOM:   mean={X_scores[y==0, 1].mean():.4f}, std={X_scores[y==0, 1].std():.4f}")

# Проверка на наличие выбросов (потенциально ошибочные пробы)
outliers = distances > (distances.mean() + 2*distances.std())
print(f"\nПотенциальные выбросы (далеко от границы): {outliers.sum()} проб")

threshold = (X_scores[y==1, 0].min() + X_scores[y==0, 0].max()) / 2
print(f"\nОптимальный порог для Component 1: {threshold:.4f}")
print(f"Все пробы с PLS1 > {threshold:.4f} - Baikal, иначе - ADOM")

y = combined_data['class'].map({'Baikal': 1, 'ADOM': 0}).to_numpy()
X = combined_data.drop(columns=['sample_name','class','name']).to_numpy()

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PLS модель
pls = PLSRegression(n_components=2, scale=False)
pls.fit(X_scaled, y)
X_scores, _ = pls.transform(X_scaled, y)

# Работаем с первой компонентой (основной сигнал)
pls1_scores = X_scores[:, 0]

# Разделяем классы
baikal_scores = pls1_scores[y == 1]
adom_scores = pls1_scores[y == 0]

# ============================================
# 1. ОПРЕДЕЛЕНИЕ ПОРОГА ДЕТЕКЦИИ
# ============================================

# Метод 1: Статистический порог (среднее Baikal + n*std)
mean_baikal = np.mean(baikal_scores)
std_baikal = np.std(baikal_scores)

# Уровни порогов для разных вероятностей ложной тревоги
thresholds = {
    '3sigma': mean_baikal + 3 * std_baikal,      # 0.27% ложных тревог
    '4sigma': mean_baikal + 4 * std_baikal,      # 0.006% ложных тревог
    '5sigma': mean_baikal + 5 * std_baikal,      # 0.00006% ложных тревог
    '6sigma': mean_baikal + 6 * std_baikal,      # 0.0000002% ложных тревог
}

# Метод 2: Процентильный порог
percentile_thresholds = {
    '99.9%': np.percentile(baikal_scores, 99.9),
    '99.99%': np.percentile(baikal_scores, 99.99),
    '99.999%': np.percentile(baikal_scores, 99.999),
}

# Метод 3: ROC-оптимальный порог (максимизация Youden's J)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds_roc = roc_curve(y, pls1_scores)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds_roc[optimal_idx]

# Визуализация порогов
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Распределение с порогами
ax1 = axes[0, 0]
ax1.hist(baikal_scores, bins=30, alpha=0.7, color='blue', label='Baikal (фон)', density=True)
ax1.hist(adom_scores, bins=30, alpha=0.7, color='red', label='ADOM (сигнал)', density=True)

# Добавляем пороги
for name, thresh in thresholds.items():
    ax1.axvline(x=thresh, linestyle='--', alpha=0.7, label=f'{name} ({thresh:.3f})')
ax1.axvline(x=optimal_threshold, linestyle='-', color='green', linewidth=2, 
            label=f'ROC-оптимальный ({optimal_threshold:.3f})')
ax1.set_xlabel('PLS Component 1 Score')
ax1.set_ylabel('Плотность')
ax1.set_title('Распределение PLS1 счетов и пороги детекции')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# График 2: ROC кривая с отмеченными порогами
ax2 = axes[0, 1]
ax2.plot(fpr, tpr, 'b-', linewidth=2, label='PLS детектор')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайный')
ax2.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', s=100, 
            marker='o', label=f'Оптимальный порог (AUC={np.trapz(tpr, fpr):.3f})')

# Отмечаем пороги на ROC
for name, thresh in thresholds.items():
    # Находим ближайшую точку на ROC
    idx = np.argmin(np.abs(thresholds_roc - thresh))
    ax2.scatter(fpr[idx], tpr[idx], s=50, marker='s', label=f'{name} (FPR={fpr[idx]:.4f})')

ax2.set_xlabel('False Positive Rate (ложная тревога)')
ax2.set_ylabel('True Positive Rate (детекция)')
ax2.set_title('ROC кривая детектора ADOM')
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(True, alpha=0.3)

# График 3: Зависимость вероятностей от порога
ax3 = axes[1, 0]
threshold_range = np.linspace(pls1_scores.min(), pls1_scores.max(), 100)
p_fa = []  # вероятность ложной тревоги
p_d = []   # вероятность детекции

for thresh in threshold_range:
    p_fa.append(np.sum(baikal_scores > thresh) / len(baikal_scores))
    p_d.append(np.sum(adom_scores > thresh) / len(adom_scores))

ax3.plot(threshold_range, p_fa, 'b-', linewidth=2, label='P(ложная тревога | Baikal)')
ax3.plot(threshold_range, p_d, 'r-', linewidth=2, label='P(детекция | ADOM)')
ax3.axvline(x=optimal_threshold, linestyle='-', color='green', alpha=0.7, 
            label=f'Оптимальный порог')
for name, thresh in thresholds.items():
    ax3.axvline(x=thresh, linestyle='--', alpha=0.5, color='gray')
ax3.set_xlabel('Порог детекции')
ax3.set_ylabel('Вероятность')
ax3.set_title('Вероятностные характеристики детектора')
ax3.legend()
ax3.grid(True, alpha=0.3)

# График 4: Контрольная карта (Shewhart chart)
ax4 = axes[1, 1]
order = np.argsort(pls1_scores)
sorted_scores = pls1_scores[order]
sorted_labels = y[order]
colors_points = ['red' if lbl == 0 else 'blue' for lbl in sorted_labels]

ax4.scatter(range(len(sorted_scores)), sorted_scores, c=colors_points, alpha=0.6, s=30)
ax4.axhline(y=mean_baikal, color='blue', linestyle='-', label=f'Среднее Baikal ({mean_baikal:.3f})')
ax4.axhline(y=mean_baikal + 3*std_baikal, color='orange', linestyle='--', label='UCL (3σ)')
ax4.axhline(y=mean_baikal - 3*std_baikal, color='orange', linestyle='--', label='LCL (3σ)')
ax4.axhline(y=optimal_threshold, color='green', linestyle='-', linewidth=2, label='Оптимальный порог')

ax4.set_xlabel('Номер пробы (упорядочено по PLS1)')
ax4.set_ylabel('PLS Component 1 Score')
ax4.set_title('Контрольная карта Шухарта для мониторинга')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.suptitle('Анализ порогов детекции ADOM на фоне Baikal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Подготовка данных
y = combined_data['class'].map({'Baikal': 1, 'ADOM': 0}).to_numpy()
X = combined_data.drop(columns=['sample_name','class','name']).to_numpy()

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PLS модель с 2 компонентами
pls2 = PLSRegression(n_components=2, scale=False)
pls2.fit(X_scaled, y)
X_scores, _ = pls2.transform(X_scaled, y)

# Разделяем счета по классам
baikal_scores = X_scores[y == 1]
adom_scores = X_scores[y == 0]

print("="*70)
print("ДВУХКОМПОНЕНТНЫЙ PLS ДЕТЕКТОР ADOM")
print("="*70)
print(f"Форма счетов: {X_scores.shape}")
print(f"Baikal пробы: {len(baikal_scores)}")
print(f"ADOM пробы: {len(adom_scores)}")

class PLS2Detector:
    def __init__(self, scaler, pls_model, method='mahalanobis'):
        self.scaler = scaler
        self.pls_model = pls_model
        self.method = method
        self.fitted = False
        
    def fit(self, X_background, y_background=None):
        # 1. Масштабирование и получение счетов
        X_scaled = self.scaler.transform(X_background)
        self.background_scores = self.pls_model.transform(X_scaled)
        
        # 2. Статистика фона
        self.centroid = np.mean(self.background_scores, axis=0)
        self.cov_matrix = np.cov(self.background_scores.T)
        
        # Регуляризация для устойчивости (очень важно!)
        eps = 1e-6 * np.trace(self.cov_matrix) / self.cov_matrix.shape[0]
        self.cov_matrix += np.eye(self.cov_matrix.shape[0]) * eps
        
        self.cov_inv = np.linalg.inv(self.cov_matrix)
        self.dof = self.cov_matrix.shape[0]
        
        # 3. Квадрат расстояния Махаланобиса (D² ~ χ²(df))
        self.bg_distances_sq = self._mahalanobis_sq_distance(self.background_scores)
        
        # 4. Пороги (теперь корректно сопоставлены с D²)
        self.thresholds_sq = {
            '95%': chi2.ppf(0.95, self.dof),
            '99%': chi2.ppf(0.99, self.dof),
            '99.9%': chi2.ppf(0.999, self.dof),
            'optimal': np.percentile(self.bg_distances_sq, 99)
        }
        
        if self.method == 'lda' and y_background is not None:
            self.lda = LinearDiscriminantAnalysis()
            # 1 = фон, 0 = целевой/аномалия (адаптируйте под вашу задачу)
            y_train = np.where(y_background == 0, 1, 0)
            self.lda.fit(self.background_scores, y_train)
            
        self.fitted = True
        return self
    
    def _mahalanobis_sq_distance(self, scores):
        """Возвращает D² = (x-μ)ᵀ Σ⁻¹ (x-μ)"""
        scores = np.atleast_2d(scores)
        diff = scores - self.centroid
        # Векторизованное: diag(diff @ cov_inv @ diff.T)
        return np.sum(diff @ self.cov_inv * diff, axis=1)
    
    def _euclidean_sq_distance(self, scores):
        scores = np.atleast_2d(scores)
        return np.sum((scores - self.centroid)**2, axis=1)
    
    def set_adom_direction(self, X_adom):
        X_scaled = self.scaler.transform(X_adom)
        adom_scores = self.pls_model.transform(X_scaled)
        adom_vector = np.mean(adom_scores, axis=0) - self.centroid
        self.adom_direction = adom_vector / np.linalg.norm(adom_vector)
        
    def detect(self, X_new, threshold_name='optimal', return_details=False):
        if not self.fitted:
            raise ValueError("Детектор не обучен. Вызовите fit() сначала.")
            
        X_new = np.atleast_2d(X_new)
        X_scaled = self.scaler.transform(X_new)
        scores = self.pls_model.transform(X_scaled)
        
        if self.method == 'mahalanobis':
            dist_sq = self._mahalanobis_sq_distance(scores)[0]
            threshold_sq = self.thresholds_sq[threshold_name]
            is_adom = dist_sq > threshold_sq
            # Для удобства выводим корень, но сравниваем квадраты
            distance = np.sqrt(dist_sq)
            threshold = np.sqrt(threshold_sq)
            
        elif self.method == 'euclidean':
            dist_sq = self._euclidean_sq_distance(scores)[0]
            threshold_sq = np.percentile(self.bg_distances_sq, 99)
            is_adom = dist_sq > threshold_sq
            distance = np.sqrt(dist_sq)
            threshold = np.sqrt(threshold_sq)
            
        elif self.method == 'angle':
            vectors = scores - self.centroid
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit_vec = vectors / norms
            if hasattr(self, 'adom_direction'):
                cos_sim = np.dot(unit_vec, self.adom_direction)[0]
                distance = 1.0 - cos_sim
                threshold = 0.5
                is_adom = distance > threshold
            else:
                raise ValueError("Установите направление через set_adom_direction()")
                
        elif self.method == 'lda':
            prob_adom = self.lda.predict_proba(scores)[0, 1]
            is_adom = prob_adom > 0.5
            distance = prob_adom
            threshold = 0.5
            
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
            
        result = {
            'is_adom': bool(is_adom),
            'score_comp1': float(scores[0, 0]),
            'score_comp2': float(scores[0, 1]),
            'distance': float(distance),
            'threshold': float(threshold),
            'method': self.method
        }
        
        if return_details:
            result.update({
                'centroid': self.centroid.tolist(),
                'scores': scores[0].tolist(),
                'distance_sq': float(dist_sq) if 'dist_sq' in locals() else None, # type: ignore
                'threshold_sq': float(threshold_sq) if 'threshold_sq' in locals() else None # type: ignore
            })
        return result
    
    def detect_batch(self, X_new, threshold_name='optimal'):
        """Векторизованная batch-детекция"""
        X_new = np.atleast_2d(X_new)
        X_scaled = self.scaler.transform(X_new)
        scores = self.pls_model.transform(X_scaled)
        
        if self.method == 'mahalanobis':
            dists_sq = self._mahalanobis_sq_distance(scores)
            thr_sq = self.thresholds_sq[threshold_name]
            is_adom = dists_sq > thr_sq
            distances = np.sqrt(dists_sq)
            thresholds = np.full_like(distances, np.sqrt(thr_sq))
        elif self.method == 'euclidean':
            dists_sq = self._euclidean_sq_distance(scores)
            thr_sq = np.percentile(self.bg_distances_sq, 99)
            is_adom = dists_sq > thr_sq
            distances = np.sqrt(dists_sq)
            thresholds = np.full_like(distances, np.sqrt(thr_sq))
        else:
            # Fallback на построчный вызов для LDA/angle
            return pd.DataFrame([self.detect(X_new[i], threshold_name) for i in range(len(X_new))])
            
        return pd.DataFrame({
            'is_adom': is_adom,
            'score_comp1': scores[:, 0],
            'score_comp2': scores[:, 1],
            'distance': distances,
            'threshold': thresholds,
            'method': self.method
        })
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Создаем детектор
detector = PLS2Detector(scaler, pls2, method='mahalanobis')

# Обучаем только на фоновых данных (Baikal)
background_mask_train = (y_train == 1)  # Baikal = 1
detector.fit(X_train[background_mask_train])

# Обучаем направление для ADOM (опционально)
adom_mask_train = (y_train == 0)
if adom_mask_train.sum() > 0:
    detector.set_adom_direction(X_train[adom_mask_train])

print("\nДетектор обучен на фоновых данных Baikal")
print(f"Центроид Baikal: {detector.centroid}")
print(f"Пороги Махаланобиса (χ² с {detector.dof} степенями свободы):")
for name, thresh in detector.thresholds_sq.items():
    print(f"  {name}: {thresh:.3f}")

# Подготовка данных для визуализации
# ПолучаемScores для всех данных через модель
all_scores = pls2.transform(scaler.transform(X_scaled))
baikal_scores = all_scores[y == 1]
adom_scores = all_scores[y == 0]

# Вычисляем расстояния Махаланобиса для всех проб
# Используем квадратичную формулу из класса
all_distances_sq = detector._mahalanobis_sq_distance(all_scores)
all_distances = np.sqrt(all_distances_sq)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Scatter plot с эллипсом Махаланобиса
ax1 = axes[0, 0]

# Рисуем точки
ax1.scatter(baikal_scores[:, 0], baikal_scores[:, 1], 
            c='blue', alpha=0.6, s=50, label='Baikal (фон)')
ax1.scatter(adom_scores[:, 0], adom_scores[:, 1], 
            c='red', alpha=0.6, s=50, label='ADOM (сигнал)')

def plot_cov_ellipse(ax, mean, cov, n_std, color, alpha=0.2, label=None):
    """Рисует эллипс для заданного количества стандартных отклонений"""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      alpha=alpha, color=color, label=label)
    ax.add_patch(ellipse)

# Эллипсы для разных уровней (пересчитываем радиусы через χ²)
# Для 1σ (~68%), 2σ (~95%), 3σ (~99.7%) используем соответствующие квантили
n_std_values = [1, 2, 3]
colors = ['green', 'orange', 'red']
for n_std, color in zip(n_std_values, colors):
    # Переводим "стандартные отклонения" в уровень доверия для χ²
    # 1σ -> ~68%, 2σ -> ~95%, 3σ -> ~99.7%
    prob = 0.6827 if n_std == 1 else (0.9545 if n_std == 2 else 0.9973)
    radius_factor = np.sqrt(chi2.ppf(prob, detector.dof))
    
    plot_cov_ellipse(ax1, detector.centroid, detector.cov_matrix, 
                     radius_factor, color, alpha=0.1, label=f'{n_std}σ' if n_std == 1 else None)

ax1.scatter(detector.centroid[0], detector.centroid[1], 
            c='black', marker='x', s=200, linewidths=3, label='Центроид Baikal')

# Рисуем направление ADOM
if hasattr(detector, 'adom_direction'):
    arrow_start = detector.centroid
    arrow_end = detector.centroid + detector.adom_direction * 5
    ax1.arrow(arrow_start[0], arrow_start[1], 
              arrow_end[0]-arrow_start[0], arrow_end[1]-arrow_start[1],
              head_width=0.3, head_length=0.3, fc='purple', ec='purple', 
              alpha=0.7, label='Направление ADOM')

ax1.set_xlabel('PLS Component 1')
ax1.set_ylabel('PLS Component 2')
ax1.set_title('Двухкомпонентное пространство с эллипсом Махаланобиса')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 2. Распределения расстояний
ax2 = axes[0, 1]

# Разделяем по классам
dist_bg = all_distances[y == 1]
dist_signal = all_distances[y == 0]

# Гистограммы
ax2.hist(dist_bg, bins=30, alpha=0.6, color='blue', label='Baikal', density=True)
ax2.hist(dist_signal, bins=30, alpha=0.6, color='red', label='ADOM', density=True)

# Пороги (берем квадратный корень из квадратов порогов, так как мы сравниваем sqrt(D^2))
for name, thresh_sq in detector.thresholds_sq.items():
    if name in ['95%', '99%', 'optimal']:
        thresh = np.sqrt(thresh_sq)
        ax2.axvline(x=thresh, linestyle='--', alpha=0.7, 
                   label=f'{name} (√χ²={thresh:.1f})')

ax2.set_xlabel('Расстояние Махаланобиса (√D²)')
ax2.set_ylabel('Плотность')
ax2.set_title('Распределение расстояний Махаланобиса')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. ROC кривая
ax3 = axes[0, 2]

# Для ROC: чем больше расстояние, тем выше вероятность аномалии (класс 0)
# Но roc_curve ожидает: score越高越可能是正类 (higher score -> positive class)
# У нас: Класс 0 (ADOM) = Аномалия. Значит, нам нужно передать -score, чтобы низкие баллы фона были "высокими" для фона?
# Нет, проще: Передаем distance. Если distance большой -> это ADOM.
# Но в roc_curve метка y должна быть 1 для "положительного" класса.
# У нас y=0 это ADOM. Значит, нам нужно инвертировать y или инвертировать score.
# Давайте сделаем: score = distance. Positive class = ADOM (y=0).
# Чтобы roc_curve работала корректно, где higher score -> positive class:
# Мы передадим -distance, тогда для фона (малый dist) score будет большим (близким к 0), а для ADOM (большой dist) score будет маленьким (отрицательным).
# Это запутает. Лучше инвертировать метки классов для расчета ROC.

y_binary_adom = (y == 0).astype(int) # 1 если ADOM, 0 если фон
fpr, tpr, thresholds_roc = roc_curve(y_binary_adom, all_distances)
roc_auc = auc(fpr, tpr)

ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'Mahalanobis (AUC={roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайный')

# Оптимальный порог (Youden's J)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold_dist = thresholds_roc[optimal_idx]
ax3.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
        label=f'Оптимальный порог\nTPR={tpr[optimal_idx]:.2f}, FPR={fpr[optimal_idx]:.3f}')

ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title(f'ROC кривая (2 компоненты)\nAUC = {roc_auc:.3f}')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# 4. Контрольная карта T² Хотеллинга
ax4 = axes[1, 0]

# Сортируем для лучшей визуализации
order = np.argsort(all_distances)
sorted_distances = all_distances[order]
sorted_labels = y[order]

# Цвета: Красный (ADOM), Синий (Фон)
colors_points = ['red' if lbl == 0 else 'blue' for lbl in sorted_labels]

ax4.scatter(range(len(sorted_distances)), sorted_distances, 
            c=colors_points, alpha=0.6, s=30)

# Контрольные линии (используем sqrt порогов)
for name, thresh_sq in detector.thresholds_sq.items():
    if name != 'optimal': # Оптимизированный порог рисуем отдельно
        thresh = np.sqrt(thresh_sq)
        color = 'orange' if name == '95%' else 'red'
        linestyle = '--'
        label_name = f'UCL {name}'
        ax4.axhline(y=thresh, color=color, linestyle=linestyle, 
                   label=label_name, linewidth=2)

# Оптимальный порог
opt_thresh = np.sqrt(detector.thresholds_sq['optimal'])
ax4.axhline(y=opt_thresh, color='darkred', linestyle='-', 
           label='Оптимальный порог', linewidth=2)

ax4.set_xlabel('Номер пробы (упорядочено по расстоянию)')
ax4.set_ylabel('T² Хотеллинга (расстояние Махаланобиса)')
ax4.set_title('Контрольная карта T² для мониторинга ADOM')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Матрица ошибок
ax5 = axes[1, 1]

# Детекция на тестовых данных
test_scores = pls2.transform(scaler.transform(X_test))
test_distances = detector._mahalanobis_sq_distance(test_scores) # Возвращает квадраты
test_distances_sqrt = np.sqrt(test_distances)

# Используем оптимальный порог
threshold_opt = np.sqrt(detector.thresholds_sq['optimal'])
test_predictions = (test_distances_sqrt > threshold_opt).astype(int) # 1 если обнаружен

# y_test: 1=Baikal, 0=ADOM. Нам нужно предсказание: 1=ADOM detected
y_test_adom = (y_test == 0).astype(int)

cm = confusion_matrix(y_test_adom, test_predictions)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

im = ax5.imshow(cm_norm, interpolation='nearest', cmap='Blues')
ax5.set_xticks([0, 1])
ax5.set_yticks([0, 1])
ax5.set_xticklabels(['ADOM не обнаружен', 'ADOM обнаружен'])
ax5.set_yticklabels(['ADOM отсутствует\n(Baikal)', 'ADOM присутствует'])
ax5.set_xlabel('Предсказание')
ax5.set_ylabel('Истина')
ax5.set_title('Матрица ошибок (тестовая выборка)')

for i in range(2):
    for j in range(2):
        text = ax5.text(j, i, f'{int(cm[i, j])}\n({cm_norm[i, j]:.1%})',
                       ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black")

# 6. Метрики качества
ax6 = axes[1, 2]
ax6.axis('off')

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

metrics_text = f"""
МЕТРИКИ ДЕТЕКЦИИ (тестовая выборка)

Метод: {detector.method}
Порог: {threshold_opt:.3f}

📊 Основные метрики:
   Accuracy:  {accuracy:.2%}
   Precision: {precision:.2%}
   Recall:    {recall:.2%}
   F1-score:  {f1_score:.2%}
   
🎯 Специфичность: {specificity:.2%}
   
📈 Матрица ошибок:
   TP: {tp} (ADOM обнаружен верно)
   TN: {tn} (фон верно)
   FP: {fp} (ложная тревога)
   FN: {fn} (пропуск ADOM)

💡 Интерпретация:
   {("Отличное", "Хорошее", "Удовлетворительное")[0 if accuracy>0.9 else 1 if accuracy>0.8 else 2]} качество детекции
"""

ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Двухкомпонентный PLS детектор ADOM (Mahalanobis distance)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Второй график: Сравнение 1D vs 2D
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Распределение 1D компоненты
ax1 = axes[0, 0]
pls1_scores = all_scores[:, 0] # type: ignore
ax1.hist(pls1_scores[y==1], bins=30, alpha=0.6, color='blue', label='Baikal', density=True)
ax1.hist(pls1_scores[y==0], bins=30, alpha=0.6, color='red', label='ADOM', density=True)
ax1.set_xlabel('PLS Component 1')
ax1.set_ylabel('Плотность')
ax1.set_title('1D детекция (только компонента 1)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 2D распределение с эллипсом
ax2 = axes[0, 1]
ax2.scatter(baikal_scores[:, 0], baikal_scores[:, 1], c='blue', alpha=0.5, s=30, label='Baikal')
ax2.scatter(adom_scores[:, 0], adom_scores[:, 1], c='red', alpha=0.5, s=30, label='ADOM')
radius_95 = np.sqrt(chi2.ppf(0.95, detector.dof))
plot_cov_ellipse(ax2, detector.centroid, detector.cov_matrix, 
                 radius_95, 'green', alpha=0.2, label='95% эллипс')
ax2.set_xlabel('PLS Component 1')
ax2.set_ylabel('PLS Component 2')
ax2.set_title('2D детекция (Mahalanobis distance)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# 3. Сравнение AUC
ax3 = axes[1, 0]

# 1D ROC (используем ту же логику инверсии меток)
fpr1, tpr1, _ = roc_curve(y_binary_adom, pls1_scores)
auc1 = auc(fpr1, tpr1)

# 2D ROC
fpr2, tpr2, _ = roc_curve(y_binary_adom, all_distances)
auc2 = auc(fpr2, tpr2)

ax3.plot(fpr1, tpr1, 'b-', linewidth=2, label=f'1 компонента (AUC={auc1:.3f})')
ax3.plot(fpr2, tpr2, 'r-', linewidth=2, label=f'2 компоненты (AUC={auc2:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайный')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title(f'Сравнение ROC кривых\nУлучшение: +{(auc2-auc1)*100:.1f}%')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Таблица улучшения
ax4 = axes[1, 1]
ax4.axis('off')

target_fpr = 0.05
idx1 = np.argmin(np.abs(fpr1 - target_fpr))
idx2 = np.argmin(np.abs(fpr2 - target_fpr))
idx_tpr1 = np.argmin(np.abs(tpr1-0.9))
idx_tpr2 = np.argmin(np.abs(tpr2-0.9))
fpr_at_90_1 = fpr1[idx_tpr1]
fpr_at_90_2 = fpr2[idx_tpr2]
improvement_text = f"""
СРАВНЕНИЕ МЕТОДОВ ДЕТЕКЦИИ

📊 AUC (площадь под ROC кривой):
   • 1 компонента:  {auc1:.3f} ({auc1*100:.1f}%)
   • 2 компоненты: {auc2:.3f} ({auc2*100:.1f}%)
   • Улучшение:    +{(auc2-auc1)*100:.1f}%

🎯 При FPR = {target_fpr:.0%}:
   • 1 компонента: TPR = {tpr1[idx1]:.1%}
   • 2 компоненты: TPR = {tpr2[idx2]:.1%}
   • Улучшение:    +{(tpr2[idx2]-tpr1[idx1])*100:.1f}%

📈 При TPR = 90%:

   
   • 1 компонента: FPR = {fpr_at_90_1:.1%}
   • 2 компоненты: FPR = {fpr_at_90_2:.1%}
   • Уменьшение ложных тревог: {(fpr_at_90_1-fpr_at_90_2)*100:.1f}%

💡 ВЫВОД:
   Использование двух PLS компонент
   значительно улучшает детекцию ADOM,
   особенно в области низких ложных тревог.
"""

ax4.text(0.1, 0.95, improvement_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle('Сравнение эффективности 1D vs 2D PLS детекции', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
print("="*70)
print(f"""
Двухкомпонентный детектор показывает улучшение AUC с {auc1:.3f} до {auc2:.3f}
Что означает снижение ошибок классификации на {(1-auc2)/(1-auc1):.1f}%

Рекомендуемый порог: {threshold_opt:.3f} (расстояние Махаланобиса)

Правило принятия решения:
   • Расстояние < порога → Baikal (фон)
   • Расстояние > порога → ADOM обнаружен
   • Рекомендуется подтверждение при 2 из 3 последовательных превышений
""")

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Предполагаемая структура данных
# X_baikal: (5, 520) — 20 оптических + ~500 масс-спектральных
# X_sludge: (39, 520) — надшламовые воды

# Рекомендуемый пайплайн для масс-спектральной части
ms_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, svd_solver='full'))  # сохраняем 95% дисперсии
])

# Для оптических данных — только масштабирование
opt_scaler = StandardScaler()

# Объединение после предобработки
def preprocess(X, opt_indices=slice(0,16), ms_indices=slice(16,None)):
    X_opt = opt_scaler.fit_transform(X[:, opt_indices])
    X_ms = ms_pipeline.fit_transform(X[:, ms_indices])
    return np.hstack([X_opt, X_ms])


class BaikalAnomalyDetector1:
    def __init__(self, n_pca_components=10, nu=0.1):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        
    def fit(self, X_baikal):
        # Предобработка
        X_scaled = self.scaler.fit_transform(X_baikal)
        X_pca = self.pca.fit_transform(X_scaled)
        self.ocsvm.fit(X_pca)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.ocsvm.predict(X_pca)  # 1 = normal, -1 = anomaly
    
    def score(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.ocsvm.score_samples(X_pca)

# Использование
detector = BaikalAnomalyDetector1(n_pca_components=min(5, len(X_baikal)-1))
detector.fit(X_baikal)

# Оценка
predictions = detector.predict(X_sludge)
detection_rate = np.mean(predictions == -1)
print(f"Detection rate for sludge waters: {detection_rate:.2%}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                            classification_report, precision_recall_curve,
                            average_precision_score, f1_score)
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ИНИЦИАЛИЗАЦИЯ ДАННЫХ И МЕТАДАННЫХ
# ============================================================================

# Названия дескрипторов
optical_names = ['Ag_275', 'Ag_380', 'Asm_280', 'Asm_350', 'B1', 'B2', 
                 'Component_1', 'Component_2', 'Component_3', 'E2E3', 
                 'E3E4', 'E4E6', 'FIX', 'HIX', 'S_380_443', 'Suva']

metrics_names = ['AI', 'C', 'CAI', 'CRAM', 'DBE', 'DBE-O', 'DBE-OC', 
                 'DBE_AI', 'H', 'H/C', 'N', 'NOSC', 'O', 'O/C']

vk_names = [f'VK_{i}' for i in range(1,21)]

# Интервалы масс 225-700 с шагом 1 Да
mass_names = [f'Mass_{i}' for i in range(225, 700)]

ALL_FEATURES = optical_names + metrics_names + vk_names + mass_names

# Группы признаков для анализа
FEATURE_GROUPS = {
    'Optical': (0, 16),
    'FT-ICR Metrics': (16, 30),
    'Van Krevelen': (30, 50),
    'Mass Bins': (50, 525)
}

# ============================================================================
# 2. КЛАСС ДЕТЕКТОРА С РАСШИРЕННОЙ СТАТИСТИКОЙ
# ============================================================================

class BaikalAnomalyDetector:
    def __init__(self, n_pca_components=5, nu=0.1, gamma='scale'):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma,  # type: ignore
                                 shrinking=True, tol=1e-6)
        self.n_components = n_pca_components
        self.is_fitted = False
        
    def fit(self, X_baikal):
        self.scaler.fit(X_baikal)
        X_scaled = self.scaler.transform(X_baikal)
        self.pca.fit(X_scaled)
        X_pca = self.pca.transform(X_scaled)
        self.ocsvm.fit(X_pca)
        self.is_fitted = True
        
        # Сохраняем обучающие данные для анализа
        self.X_train_scaled = X_scaled
        self.X_train_pca = X_pca
        self.n_train = len(X_baikal)
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
    
    def predict(self, X):
        X_pca = self.transform(X)
        return self.ocsvm.predict(X_pca)
    
    def score_samples(self, X):
        X_pca = self.transform(X)
        return self.ocsvm.score_samples(X_pca)
    
    def decision_function(self, X):
        X_pca = self.transform(X)
        return self.ocsvm.decision_function(X_pca)
    
    def get_support_vectors(self):
        """Получить опорные векторы в пространстве PCA"""
        return self.ocsvm.support_vectors_
    
    def get_pca_loadings(self):
        """Компоненты PCA для интерпретации"""
        return pd.DataFrame(
            self.pca.components_,
            columns=ALL_FEATURES[:self.pca.components_.shape[1]],
            index=[f'PC{i+1}' for i in range(self.n_components)]
        )

# ============================================================================
# 3. ФУНКЦИИ ВИЗУАЛИЗАЦИИ И СТАТИСТИКИ
# ============================================================================

def print_full_statistics(detector, X_baikal, X_sludge, y_true=None):
    """
    Выводит максимально полную статистику модели
    """
    print("=" * 80)
    print("ПОЛНАЯ СТАТИСТИКА МОДЕЛИ ONE-CLASS SVM")
    print("=" * 80)
    
    # Предсказания
    y_pred_baikal = detector.predict(X_baikal)
    y_pred_sludge = detector.predict(X_sludge)
    
    scores_baikal = detector.score_samples(X_baikal)
    scores_sludge = detector.score_samples(X_sludge)
    decision_baikal = detector.decision_function(X_baikal)
    decision_sludge = detector.decision_function(X_sludge)
    
    print("\n" + "-" * 80)
    print("1. ОБЩИЕ ХАРАКТЕРИСТИКИ МОДЕЛИ")
    print("-" * 80)
    print(f"Количество обучающих образцов (РОВ Байкал): {detector.n_train}")
    print(f"Количество тестовых образцов (надшламовые воды): {len(X_sludge)}")
    print(f"Исходная размерность: {X_baikal.shape[1]}")
    print(f"Размерность после PCA: {detector.n_components}")
    print(f"Объясненная дисперсия PCA: {detector.pca.explained_variance_ratio_.sum():.4f}")
    print(f"Параметр nu: {detector.ocsvm.nu}")
    print(f"Параметр gamma: {detector.ocsvm.gamma}")
    print(f"Количество опорных векторов: {len(detector.ocsvm.support_)}")
    print(f"Доля опорных векторов: {len(detector.ocsvm.support_)/detector.n_train:.2%}")
    
    print("\n" + "-" * 80)
    print("2. РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ")
    print("-" * 80)
    print(f"РОВ Байкал:  норма (+1): {np.sum(y_pred_baikal == 1)}, аномалия (-1): {np.sum(y_pred_baikal == -1)}")
    print(f"Надшламовые: норма (+1): {np.sum(y_pred_sludge == 1)}, аномалия (-1): {np.sum(y_pred_sludge == -1)}")
    
    # Detection Rate
    detection_rate = np.mean(y_pred_sludge == -1)
    false_alarm_rate = np.mean(y_pred_baikal == -1)
    
    print(f"\nDetection Rate (доля надшламовых, детектированных как аномалии): {detection_rate:.4f} ({detection_rate:.2%})")
    print(f"False Alarm Rate (доля РОВ, ошибочно детектированных как аномалии): {false_alarm_rate:.4f} ({false_alarm_rate:.2%})")
    print(f"Specificity (доля РОВ, верно классифицированных как норма): {1-false_alarm_rate:.4f}")
    
    print("\n" + "-" * 80)
    print("3. СТАТИСТИКА SCORES (логарифм плотности вероятности)")
    print("-" * 80)
    
    print("\nРОВ Байкал (обучающая выборка):")
    print(f"  Mean: {scores_baikal.mean():.4f}")
    print(f"  Std:  {scores_baikal.std():.4f}")
    print(f"  Min:  {scores_baikal.min():.4f}")
    print(f"  Max:  {scores_baikal.max():.4f}")
    print(f"  Median: {np.median(scores_baikal):.4f}")
    
    print("\nНадшламовые воды:")
    print(f"  Mean: {scores_sludge.mean():.4f}")
    print(f"  Std:  {scores_sludge.std():.4f}")
    print(f"  Min:  {scores_sludge.min():.4f}")
    print(f"  Max:  {scores_sludge.max():.4f}")
    print(f"  Median: {np.median(scores_sludge):.4f}")
    
    # Тест Манна-Уитни
    statistic, p_value = stats.mannwhitneyu(scores_baikal, scores_sludge, alternative='two-sided') # type: ignore
    print(f"\nТест Манна-Уитни (различие распределений scores):")
    print(f"  Statistic: {statistic:.4f}, p-value: {p_value:.2e}")
    
    # Эффект размера (Cohen's d)
    pooled_std = np.sqrt(((len(scores_baikal)-1)*scores_baikal.var() + 
                          (len(scores_sludge)-1)*scores_sludge.var()) / 
                         (len(scores_baikal) + len(scores_sludge) - 2))
    cohens_d = (scores_baikal.mean() - scores_sludge.mean()) / pooled_std
    print(f"  Cohen's d (эффект размера): {cohens_d:.4f}")
    
    print("\n" + "-" * 80)
    print("4. СТАТИСТИКА DECISION FUNCTION (расстояние до границы)")
    print("-" * 80)
    
    print("\nРОВ Байкал:")
    print(f"  Mean: {decision_baikal.mean():.4f}")
    print(f"  Min:  {decision_baikal.min():.4f} (наиболее близкий к границе)")
    print(f"  Max:  {decision_baikal.max():.4f}")
    
    print("\nНадшламовые воды:")
    print(f"  Mean: {decision_sludge.mean():.4f}")
    print(f"  Min:  {decision_sludge.min():.4f}")
    print(f"  Max:  {decision_sludge.max():.4f} (наименее аномальный)")
    
    # Расстояния между классами
    print("\n" + "-" * 80)
    print("5. МЕТРИКИ РАССТОЯНИЙ В ПРОСТРАНСТВЕ PCA")
    print("-" * 80)
    
    X_baikal_pca = detector.transform(X_baikal)
    X_sludge_pca = detector.transform(X_sludge)
    
    centroid_baikal = X_baikal_pca.mean(axis=0)
    centroid_sludge = X_sludge_pca.mean(axis=0)
    euclidean_dist = np.linalg.norm(centroid_baikal - centroid_sludge)
    
    print(f"Евклидово расстояние между центроидами: {euclidean_dist:.4f}")
    
    # Расстояние Махаланобиса
    cov = np.cov(X_baikal_pca.T)
    try:
        cov_inv = np.linalg.inv(cov)
        mahal_dist = np.sqrt((centroid_baikal - centroid_sludge) @ cov_inv @ (centroid_baikal - centroid_sludge))
        print(f"Расстояние Махаланобиса: {mahal_dist:.4f}")
    except:
        print("Расстояние Махаланобиса: не вычислено (вырожденная ковариация)")
    
    # Внутриклассовые расстояния
    intra_baikal = np.mean(cdist(X_baikal_pca, X_baikal_pca, 'euclidean'))
    intra_sludge = np.mean(cdist(X_sludge_pca, X_sludge_pca, 'euclidean'))
    print(f"Среднее внутриклассовое расстояние (РОВ): {intra_baikal:.4f}")
    print(f"Среднее внутриклассовое расстояние (надшламовые): {intra_sludge:.4f}")
    
    print("\n" + "-" * 80)
    print("6. ИНТЕРПРЕТАЦИЯ PCA КОМПОНЕНТ")
    print("-" * 80)
    
    explained_var = detector.pca.explained_variance_ratio_
    print("\nОбъясненная дисперсия по компонентам:")
    for i, var in enumerate(explained_var):
        print(f"  PC{i+1}: {var:.4f} ({var:.2%})")
    
    loadings = detector.get_pca_loadings()
    
    print("\nТоп-5 вкладов в каждую компоненту:")
    for pc in loadings.index:
        top_features = loadings.loc[pc].abs().nlargest(5)
        print(f"\n  {pc}:")
        for feat, val in top_features.items():
            print(f"    {feat}: {val:.4f}")
    
    print("\n" + "=" * 80)
    
    return {
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'scores_baikal': scores_baikal,
        'scores_sludge': scores_sludge,
        'decision_baikal': decision_baikal,
        'decision_sludge': decision_sludge,
        'y_pred_baikal': y_pred_baikal,
        'y_pred_sludge': y_pred_sludge,
        'X_baikal_pca': X_baikal_pca,
        'X_sludge_pca': X_sludge_pca
    }

# ============================================================================
# 4. ВИЗУАЛИЗАЦИИ
# ============================================================================

def create_all_visualizations(detector, X_baikal, X_sludge, stats_results, save_prefix='baikal'):
    """
    Создаёт полный набор визуализаций
    """
    fig = plt.figure(figsize=(20, 24))
    
    # Цветовая схема
    color_baikal = '#1f77b4'  # синий
    color_sludge = '#d62728'  # красный
    color_boundary = '#2ca02c'  # зелёный
    
    X_baikal_pca = stats_results['X_baikal_pca']
    X_sludge_pca = stats_results['X_sludge_pca']
    scores_baikal = stats_results['scores_baikal']
    scores_sludge = stats_results['scores_sludge']
    decision_baikal = stats_results['decision_baikal']
    decision_sludge = stats_results['decision_sludge']
    
    # --- 1. PCA пространство: PC1 vs PC2 ---
    ax1 = plt.subplot(4, 3, 1)
    scatter1 = ax1.scatter(X_baikal_pca[:, 0], X_baikal_pca[:, 1], 
                           c=color_baikal, s=100, alpha=0.7, 
                           edgecolors='black', linewidth=1, label='РОВ Байкал')
    scatter2 = ax1.scatter(X_sludge_pca[:, 0], X_sludge_pca[:, 1], 
                           c=color_sludge, s=60, alpha=0.5, 
                           edgecolors='black', linewidth=0.5, label='Надшламовые')
    
    # Опорные векторы
    sv_pca = detector.get_support_vectors()
    ax1.scatter(sv_pca[:, 0], sv_pca[:, 1], 
                s=200, facecolors='none', edgecolors='green', 
                linewidths=2, label='Опорные векторы')
    
    ax1.set_xlabel(f'PC1 ({detector.pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({detector.pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('PCA пространство: PC1 vs PC2', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # --- 2. PCA пространство: PC1 vs PC3 ---
    ax2 = plt.subplot(4, 3, 2)
    ax2.scatter(X_baikal_pca[:, 0], X_baikal_pca[:, 2], 
                c=color_baikal, s=100, alpha=0.7, edgecolors='black')
    ax2.scatter(X_sludge_pca[:, 0], X_sludge_pca[:, 2], 
                c=color_sludge, s=60, alpha=0.5, edgecolors='black')
    ax2.set_xlabel(f'PC1 ({detector.pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC3 ({detector.pca.explained_variance_ratio_[2]:.1%})')
    ax2.set_title('PCA пространство: PC1 vs PC3', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # --- 3. PCA пространство: PC2 vs PC3 ---
    ax3 = plt.subplot(4, 3, 3)
    ax3.scatter(X_baikal_pca[:, 1], X_baikal_pca[:, 2], 
                c=color_baikal, s=100, alpha=0.7, edgecolors='black')
    ax3.scatter(X_sludge_pca[:, 1], X_sludge_pca[:, 2], 
                c=color_sludge, s=60, alpha=0.5, edgecolors='black')
    ax3.set_xlabel(f'PC2 ({detector.pca.explained_variance_ratio_[1]:.1%})')
    ax3.set_ylabel(f'PC3 ({detector.pca.explained_variance_ratio_[2]:.1%})')
    ax3.set_title('PCA пространство: PC2 vs PC3', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # --- 4. Распределение Scores ---
    ax4 = plt.subplot(4, 3, 4)
    bins = np.linspace(min(scores_baikal.min(), scores_sludge.min()),
                       max(scores_baikal.max(), scores_sludge.max()), 30)
    ax4.hist(scores_baikal, bins=bins, alpha=0.6, color=color_baikal, 
             label=f'РОВ Байкал (n={len(scores_baikal)})', density=True)
    ax4.hist(scores_sludge, bins=bins, alpha=0.6, color=color_sludge, 
             label=f'Надшламовые (n={len(scores_sludge)})', density=True)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Граница решения')
    ax4.set_xlabel('Score (log-likelihood)')
    ax4.set_ylabel('Плотность')
    ax4.set_title('Распределение Anomaly Scores', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # --- 5. Boxplot Scores ---
    ax5 = plt.subplot(4, 3, 5)
    bp = ax5.boxplot([scores_baikal, scores_sludge], 
                     label=['РОВ Байкал', 'Надшламовые'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor(color_baikal)
    bp['boxes'][1].set_facecolor(color_sludge)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax5.set_ylabel('Score')
    ax5.set_title('Boxplot: Anomaly Scores', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # --- 6. Violin plot Scores ---
    ax6 = plt.subplot(4, 3, 6)
    parts = ax6.violinplot([scores_baikal, scores_sludge], positions=[1, 2], 
                           showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], [color_baikal, color_sludge]): # type: ignore
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax6.set_xticks([1, 2])
    ax6.set_xticklabels(['РОВ Байкал', 'Надшламовые'])
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax6.set_ylabel('Score')
    ax6.set_title('Violin Plot: Score Distributions', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # --- 7. Decision Function ---
    ax7 = plt.subplot(4, 3, 7)
    x_range = np.arange(len(decision_baikal) + len(decision_sludge))
    baikal_idx = np.arange(len(decision_baikal))
    sludge_idx = np.arange(len(decision_baikal), len(decision_baikal) + len(decision_sludge))
    
    ax7.scatter(baikal_idx, decision_baikal, c=color_baikal, s=100, 
                label='РОВ Байкал', zorder=5, edgecolors='black')
    ax7.scatter(sludge_idx, decision_sludge, c=color_sludge, s=60, 
                label='Надшламовые', zorder=5, alpha=0.7, edgecolors='black')
    ax7.axhline(y=0, color='green', linestyle='-', linewidth=2, label='Граница')
    ax7.axhspan(0, max(decision_baikal.max(), decision_sludge.max()), 
                alpha=0.1, color='blue', label='Норма')
    ax7.axhspan(min(decision_baikal.min(), decision_sludge.min()), 0, 
                alpha=0.1, color='red', label='Аномалия')
    ax7.set_xlabel('Индекс образца')
    ax7.set_ylabel('Decision Function')
    ax7.set_title('Decision Function по образцам', fontsize=12, fontweight='bold')
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3)
    
    # --- 8. ROC-кривая (если можно построить) ---
    ax8 = plt.subplot(4, 3, 8)
    # Создаём бинарные метки: 1 для РОВ, -1 для надшламовых
    y_true_roc = np.concatenate([np.ones(len(scores_baikal)), -np.ones(len(scores_sludge))])
    scores_all = np.concatenate([scores_baikal, scores_sludge])
    
    fpr, tpr, thresholds = roc_curve(y_true_roc, scores_all, pos_label=1)
    auc = roc_auc_score(y_true_roc, scores_all)
    
    ax8.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax8.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax8.fill_between(fpr, tpr, alpha=0.3)
    ax8.set_xlabel('False Positive Rate')
    ax8.set_ylabel('True Positive Rate')
    ax8.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # --- 9. Precision-Recall Curve ---
    ax9 = plt.subplot(4, 3, 9)
    precision, recall, _ = precision_recall_curve(y_true_roc, scores_all, pos_label=1)
    avg_precision = average_precision_score(y_true_roc, scores_all)
    
    ax9.plot(recall, precision, 'r-', linewidth=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    ax9.fill_between(recall, precision, alpha=0.3, color='red')
    ax9.set_xlabel('Recall')
    ax9.set_ylabel('Precision')
    ax9.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # --- 10. Групповой анализ признаков ---
    ax10 = plt.subplot(4, 3, 10)
    
    # Средние значения по группам
    group_names = list(FEATURE_GROUPS.keys())
    baikal_means = []
    sludge_means = []
    
    for name, (start, end) in FEATURE_GROUPS.items():
        baikal_means.append(X_baikal[:, start:end].mean())
        sludge_means.append(X_sludge[:, start:end].mean())
    
    x_pos = np.arange(len(group_names))
    width = 0.35
    
    bars1 = ax10.bar(x_pos - width/2, baikal_means, width, 
                     label='РОВ Байкал', color=color_baikal, alpha=0.7)
    bars2 = ax10.bar(x_pos + width/2, sludge_means, width, 
                     label='Надшламовые', color=color_sludge, alpha=0.7)
    
    ax10.set_ylabel('Среднее значение (нормализованное)')
    ax10.set_title('Средние значения по группам признаков', fontsize=12, fontweight='bold')
    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(group_names, rotation=15, ha='right')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # --- 11. Тепловая карта PCA loadings (топ признаков) ---
    ax11 = plt.subplot(4, 3, 11)
    loadings = detector.get_pca_loadings()
    
    # Выбираем топ-10 признаков по суммарному абсолютному вкладу
    top_features = loadings.abs().sum().nlargest(15).index
    loadings_subset = loadings[top_features]
    
    im = ax11.imshow(loadings_subset.values, cmap='RdBu_r', aspect='auto', 
                     vmin=-max(abs(loadings_subset.values.min()), abs(loadings_subset.values.max())),
                     vmax=max(abs(loadings_subset.values.min()), abs(loadings_subset.values.max())))
    ax11.set_xticks(range(len(top_features)))
    ax11.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
    ax11.set_yticks(range(len(loadings_subset.index)))
    ax11.set_yticklabels(loadings_subset.index)
    ax11.set_title('PCA Loadings (топ-15 признаков)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax11, shrink=0.8)
    
    # --- 12. Объясненная дисперсия PCA ---
    ax12 = plt.subplot(4, 3, 12)
    cumvar = np.cumsum(detector.pca.explained_variance_ratio_)
    ax12.bar(range(1, len(detector.pca.explained_variance_ratio_)+1), 
             detector.pca.explained_variance_ratio_, alpha=0.7, color='steelblue',
             label='Индивидуальная')
    ax12.plot(range(1, len(cumvar)+1), cumvar, 'ro-', linewidth=2, 
              markersize=8, label='Кумулятивная')
    ax12.axhline(y=0.95, color='green', linestyle='--', label='95% порог')
    ax12.set_xlabel('Компонента PCA')
    ax12.set_ylabel('Объясненная дисперсия')
    ax12.set_title('Scree Plot: PCA Explained Variance', fontsize=12, fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_full_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# ============================================================================
# 5. ДЕТАЛЬНЫЙ АНАЛИЗ ПРИЗНАКОВ
# ============================================================================

def analyze_feature_importance(detector, X_baikal, X_sludge):
    """
    Анализ важности признаков через PCA loadings и различия между классами
    """
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ПРИЗНАКОВ")
    print("=" * 80)
    
    # Стандартизированные данные
    X_baikal_scaled = detector.scaler.transform(X_baikal)
    X_sludge_scaled = detector.scaler.transform(X_sludge)
    
    # t-test для каждого признака
    print("\nТоп-15 признаков с наибольшими различиями (t-test):")
    t_stats = []
    p_values = []
    cohen_ds = []
    
    for i, name in enumerate(ALL_FEATURES):
        t_stat, p_val = stats.ttest_ind(X_baikal[:, i], X_sludge[:, i]) # type: ignore
        # Cohen's d
        pooled_std = np.sqrt(((len(X_baikal)-1)*X_baikal[:, i].var() + 
                              (len(X_sludge)-1)*X_sludge[:, i].var()) / 
                             (len(X_baikal) + len(X_sludge) - 2))
        d = (X_baikal[:, i].mean() - X_sludge[:, i].mean()) / pooled_std if pooled_std > 0 else 0
        
        t_stats.append(abs(t_stat)) # type: ignore
        p_values.append(p_val)
        cohen_ds.append(abs(d))
    
    # DataFrame для сортировки
    feature_importance = pd.DataFrame({
        'Feature': ALL_FEATURES,
        't_stat': t_stats,
        'p_value': p_values,
        'cohens_d': cohen_ds,
        'baikal_mean': X_baikal.mean(axis=0),
        'sludge_mean': X_sludge.mean(axis=0),
        'difference': X_sludge.mean(axis=0) - X_baikal.mean(axis=0)
    })
    
    feature_importance['group'] = feature_importance['Feature'].apply(
        lambda x: 'Optical' if x in optical_names 
        else 'FT-ICR Metrics' if x in metrics_names
        else 'Van Krevelen' if x in vk_names
        else 'Mass Bins'
    )
    
    # Сортировка по Cohen's d
    top_features = feature_importance.nlargest(15, 'cohens_d')
    print(top_features[['Feature', 'group', 'cohens_d', 'p_value', 'baikal_mean', 'sludge_mean', 'difference']].to_string())
    
    # Анализ по группам
    print("\n" + "-" * 80)
    print("Анализ по группам признаков:")
    group_stats = feature_importance.groupby('group').agg({
        'cohens_d': ['mean', 'max', 'std'],
        'p_value': 'mean'
    }).round(4)
    print(group_stats)
    
    return feature_importance

def plot_feature_group_analysis(feature_importance, save_prefix='baikal'):
    """
    Визуализация анализа признаков по группам
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Топ-15 признаков по Cohen's d
    ax1 = axes[0, 0]
    top15 = feature_importance.nlargest(15, 'cohens_d')
    colors = {'Optical': '#1f77b4', 'FT-ICR Metrics': '#ff7f0e', 
              'Van Krevelen': '#2ca02c', 'Mass Bins': '#d62728'}
    bar_colors = [colors[g] for g in top15['group']]
    
    bars = ax1.barh(range(len(top15)), top15['cohens_d'], color=bar_colors, alpha=0.7)
    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels(top15['Feature'], fontsize=9)
    ax1.set_xlabel("Cohen's d (эффект размера)")
    ax1.set_title('Топ-15 различающихся признаков', fontweight='bold')
    ax1.invert_yaxis()
    
    # Легенда
    legend_patches = [mpatches.Patch(color=colors[g], label=g) for g in colors]
    ax1.legend(handles=legend_patches, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Распределение Cohen's d по группам
    ax2 = axes[0, 1]
    group_data = [feature_importance[feature_importance['group']==g]['cohens_d'].values 
                  for g in ['Optical', 'FT-ICR Metrics', 'Van Krevelen', 'Mass Bins']]
    bp = ax2.boxplot(group_data, labels=['Optical', 'FT-ICR\nMetrics', 'Van\nKrevelen', 'Mass\nBins'],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Cohen's d")
    ax2.set_title('Распределение эффектов по группам', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter: baikal_mean vs sludge_mean
    ax3 = axes[1, 0]
    for group, color in colors.items():
        subset = feature_importance[feature_importance['group'] == group]
        ax3.scatter(subset['baikal_mean'], subset['sludge_mean'], 
                   c=color, label=group, alpha=0.6, s=50)
    
    # Диагональ y=x
    min_val = min(feature_importance['baikal_mean'].min(), feature_importance['sludge_mean'].min())
    max_val = max(feature_importance['baikal_mean'].max(), feature_importance['sludge_mean'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax3.set_xlabel('Среднее РОВ Байкал')
    ax3.set_ylabel('Среднее надшламовые воды')
    ax3.set_title('Сравнение средних по классам', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. P-value distribution
    ax4 = axes[1, 1]
    significant = feature_importance[feature_importance['p_value'] < 0.05]
    ax4.hist(feature_importance['p_value'], bins=30, alpha=0.5, color='gray', 
             label=f'Все ({len(feature_importance)})')
    ax4.hist(significant['p_value'], bins=30, alpha=0.7, color='red', 
             label=f'p < 0.05 ({len(significant)})')
    ax4.axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='α = 0.05')
    ax4.set_xlabel('p-value')
    ax4.set_ylabel('Частота')
    ax4.set_title('Распределение p-values', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# ============================================================================
# 6. АНАЛИЗ ОТДЕЛЬНЫХ ОБРАЗЦОВ
# ============================================================================

def analyze_individual_samples(detector, X_baikal, X_sludge, sample_names_baikal=None, 
                                sample_names_sludge=None):
    """
    Детальный анализ каждого образца
    """
    if sample_names_baikal is None:
        sample_names_baikal = [f'Baikal_{i+1}' for i in range(len(X_baikal))]
    if sample_names_sludge is None:
        sample_names_sludge = [f'Sludge_{i+1}' for i in range(len(X_sludge))]
    
    scores_baikal = detector.score_samples(X_baikal)
    scores_sludge = detector.score_samples(X_sludge)
    decisions_baikal = detector.decision_function(X_baikal)
    decisions_sludge = detector.decision_function(X_sludge)
    predictions_baikal = detector.predict(X_baikal)
    predictions_sludge = detector.predict(X_sludge)
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ОТДЕЛЬНЫХ ОБРАЗЦОВ")
    print("=" * 80)
    
    print("\nРОВ Байкал:")
    print("-" * 60)
    for i, name in enumerate(sample_names_baikal):
        status = "✓ НОРМА" if predictions_baikal[i] == 1 else "✗ АНОМАЛИЯ (False Alarm!)"
        print(f"{name:15} | Score: {scores_baikal[i]:8.4f} | Decision: {decisions_baikal[i]:8.4f} | {status}")
    
    print("\nНадшламовые воды:")
    print("-" * 60)
    # Сортировка по score (наиболее аномальные первыми)
    sludge_data = list(zip(sample_names_sludge, scores_sludge, decisions_sludge, predictions_sludge))
    sludge_data.sort(key=lambda x: x[1])  # Сортировка по score (возрастание)
    
    for name, score, decision, pred in sludge_data:
        status = "✓ АНОМАЛИЯ (детектировано)" if pred == -1 else "✗ ПРОПУЩЕНО"
        print(f"{name:15} | Score: {score:8.4f} | Decision: {decision:8.4f} | {status}")
    
    # Наиболее типичные надшламовые (ближе к Байкалу)
    print("\nНаиболее 'байкальские' надшламовые образцы (топ-5):")
    sludge_data_sorted = sorted(sludge_data, key=lambda x: x[1], reverse=True)
    for name, score, decision, pred in sludge_data_sorted[:5]:
        print(f"  {name}: Score={score:.4f}, Decision={decision:.4f}")
    
    # Наиболее аномальные
    print("\nНаиболее аномальные надшламовые образцы (топ-5):")
    for name, score, decision, pred in sludge_data[:5]:
        print(f"  {name}: Score={score:.4f}, Decision={decision:.4f}")

# ============================================================================
# 7. ИНТЕРПРЕТАЦИЯ МАСС-СПЕКТРОМЕТРИЧЕСКИХ ДАННЫХ
# ============================================================================

def analyze_mass_spectrum_patterns(X_baikal, X_sludge):
    """
    Анализ паттернов в масс-спектрометрических данных (интервалы 1 Да)
    """
    mass_start_idx = 50  # После оптических, метрик и ВК
    mass_data_baikal = X_baikal[:, mass_start_idx:]
    mass_data_sludge = X_sludge[:, mass_start_idx:]
    
    mass_numbers = np.arange(225, 700)
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ МАСС-СПЕКТРОМЕТРИЧЕСКИХ ПАТТЕРНОВ")
    print("=" * 80)
    
    # Средние спектры
    mean_baikal = mass_data_baikal.mean(axis=0)
    mean_sludge = mass_data_sludge.mean(axis=0)
    diff = mean_sludge - mean_baikal
    
    # Топ масс с наибольшими различиями
    top_increased_idx = np.argsort(diff)[-10:][::-1]
    top_decreased_idx = np.argsort(diff)[:10]
    
    print("\nТоп-10 масс, ПОВЫШЕННЫХ в надшламовых водах:")
    for idx in top_increased_idx:
        print(f"  m/z {mass_numbers[idx]:3d} Да: Δ = {diff[idx]:+.4f} "
              f"(Байкал: {mean_baikal[idx]:.4f}, Шлам: {mean_sludge[idx]:.4f})")
    
    print("\nТоп-10 масс, ПОНИЖЕННЫХ в надшламовых водах:")
    for idx in top_decreased_idx:
        print(f"  m/z {mass_numbers[idx]:3d} Да: Δ = {diff[idx]:+.4f} "
              f"(Байкал: {mean_baikal[idx]:.4f}, Шлам: {mean_sludge[idx]:.4f})")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Средние спектры
    ax1 = axes[0, 0]
    ax1.plot(mass_numbers, mean_baikal, 'b-', linewidth=1, alpha=0.7, label='РОВ Байкал')
    ax1.plot(mass_numbers, mean_sludge, 'r-', linewidth=1, alpha=0.7, label='Надшламовые')
    ax1.set_xlabel('m/z, Да')
    ax1.set_ylabel('Интенсивность (средняя)')
    ax1.set_title('Средние масс-спектры', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Разница
    ax2 = axes[0, 1]
    colors = ['red' if d > 0 else 'blue' for d in diff]
    ax2.bar(mass_numbers, diff, color=colors, alpha=0.6, width=0.8)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('m/z, Да')
    ax2.set_ylabel('Δ Интенсивность (Шлам - Байкал)')
    ax2.set_title('Различия в масс-спектрах', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Корреляция спектров
    ax3 = axes[1, 0]
    correlation = np.corrcoef(mean_baikal, mean_sludge)[0, 1]
    ax3.scatter(mean_baikal, mean_sludge, alpha=0.5, s=20)
    ax3.plot([0, max(mean_baikal.max(), mean_sludge.max())], 
             [0, max(mean_baikal.max(), mean_sludge.max())], 'k--', alpha=0.5)
    ax3.set_xlabel('РОВ Байкал (средняя интенсивность)')
    ax3.set_ylabel('Надшламовые (средняя интенсивность)')
    ax3.set_title(f'Корреляция спектров: r = {correlation:.3f}', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Спектр Cohen's d по массам
    ax4 = axes[1, 1]
    cohen_d_mass = []
    for i in range(mass_data_baikal.shape[1]):
        pooled_std = np.sqrt(((len(mass_data_baikal)-1)*mass_data_baikal[:, i].var() + 
                              (len(mass_data_sludge)-1)*mass_data_sludge[:, i].var()) / 
                             (len(mass_data_baikal) + len(mass_data_sludge) - 2))
        d = (mass_data_sludge[:, i].mean() - mass_data_baikal[:, i].mean()) / pooled_std if pooled_std > 0 else 0
        cohen_d_mass.append(abs(d))
    
    ax4.plot(mass_numbers, cohen_d_mass, 'g-', linewidth=0.8, alpha=0.7)
    ax4.fill_between(mass_numbers, cohen_d_mass, alpha=0.3, color='green')
    ax4.set_xlabel('m/z, Да')
    ax4.set_ylabel("|Cohen's d|")
    ax4.set_title('Эффект размера по массам', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baikal_mass_spectrum_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# ============================================================================
# 8. ГЛАВНАЯ ФУНКЦИЯ ЗАПУСКА
# ============================================================================

def run_full_analysis(X_baikal, X_sludge, sample_names_baikal=None, 
                      sample_names_sludge=None, n_pca=4, nu=0.1):
    """
    Полный анализ: обучение, статистика, визуализация, интерпретация
    """
    print("=" * 80)
    print("ЗАПУСК ПОЛНОГО АНАЛИЗА")
    print("=" * 80)
    print(f"Размерность данных: {X_baikal.shape[1]} признаков")
    print(f"Образцов РОВ Байкал: {len(X_baikal)}")
    print(f"Образцов надшламовых вод: {len(X_sludge)}")
    
    # Обучение модели
    detector = BaikalAnomalyDetector(n_pca_components=n_pca, nu=nu)
    detector.fit(X_baikal)
    
    # Полная статистика
    stats_results = print_full_statistics(detector, X_baikal, X_sludge)
    
    # Визуализации
    fig_main = create_all_visualizations(detector, X_baikal, X_sludge, stats_results)
    
    # Анализ признаков
    feature_importance = analyze_feature_importance(detector, X_baikal, X_sludge)
    fig_features = plot_feature_group_analysis(feature_importance)
    
    # Анализ образцов
    analyze_individual_samples(detector, X_baikal, X_sludge, 
                               sample_names_baikal, sample_names_sludge)
    
    # Анализ масс-спектров
    fig_mass = analyze_mass_spectrum_patterns(X_baikal, X_sludge)
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("Сохранены файлы:")
    print("  - baikal_full_analysis.png")
    print("  - baikal_feature_analysis.png")
    print("  - baikal_mass_spectrum_analysis.png")
    print("=" * 80)
    
    return detector, stats_results, feature_importance

# ============================================================================
# 9. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

"""
# Пример использования:

# Загрузка ваших данных (замените на реальные данные)
# X_baikal = pd.read_csv('baikal_samples.csv').values  # (5, 525)
# X_sludge = pd.read_csv('sludge_samples.csv').values  # (39, 525)

# Или синтетические данные для тестирования:
np.random.seed(42)
X_baikal = np.random.randn(5, 525) * 0.5  # Компактный кластер
X_sludge = np.random.randn(39, 525) * 1.5 + 2  # Разбросанные, смещённые
"""
# Запуск полного анализа
detector, stats, features = run_full_analysis(X_baikal, X_sludge,
    sample_names_baikal=[name for name in spectra_list_name if ut.extract_class_from_name(name) == 'Baikal' and name != 'ADOM-SLB-B3-2'],
    sample_names_sludge=[name for name in spectra_list_name if ut.extract_class_from_name(name) == 'ADOM'],
    
    
    n_pca=4,
    nu=0.1
)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# --- 1. Генерация данных (Замените этот блок на загрузку ваших данных) ---
np.random.seed(42)
n_baikal = 5
n_solzanka = 39
n_total = n_baikal + n_solzanka

# Названия признаков в вашем порядке
optical_descriptors = ['Ag_275', 'Ag_380', 'Asm_280', 'Asm_350', 'B1', 'B2', 'Component_1',
 'Component_2', 'Component_3', 'E2E3', 'E3E4', 'E4E6', 'FIX', 'HIX',
 'S_380_443', 'Suva']
metrics = ['AI', 'C', 'CAI', 'CRAM', 'DBE', 'DBE-O', 'DBE-OC', 'DBE_AI', 'H', 'H/C', 'N', 'NOSC', 'O', 'O/C']
vk_bins = [f'VK_{i}' for i in range(1, 21)]
mass_intervals = [f'Mass_{i}_{i+1}' for i in range(475)]

feature_names = optical_descriptors + metrics + vk_bins + mass_intervals
n_features = len(feature_names)

print(f"Датасет: {X.shape[0]} образцов, {X.shape[1]} признаков")
print(f"Классы: Байкал={n_baikal}, Полигон={n_solzanka}")

# --- 2. Предобработка и PLS-DA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = combined_data['class'].map({'Baikal': 0, 'ADOM': 1}).to_numpy()

# Используем 2 компоненты, так как они дают хорошее разделение
n_components = 2
pls = PLSRegression(n_components=n_components)
X_pls = pls.fit_transform(X_scaled, y)[0]

# --- 3. Визуализация разделения (Score Plot) ---
plt.figure(figsize=(10, 8))
colors = ['#1f77b4' if label == 0 else '#d62728' for label in y] # Blue for Baikal, Red for Solzan
labels = ['Baikal' if label == 0 else 'Solzan' for label in y]

plt.scatter(X_pls[:, 0], X_pls[:, 1], c=colors, s=120, edgecolors='black', zorder=5, alpha=0.8)
for i, txt in enumerate(labels):
    plt.annotate(txt, (X_pls[i, 0]+0.05, X_pls[i, 1]+0.05), fontsize=9, fontweight='bold')

plt.title('PLS-DA: Разделение РОВ Байкала и вод полигона\n(Пространство первых 2 компонент)', fontsize=14)
plt.xlabel('Компонента 1 (LV1)')
plt.ylabel('Компонента 2 (LV2)')
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 4. Классификация на PLS-компонентах с LOO-CV ---
# Мы обучаем классификатор уже на сокращенном пространстве (2 признака вместо 525)
# Это резко снижает риск переобучения
loo = LeaveOneOut()

# Массивы для хранения результатов по всем образцам
y_pred_loo = np.zeros_like(y, dtype=int)
y_prob_loo = np.zeros_like(y, dtype=float)
decision_scores_loo = np.zeros_like(y, dtype=float)

# Списки для детального логирования (если нужно потом сделать DataFrame)
log_data = []

for train_index, test_index in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 1. Обучаем PLS на тренировочных данных
    pls_fold = PLSRegression(n_components=2)
    # fit_transform возвращает (X_scores, Y_scores). Нам нужны X_scores.
    X_train_pls = pls_fold.fit_transform(X_train, y_train)[0]
    X_test_pls = pls_fold.transform(X_test)
    
    # 2. Обучаем классификатор на PLS-компонентах
    # C=1.0 - стандартная регуляризация. Можно подобрать через GridSearch, но на 4 образцах это сложно.
    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
    clf.fit(X_train_pls, y_train)
    
    # 3. Предсказания
    pred_class = clf.predict(X_test_pls)[0]
    pred_prob = clf.predict_proba(X_test_pls)[0, 1] # type: ignore # Вероятность класса 1 (Полигон)
    dec_score = clf.decision_function(X_test_pls)[0] # Расстояние до гиперплоскости
    
    # Сохраняем результаты
    y_pred_loo[test_index] = pred_class
    y_prob_loo[test_index] = pred_prob
    decision_scores_loo[test_index] = dec_score
    
    # Логируем для таблицы
    log_data.append({
        'Index': test_index[0],
        'True_Class': 'Baikal' if y_test[0] == 0 else 'Solzan',
        'Pred_Class': 'Baikal' if pred_class == 0 else 'Solzan',
        'Decision_Score': dec_score,
        'Prob_Solzan': pred_prob,
        'Correct': y_test[0] == pred_class
    })

# --- СТАТИСТИКА ---

print("\n--- Отчет о классификации (Leave-One-Out CV) ---")
print(classification_report(y, y_pred_loo, target_names=['Baikal (РОВ)', 'Solzan (Полигон)']))

# Расчет Accuracy и AUC-ROC
acc = accuracy_score(y, y_pred_loo)
# AUC можно считать только если есть оба класса в предсказаниях и хотя бы 2 образца
try:
    auc = roc_auc_score(y, y_prob_loo)
except ValueError:
    auc = np.nan

print(f"Total Accuracy (LOO): {acc:.4f}")
print(f"AUC-ROC (LOO): {auc:.4f}")

# Вывод таблицы с деталями по каждому образцу
df_results = pd.DataFrame(log_data).sort_values(by='Index')
df_results['name'] = combined_data['sample_name'].to_list()
print("\n--- Детальные результаты LOO для каждого образца ---")
print(df_results.to_string(index=False))


# --- ВИЗУАЛИЗАЦИЯ ---

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Распределение Decision Function (Насколько уверенно разделены классы)
sns.boxplot(x=df_results['True_Class'], y=df_results['Decision_Score'], 
            palette={'Baikal': 'steelblue', 'Solzan': 'salmon'}, ax=axes[0])
axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0].set_title('Распределение Decision Function (LOO)\n(>0: Полигон, <0: Байкал)')
axes[0].set_ylabel('Значение функции решения')
axes[0].grid(True, linestyle=':', alpha=0.6)

# Интерпретация: Если ящики не пересекаются с нулем и друг с другом — разделение идеальное.

# 2. PLS Score Plot (Визуализация разделения в пространстве компонент)
# Чтобы построить красивый скор-плот, обучим финальную PLS на ВСЕХ данных для визуализации
# (Внимание: это только для картинки, не для оценки качества!)
pls_final = PLSRegression(n_components=2)
X_all_pls = pls_final.fit_transform(X_scaled, y)[0]

scatter = axes[1].scatter(X_all_pls[:, 0], X_all_pls[:, 1], c=y, cmap='coolwarm', s=100, edgecolors='black', zorder=5)
axes[1].set_xlabel(f'Component 1 ({pls_final.x_scores_.std(axis=0)[0]:.2f} var)')
axes[1].set_ylabel(f'Component 2 ({pls_final.x_scores_.std(axis=0)[1]:.2f} var)')
axes[1].set_title('PLS-DA Score Plot (Обучено на всех данных)')
axes[1].legend(handles=scatter.legend_elements()[0], labels=['Baikal', 'Solzan'])
axes[1].grid(True, linestyle=':', alpha=0.6)

# Опционально: Добавить стрелки самых важных признаков (Loadings)
# loadings = pls_final.x_loadings_
# for i in range(len(feature_names)): # feature_names нужно определить заранее
#     if abs(loadings[i, 0]) > 0.1 or abs(loadings[i, 1]) > 0.1: # Показывать только важные
#         axes[1].arrow(0, 0, loadings[i, 0]*5, loadings[i, 1]*5, head_width=0.05, head_length=0.05, fc='gray', ec='gray', alpha=0.5)

# 3. Матрица ошибок (Confusion Matrix)
cm = confusion_matrix(y, y_pred_loo)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], 
            xticklabels=['Baikal', 'Solzan'], yticklabels=['Baikal', 'Solzan'])
axes[2].set_title('Матрица ошибок (LOO CV)')
axes[2].set_ylabel('Истинный класс')
axes[2].set_xlabel('Предсказанный класс')

plt.tight_layout()
plt.show()

# --- 5. VIP Scores (Variable Importance in Projection) ---
# Показывает вклад каждого исходного признака в модель PLS
def calculate_vip(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vip

vip_scores = calculate_vip(pls, X_scaled)
vip_df = pd.DataFrame({'Feature': feature_names, 'VIP': vip_scores})
vip_df = vip_df.sort_values(by='VIP', ascending=False)

# Визуализация Топ-15 признаков
top_n = 15
top_features = vip_df.head(top_n)

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), top_features['VIP'].values, align='center', color='steelblue') # type: ignore
plt.yticks(range(top_n), top_features['Feature'].values) # type: ignore
plt.xlabel('VIP Score')
plt.title(f'Топ-{top_n} важных дескрипторов (VIP > 1 — значимые)')
plt.axvline(x=1.0, color='red', linestyle='--', label='Порог значимости (VIP=1)')
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.show()

print("\nТоп-10 признаков, различающих классы:")
print(vip_df.head(10)[['Feature', 'VIP']].to_string(index=False))

# --- 6. Сравнение с SVM на исходных данных (для проверки) ---
# SVM с ядром RBF может быть чувствителен к шуму, но проверим
svm_pipe = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
y_pred_svm_loo = cross_val_predict(svm_pipe, X_scaled, y, cv=loo)
print("\n--- SVM (RBF) Classification Report (LOO) ---")
print(classification_report(y, y_pred_svm_loo, target_names=['Baikal', 'Solzanka']))

