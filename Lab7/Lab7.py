import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# 1. Импорт датасета для кластеризации
print("=" * 50)
print("1. ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
print("=" * 50)

# Создадим синтетический датасет для демонстрации (в реальном случае загрузите свой файл)
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, n_features=5, 
                       random_state=42, cluster_std=1.5)

# Преобразуем в DataFrame
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)

print("Первые 5 строк датасета:")
print(df.head())
print(f"\nРазмер датасета: {df.shape}")
print(f"\nИнформация о датасете:")
print(df.info())
print(f"\nСтатистика датасета:")
print(df.describe())

# 2. Очистка данных и предобработка
print("\n" + "=" * 50)
print("2. ОЧИСТКА И ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 50)

# Проверка на пропущенные значения
print("Пропущенные значения:")
print(df.isnull().sum())

# Добавим немного пропущенных значений для демонстрации
np.random.seed(42)
mask = np.random.random(df.shape) < 0.03
df_with_missing = df.mask(mask)

print("\nПропущенные значения после добавления:")
print(df_with_missing.isnull().sum())

# Заполнение пропущенных значений
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_clean = pd.DataFrame(imputer.fit_transform(df_with_missing), 
                       columns=df.columns)

print("\nПропущенные значения после обработки:")
print(df_clean.isnull().sum())

# Обнаружение выбросов с помощью IQR
def detect_outliers_iqr(df):
    outliers_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    return outliers_mask

outliers_mask = detect_outliers_iqr(df_clean)
print(f"\nКоличество выбросов по признакам:")
print(outliers_mask.sum())

# Визуализация выбросов
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
df_clean.boxplot()
plt.title('Ящики с усами до обработки выбросов')
plt.xticks(rotation=45)

# Обработка выбросов - winsorization
def winsorize_data(df, limits=(0.05, 0.05)):
    df_winsorized = df.copy()
    for col in df.columns:
        lower_limit = df[col].quantile(limits[0])
        upper_limit = df[col].quantile(1 - limits[1])
        df_winsorized[col] = np.clip(df[col], lower_limit, upper_limit)
    return df_winsorized

df_no_outliers = winsorize_data(df_clean)

plt.subplot(1, 2, 2)
df_no_outliers.boxplot()
plt.title('Ящики с усами после обработки выбросов')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_no_outliers)
X_scaled_df = pd.DataFrame(X_scaled, columns=df_no_outliers.columns)

print("\nДанные после стандартизации (первые 5 строк):")
print(X_scaled_df.head())

# 3. Понижение размерности
print("\n" + "=" * 50)
print("3. ПОНИЖЕНИЕ РАЗМЕРНОСТИ")
print("=" * 50)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Объясненная дисперсия
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Объясненная дисперсия PCA:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"Компонента {i+1}: {var:.4f} ({cum_var:.4f})")

# Визуализация объясненной дисперсии
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.xlabel('Номер компоненты')
plt.ylabel('Объясненная дисперсия')
plt.title('Объясненная дисперсия PCA')
plt.grid(True, alpha=0.3)

# Выбираем количество компонент, объясняющих 95% дисперсии
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nКоличество компонент для 95% дисперсии: {n_components}")

# Применяем PCA с выбранным количеством компонент
pca_final = PCA(n_components=2)
X_pca_final = pca_final.fit_transform(X_scaled)

plt.subplot(1, 3, 2)
plt.scatter(X_pca_final[:, 0], X_pca_final[:, 1], alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Данные после PCA (2 компоненты)')

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.subplot(1, 3, 3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Данные после t-SNE')

plt.tight_layout()
plt.show()

# 4. Создание моделей кластеризации
print("\n" + "=" * 50)
print("4. СОЗДАНИЕ МОДЕЛЕЙ КЛАСТЕРИЗАЦИИ")
print("=" * 50)

# Будем использовать данные после PCA для визуализации
X_for_clustering = X_pca_final

# 5. Подбор оптимального количества кластеров для K-means
print("\n" + "=" * 50)
print("5. ПОДБОР ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ")
print("=" * 50)

# Метод локтя и силуэтный анализ для K-means
k_range = range(2, 11)
inertia = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Визуализация методов выбора k
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Количество кластеров')
plt.ylabel('Inertia')
plt.title('Метод локтя для K-means')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Количество кластеров')
plt.ylabel('Silhouette Score')
plt.title('Силуэтный анализ для K-means')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Выбор оптимального k
optimal_k_elbow = 4  # Можно автоматизировать поиск "локтя"
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"Оптимальное k по методу локтя: {optimal_k_elbow}")
print(f"Оптимальное k по силуэтному анализу: {optimal_k_silhouette}")

# Подбор параметров для DBSCAN
def find_optimal_eps(X, k_min=2):
    neighbors = NearestNeighbors(n_neighbors=k_min)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, k_min-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Точки данных')
    plt.ylabel('Расстояние до k-го соседа')
    plt.title('Метод колена для подбора eps в DBSCAN')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return distances

distances = find_optimal_eps(X_scaled)
# Выбираем eps на основе графика (точка наибольшей кривизны)
optimal_eps = 0.5

print(f"Выбранное eps для DBSCAN: {optimal_eps}")

# 6. Обучение различных моделей кластеризации
print("\n" + "=" * 50)
print("6. ОБУЧЕНИЕ МОДЕЛЕЙ КЛАСТЕРИЗАЦИИ")
print("=" * 50)

models = {
    'K-means (k=4)': KMeans(n_clusters=4, random_state=42, n_init=10),
    'K-means (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
    'DBSCAN': DBSCAN(eps=optimal_eps, min_samples=5),
    'Agglomerative (k=4)': AgglomerativeClustering(n_clusters=4),
    'Agglomerative (k=3)': AgglomerativeClustering(n_clusters=3)
}

results = {}

for name, model in models.items():
    if name.startswith('DBSCAN'):
        labels = model.fit_predict(X_scaled)
    else:
        labels = model.fit_predict(X_scaled)
    
    results[name] = labels
    
    # Подсчет кластеров (для DBSCAN -1 это шум)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1) if -1 in unique_labels else 0
    
    print(f"\n{name}:")
    print(f"  Количество кластеров: {n_clusters}")
    if n_noise > 0:
        print(f"  Точки шума: {n_noise}")

# 7. Оценка качества кластеризации
print("\n" + "=" * 50)
print("7. ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
print("=" * 50)

metrics_results = []

for name, labels in results.items():
    if len(np.unique(labels)) > 1:  # Нужно как минимум 2 кластера для метрик
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
        
        metrics_results.append({
            'Model': name,
            'Silhouette': silhouette,
            'Calinski-Harabasz': calinski,
            'Davies-Bouldin': davies
        })
        
        print(f"\n{name}:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski:.4f}")
        print(f"  Davies-Bouldin Score: {davies:.4f}")

# Создаем DataFrame с результатами метрик
metrics_df = pd.DataFrame(metrics_results)
print("\nСравнение моделей по метрикам:")
print(metrics_df.round(4))

# 8. Визуализация результатов кластеризации
print("\n" + "=" * 50)
print("8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 50)

# Визуализация всех методов кластеризации
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (name, labels) in enumerate(results.items()):
    scatter = axes[idx].scatter(X_for_clustering[:, 0], X_for_clustering[:, 1], 
                               c=labels, cmap='viridis', alpha=0.7)
    axes[idx].set_title(f'{name}\n(Silhouette: {silhouette_score(X_scaled, labels):.3f})')
    axes[idx].set_xlabel('PC1')
    axes[idx].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[idx])

# Дендрограмма для иерархической кластеризации
axes[5].clear()
linked = linkage(X_scaled, 'ward')
dendrogram(linked, ax=axes[5], truncate_mode='level', p=5)
axes[5].set_title('Дендрограмма\n(Иерархическая кластеризация)')
axes[5].set_xlabel('Образцы')
axes[5].set_ylabel('Расстояние')

plt.tight_layout()
plt.show()

# Дополнительная визуализация в пространстве t-SNE
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (name, labels) in enumerate(results.items()):
    scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                               c=labels, cmap='viridis', alpha=0.7)
    axes[idx].set_title(f'{name} (t-SNE)')
    axes[idx].set_xlabel('t-SNE 1')
    axes[idx].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[idx])

axes[5].axis('off')
plt.tight_layout()
plt.show()

# 9. Анализ характеристик кластеров для лучшей модели
print("\n" + "=" * 50)
print("9. АНАЛИЗ ХАРАКТЕРИСТИК КЛАСТЕРОВ")
print("=" * 50)

# Выбираем лучшую модель по силуэтному score
best_model_name = metrics_df.loc[metrics_df['Silhouette'].idxmax(), 'Model']
best_labels = results[best_model_name]

print(f"Лучшая модель: {best_model_name}")

# Добавляем метки кластеров к исходным данным
df_with_clusters = df_no_outliers.copy()
df_with_clusters['Cluster'] = best_labels

# Анализ средних значений по кластерам
cluster_means = df_with_clusters.groupby('Cluster').mean()
print("\nСредние значения признаков по кластерам:")
print(cluster_means.round(3))

# Визуализация средних значений по кластерам
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, annot=True, cmap='YlOrRd', center=0)
plt.title('Средние значения признаков по кластерам')
plt.xlabel('Кластер')
plt.ylabel('Признак')
plt.tight_layout()
plt.show()

# Размеры кластеров
cluster_sizes = df_with_clusters['Cluster'].value_counts().sort_index()
print("\nРазмеры кластеров:")
for cluster, size in cluster_sizes.items():
    print(f"Кластер {cluster}: {size} точек ({size/len(df_with_clusters)*100:.1f}%)")

print("\nКластеризация завершена успешно!")