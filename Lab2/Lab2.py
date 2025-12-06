import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. Импорт датасета
df = pd.read_csv('CarPrice_Assignment.csv')
print("1. Первые 5 строк датасета:")
print(df.head())
print(f"\nРазмер датасета: {df.shape}")
print(f"\nИнформация о данных:")
print(df.info())

# 2. Проверка на пропуски и аномальные значения
print("2. Проверка на пропущенные значения:")
print(df.isnull().sum())

# Визуализация пропущенных значений
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap пропущенных значений')
plt.show()

# Статистический анализ
print("\nСтатистическое описание числовых признаков:")
print(df.describe())

# Поиск аномальных значений с помощью boxplot
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(20, 15))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(5, 5, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()

# Анализ распределения числовых признаков
plt.figure(figsize=(20, 18))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(5, 5, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Распределение {col}')
plt.tight_layout()
plt.show()

# Заполнение пропусков (если они есть)
if df.isnull().sum().sum() > 0:
    print("Заполняем пропущенные значения...")
    
    # Для числовых признаков - медианой
    numeric_imputer = SimpleImputer(strategy='median')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # Для категориальных - наиболее частым значением
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
else:
    print("Пропущенных значений не обнаружено")

# Удаление выбросов с помощью IQR метода
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Применяем к числовым признакам (кроме car_ID)
numeric_cols_for_outliers = [col for col in numeric_cols if col != 'car_ID']
original_size = df.shape[0]

for col in numeric_cols_for_outliers:
    df = remove_outliers_iqr(df, col)

removed_count = original_size - df.shape[0]
print(f"Удалено выбросов: {removed_count}")
print(f"Размер датасета после удаления выбросов: {df.shape}")

# Проверка после обработки
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap пропущенных значений после обработки')
plt.show()

# 3. Генерация новых признаков и кодирование категориальных
# Генерация новых признаков
df['power_to_weight'] = df['horsepower'] / df['curbweight']
df['size_ratio'] = df['carlength'] * df['carwidth'] * df['carheight']
df['fuel_efficiency'] = (df['citympg'] + df['highwaympg']) / 2
df['engine_power_density'] = df['horsepower'] / df['enginesize']

print("Новые признаки:")
print(df[['power_to_weight', 'size_ratio', 'fuel_efficiency', 'engine_power_density']].head())

# Кодирование категориальных признаков
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Категориальные признаки: {list(categorical_cols)}")

# One-Hot Encoding для бинарных и мультиклассовых признаков с малым количеством категорий
ohe_cols = ['fueltype', 'aspiration', 'doornumber', 'drivewheel', 'enginelocation']
df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols, drop_first=True)

# Label Encoding для остальных категориальных признаков
label_encoder = LabelEncoder()
remaining_cat_cols = [col for col in categorical_cols if col not in ohe_cols and col != 'CarName']

for col in remaining_cat_cols:
    if col in df.columns:  # Проверяем, что колонка еще существует
        df[col] = label_encoder.fit_transform(df[col])

# Обработка CarName - извлечение бренда
df['brand'] = df['CarName'].apply(lambda x: x.split()[0])
df['is_premium_brand'] = df['brand'].apply(lambda x: 1 if x in ['bmw', 'mercedes', 'audi', 'porsche', 'jaguar'] else 0)
df = df.drop(['CarName', 'brand'], axis=1)

print(f"\nРазмер датасета после кодирования: {df.shape}")
print(f"\nНовые названия колонок: {df.columns.tolist()}")

# 4. Стандартизация и нормализация данных
# Выделяем числовые признаки для стандартизации
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Исключаем бинарные и dummy переменные из стандартизации
cols_to_exclude = [col for col in df.columns if any(x in col for x in ['fueltype_', 'aspiration_', 'doornumber_', 'drivewheel_', 'enginelocation_', 'is_premium_brand'])]
cols_to_standardize = [col for col in numeric_cols if col not in cols_to_exclude]

print("Признаки для стандартизации:", cols_to_standardize)

# Стандартизация
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

# Нормализация (min-max scaling)
df_normalized = df.copy()
for col in cols_to_standardize:
    df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)  # +1e-8 чтобы избежать деления на 0

print("\nДанные после стандартизации (первые 5 строк):")
print(df_standardized[cols_to_standardize].head().round(3))

print("\nДанные после нормализации (первые 5 строк):")
print(df_normalized[cols_to_standardize].head().round(3))

# Визуализация распределения после стандартизации
plt.figure(figsize=(20, 15))
cols_to_plot = cols_to_standardize[:min(12, len(cols_to_standardize))]
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df_standardized[col], kde=True, bins=30)
    plt.title(f'Стандартизированное распределение {col}')
plt.tight_layout()
plt.show()

# Сохранение обработанных данных
df_standardized.to_csv('processed_car_data_standardized.csv', index=False)
df_normalized.to_csv('processed_car_data_normalized.csv', index=False)

print("\nОбработка данных завершена! Файлы сохранены.")

# Дополнительная визуализация для проверки результатов
# Корреляционная матрица (только основные признаки для читаемости)
plt.figure(figsize=(16, 12))
main_cols = ['price', 'horsepower', 'enginesize', 'curbweight', 'citympg', 'highwaympg', 
             'power_to_weight', 'fuel_efficiency', 'is_premium_brand']
correlation_matrix = df_standardized[main_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляционная матрица после обработки данных')
plt.show()

# Проверка распределения целевой переменной
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df_standardized['price'], kde=True)
plt.title('Распределение price после стандартизации')

plt.subplot(1, 3, 2)
sns.boxplot(y=df_standardized['price'])
plt.title('Boxplot price после обработки')

plt.subplot(1, 3, 3)
sns.scatterplot(x=df_standardized['horsepower'], y=df_standardized['price'])
plt.title('Зависимость price от horsepower')
plt.tight_layout()
plt.show()

# Финальная информация о данных
print("\nФинальная информация о данных:")
print(df_standardized.info())
print(f"\nФинальный размер датасета: {df_standardized.shape}")