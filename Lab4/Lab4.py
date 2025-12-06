import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных
print("=" * 50)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 50)

try:
    df = pd.read_csv('healthy_meal_plans.csv')
    print("✓ Файл 'Healthy Meal Plan Dataset.csv' успешно загружен")
except:
    try:
        df = pd.read_csv('healthy_meal_plan.csv')
        print("✓ Файл 'healthy_meal_plan.csv' успешно загружен")
    except:
        try:
            df = pd.read_csv('meal_plan.csv')
            print("✓ Файл 'meal_plan.csv' успешно загружен")
        except Exception as e:
            print(f"✗ Ошибка загрузки файла: {e}")
            exit()

print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print("\nПервые 3 строки:")
print(df.head(3))

# 2. Очистка датасета от пропусков и выбросов
print("\n" + "=" * 50)
print("2. ОЧИСТКА ДАННЫХ")
print("=" * 50)

# 2.1 Обработка пропущенных значений
print("\n2.1 ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
missing_before = df.isnull().sum().sum()
print(f"Пропущенных значений до обработки: {missing_before}")

if missing_before > 0:
    print("Детали пропущенных значений по колонкам:")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} пропусков")
    
    # Заполнение пропусков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        print("✓ Числовые пропуски заполнены медианой")
    
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        print("✓ Категориальные пропуски заполнены модой")

missing_after = df.isnull().sum().sum()
print(f"Пропущенных значений после обработки: {missing_after}")

# 2.2 Обработка выбросов
print("\n2.2 ОБРАБОТКА ВЫБРОСОВ")
numeric_columns = df.select_dtypes(include=[np.number]).columns

if len(numeric_columns) > 0:
    outliers_info = []
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers_count > 0:
            outliers_info.append((col, outliers_count))
            # Обработка выбросов
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    if outliers_info:
        print("Найдены выбросы в колонках:")
        for col, count in outliers_info:
            print(f"  {col}: {count} выбросов")
        print(" Выбросы обработаны (заменены граничными значениями)")
    else:
        print(" Выбросы не обнаружены")
else:
    print(" Числовых колонок для анализа выбросов не найдено")

# 3. Выбор целевой переменной и формирование набора признаков
print("\n" + "=" * 50)
print("3. ВЫБОР ЦЕЛЕВОЙ ПЕРЕМЕННОЙ И ПРИЗНАКОВ")
print("=" * 50)

# Анализ колонок для выбора целевой переменной
print("Анализ колонок для выбора целевой переменной:")
for col in df.columns:
    unique_count = df[col].nunique()
    data_type = df[col].dtype
    print(f"  {col}: {data_type}, {unique_count} уникальных значений")

# Поиск существующей бинарной переменной
binary_candidates = []
for col in df.columns:
    if df[col].nunique() == 2:
        binary_candidates.append(col)

if binary_candidates:
    target_column = binary_candidates[0]
    print(f"\n Найдена бинарная целевая переменная: '{target_column}'")
    print(f"Распределение значений:")
    print(df[target_column].value_counts())
else:
    # Создание бинарной целевой переменной
    print("\n Готовой бинарной переменной не найдено, создаем новую...")
    
    # Выбираем первую числовую колонку для создания бинарной цели
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        base_col = numeric_cols[0]
        median_val = df[base_col].median()
        df['is_high_value'] = (df[base_col] > median_val).astype(int)
        target_column = 'is_high_value'
        print(f"✓ Создана бинарная переменная 'is_high_value' из '{base_col}'")
        print(f"Распределение значений:")
        print(df[target_column].value_counts())
    else:
        # Если нет числовых колонок, используем первую категориальную
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            base_col = categorical_cols[0]
            le = LabelEncoder()
            df['binary_target'] = le.fit_transform(df[base_col])
            # Если больше 2 классов, оставляем только первые два
            if df['binary_target'].nunique() > 2:
                df = df[df['binary_target'].isin([0, 1])].copy()
            target_column = 'binary_target'
            print(f" Создана бинарная переменная из '{base_col}'")
            print(f"Распределение значений:")
            print(df[target_column].value_counts())

# Формирование набора признаков
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"\n✓ Набор признаков сформирован:")
print(f"  Признаки: {X.shape[1]} колонок")
print(f"  Целевая переменная: {target_column}")

# Кодирование категориальных признаков
categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print(f"\nКодируем категориальные признаки: {list(categorical_columns)}")
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    print(f"✓ После кодирования: {X_encoded.shape[1]} признаков")
else:
    X_encoded = X.copy()
    print("✓ Категориальных признаков для кодирования не найдено")

# Стандартизация данных
print("\n2.3 СТАНДАРТИЗАЦИЯ ДАННЫХ")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

print("✓ Данные стандартизированы (StandardScaler)")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n✓ Данные разделены на train/test:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# 4. Создание и тестирование моделей
print("\n" + "=" * 50)
print("4. СОЗДАНИЕ И ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
print("=" * 50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
}

results = {}

# Тестирование каждой модели по отдельности
for i, (name, model) in enumerate(models.items(), 1):
    print(f"\n{'='*30}")
    print(f"МОДЕЛЬ {i}: {name}")
    print(f"{'='*30}")
    
    print(f"Обучение модели...")
    model.fit(X_train, y_train)
    
    print("Предсказание на тестовых данных...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print("✓ Модель обучена и протестирована")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    
    # Детальная оценка для каждой модели
    print("\n  Детальный отчет:")
    print(classification_report(y_test, y_pred))
    
    # Матрица ошибок для каждой модели
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Матрица ошибок\n{name}')
    plt.ylabel('Фактические')
    plt.xlabel('Предсказанные')
    
    # ROC-кривая для каждой модели
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_roc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайный классификатор')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-кривая\n{name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Важность признаков (для Decision Tree)
    if name == 'Decision Tree':
        plt.subplot(1, 3, 3)
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(10)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Важность')
        plt.title('Топ-10 важных признаков')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# 5. Сравнение моделей и выбор лучшей
print("\n" + "=" * 50)
print("5. СРАВНЕНИЕ МОДЕЛЕЙ И ВЫБОР ЛУЧШЕЙ")
print("=" * 50)

# Сравнительная таблица
print("\nСРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ:")
print("-" * 50)
print(f"{'Модель':<20} {'Accuracy':<10} {'AUC-ROC':<10}")
print("-" * 50)
for name in models.keys():
    acc = results[name]['accuracy']
    auc = results[name]['auc_roc']
    print(f"{name:<20} {acc:<10.4f} {auc:<10.4f}")

# Выбор лучшей модели
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
best_auc = results[best_model_name]['auc_roc']

print("\n" + "=" * 30)
print(" ЛУЧШАЯ МОДЕЛЬ")
print("=" * 30)
print(f"Модель: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"AUC-ROC: {best_auc:.4f}")

# Финальная оценка лучшей модели
print(f"\nФИНАЛЬНАЯ ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ ({best_model_name}):")
print("=" * 50)

best_model = results[best_model_name]['model']
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = results[best_model_name]['y_pred_proba']

print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Финальная визуализация
plt.figure(figsize=(15, 5))

# Сравнение всех моделей
plt.subplot(1, 3, 1)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
colors = ['lightblue', 'lightgreen', 'lightcoral']
bars = plt.bar(model_names, accuracies, color=colors)

# Подсветка лучшей модели
best_idx = model_names.index(best_model_name)
bars[best_idx].set_color('gold')
bars[best_idx].set_edgecolor('black')

plt.title('Сравнение Accuracy моделей')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# ROC-кривые всех моделей
plt.subplot(1, 3, 2)
for name in model_names:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    auc_score = results[name]['auc_roc']
    line_style = '-' if name == best_model_name else '--'
    line_width = 3 if name == best_model_name else 2
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', 
             linestyle=line_style, linewidth=line_width)

plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Случайный классификатор')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые всех моделей')
plt.legend()
plt.grid(True, alpha=0.3)

# Финальная матрица ошибок лучшей модели
plt.subplot(1, 3, 3)
cm_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='YlOrRd', cbar=False)
plt.title(f'Матрица ошибок\nЛУЧШАЯ: {best_model_name}')
plt.ylabel('Фактические значения')
plt.xlabel('Предсказанные значения')

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print(" АНАЛИЗ ЗАВЕРШЕН")
print("=" * 50)
print(f"Лучшая модель для данного датасета: {best_model_name}")
print(f"С accuracy: {best_accuracy:.4f}")
print(f"С AUC-ROC: {best_auc:.4f}")

# Дополнительная информация о данных
print(f"\nИНФОРМАЦИЯ О ДАННЫХ:")
print(f"  Исходный размер: {df.shape}")
print(f"  Признаков после обработки: {X_encoded.shape[1]}")
print(f"  Размер train: {X_train.shape}")
print(f"  Размер test: {X_test.shape}")