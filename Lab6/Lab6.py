import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. Подготовка и анализ данных
print("=" * 50)
print("1. ПОДГОТОВКА И АНАЛИЗ ДАННЫХ")
print("=" * 50)

# Создадим синтетический dataset для демонстрации
X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, 
                          n_redundant=3, n_clusters_per_class=1, random_state=42)

# Преобразуем в DataFrame для наглядности
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Первые 5 строк датасета:")
print(df.head())
print(f"\nРазмер датасета: {df.shape}")
print(f"\nИнформация о датасете:")
print(df.info())
print(f"\nСтатистика датасета:")
print(df.describe())

# Проверка на пропущенные значения
print(f"\nПропущенные значения:")
print(df.isnull().sum())

# Добавим немного пропущенных значений ТОЛЬКО в признаки для демонстрации обработки
np.random.seed(42)
mask = np.random.random(df[feature_names].shape) < 0.02  # Только для признаков
df_missing = df.copy()
df_missing[feature_names] = df_missing[feature_names].mask(mask)

print(f"\nПропущенные значения после добавления:")
print(df_missing.isnull().sum())

# Обработка пропущенных значений ТОЛЬКО в признаках
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df_missing[feature_names])
df_clean = pd.DataFrame(X_imputed, columns=feature_names)
df_clean['target'] = df['target'].values  # Берем исходные значения целевой переменной

print(f"\nПропущенные значения после обработки:")
print(df_clean.isnull().sum())

# Разделение на признаки и целевую переменную
X = df_clean[feature_names]
y = df_clean['target']

# Проверим, что в y нет NaN
print(f"\nПроверка целевой переменной: NaN в y - {np.isnan(y).sum()}")

# Нормализация/стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

print(f"\nДанные после стандартизации (первые 5 строк):")
print(X_scaled.head())

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмеры выборок:")
print(f"Тренировочная: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Тестовая: X_test {X_test.shape}, y_test {y_test.shape}")

# 2. Обучение модели градиентного бустинга с исходными параметрами
print("\n" + "=" * 50)
print("2. ОБУЧЕНИЕ МОДЕЛИ ГРАДИЕНТНОГО БУСТИНГА")
print("=" * 50)

# Создание и обучение модели с исходными параметрами
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Прогнозы на тестовой части
y_pred = gb_model.predict(X_test)
y_pred_proba = gb_model.predict_proba(X_test)

print("Модель обучена с исходными параметрами")

# 3. Оценка качества модели
print("\n" + "=" * 50)
print("3. ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
print("=" * 50)

# Расчет метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Метрики качества (исходная модель):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nОтчет классификации:")
print(classification_report(y_test, y_pred))

# 4. Визуализация результатов
print("\n" + "=" * 50)
print("4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 50)

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 5))

# Матрица ошибок
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Матрица ошибок (Confusion Matrix)')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')

# Важность признаков
plt.subplot(1, 2, 2)
feature_importance = gb_model.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.title('Важность признаков')
plt.xlabel('Номер признака')
plt.ylabel('Важность')
plt.xticks(range(len(feature_importance)), [f'Feature {i}' for i in indices])

plt.tight_layout()
plt.show()

# 5-6. Настройка гиперпараметров с помощью GridSearchCV
print("\n" + "=" * 50)
print("5-6. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ (GridSearchCV)")
print("=" * 50)

# Определение параметров для поиска
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Выполняется поиск по сетке параметров...")
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nЛучшие параметры:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

print(f"\nЛучшая оценка (F1): {grid_search.best_score_:.4f}")

# 7. Обучение и оценка с новыми параметрами
print("\n" + "=" * 50)
print("7. ОБУЧЕНИЕ И ОЦЕНКА С НОВЫМИ ПАРАМЕТРАМИ")
print("=" * 50)

# Обучение модели с лучшими параметрами
best_gb_model = grid_search.best_estimator_
y_pred_best = best_gb_model.predict(X_test)

# Расчет метрик для улучшенной модели
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print("Метрики качества (оптимизированная модель):")
print(f"Accuracy: {accuracy_best:.4f}")
print(f"Precision: {precision_best:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"F1-Score: {f1_best:.4f}")

print("\nСравнение моделей:")
print(f"{'Метрика':<12} {'Исходная':<10} {'Оптимизированная':<15} {'Изменение':<10}")
print("-" * 50)
print(f"{'Accuracy':<12} {accuracy:.4f}    {accuracy_best:.4f}         {accuracy_best-accuracy:+.4f}")
print(f"{'Precision':<12} {precision:.4f}    {precision_best:.4f}         {precision_best-precision:+.4f}")
print(f"{'Recall':<12} {recall:.4f}    {recall_best:.4f}         {recall_best-recall:+.4f}")
print(f"{'F1-Score':<12} {f1:.4f}    {f1_best:.4f}         {f1_best-f1:+.4f}")

# 8. Исследование влияния скорости обучения и количества деревьев
print("\n" + "=" * 50)
print("8. ИССЛЕДОВАНИЕ ВЛИЯНИЯ ПАРАМЕТРОВ")
print("=" * 50)

# Исследование влияния learning_rate
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
n_estimators_list = [50, 100, 200, 300, 500]

# Влияние learning_rate
train_scores_lr = []
test_scores_lr = []

for lr in learning_rates:
    model = GradientBoostingClassifier(
        learning_rate=lr,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    train_scores_lr.append(train_score)
    test_scores_lr.append(test_score)

# Влияние n_estimators
train_scores_ne = []
test_scores_ne = []

for n_est in n_estimators_list:
    model = GradientBoostingClassifier(
        n_estimators=n_est,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    train_scores_ne.append(train_score)
    test_scores_ne.append(test_score)

# Визуализация влияния параметров
plt.figure(figsize=(15, 5))

# Влияние learning_rate
plt.subplot(1, 2, 1)
plt.plot(learning_rates, train_scores_lr, 'o-', label='Train Score')
plt.plot(learning_rates, test_scores_lr, 'o-', label='Test Score')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Влияние Learning Rate на качество модели')
plt.legend()
plt.grid(True, alpha=0.3)

# Влияние n_estimators
plt.subplot(1, 2, 2)
plt.plot(n_estimators_list, train_scores_ne, 'o-', label='Train Score')
plt.plot(n_estimators_list, test_scores_ne, 'o-', label='Test Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Влияние количества деревьев на качество модели')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ кривых обучения
print("\nАнализ влияния параметров:")
print("Learning Rate:")
for lr, train_acc, test_acc in zip(learning_rates, train_scores_lr, test_scores_lr):
    print(f"  LR={lr:.3f}: Train={train_acc:.4f}, Test={test_acc:.4f}")

print("\nNumber of Estimators:")
for n_est, train_acc, test_acc in zip(n_estimators_list, train_scores_ne, test_scores_ne):
    print(f"  n_est={n_est}: Train={train_acc:.4f}, Test={test_acc:.4f}")

# Финальная оценка
print("\n" + "=" * 50)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("=" * 50)

print("Лучшая модель показала следующие результаты:")
print(f"Accuracy: {accuracy_best:.4f}")
print(f"Precision: {precision_best:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"F1-Score: {f1_best:.4f}")

# Матрица ошибок для лучшей модели
cm_best = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Матрица ошибок - Оптимизированная модель')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.show()

print("\nРабота завершена успешно!")