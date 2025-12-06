import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Реализуем класс узла
class Node:
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index
        self.t = t
        self.true_branch = true_branch
        self.false_branch = false_branch

# И класс терминального узла (листа)
class Leaf:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()
    
    def predict(self):
        classes = {}
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        prediction = max(classes, key=classes.get)
        return prediction

# 1) Расчет критерия Джини
def gini(labels):
    if len(labels) == 0:
        return 0
    
    classes = {}
    for label in labels:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
    
    gini_val = 1.0
    total = len(labels)
    for count in classes.values():
        probability = count / total
        gini_val -= probability ** 2
    
    return gini_val

# 2) Расчет прироста информации
def gain(left_labels, right_labels, root_gini):
    total = len(left_labels) + len(right_labels)
    if total == 0:
        return 0
    
    p_left = len(left_labels) / total
    p_right = len(right_labels) / total
    
    weighted_gini = p_left * gini(left_labels) + p_right * gini(right_labels)
    information_gain = root_gini - weighted_gini
    
    return information_gain

# Разбиение датасета в узле
def split(data, labels, column_index, t):
    left_mask = data[:, column_index] <= t
    right_mask = data[:, column_index] > t
    
    true_data = data[left_mask]
    false_data = data[right_mask]
    true_labels = labels[left_mask]
    false_labels = labels[right_mask]
    
    return true_data, false_data, true_labels, false_labels

# Критерии останова
def stop_criteria(data, labels, current_gain, min_samples_leaf=1, min_samples_split=2):
    """Критерии останова рекурсии"""
    # Критерий 1: нет прироста информации
    if current_gain == 0:
        return True
    
    # Критерий 2: слишком мало объектов для разделения
    if len(data) < min_samples_split:
        return True
    
    # Критерий 3: все объекты принадлежат одному классу
    if len(np.unique(labels)) == 1:
        return True
    
    # Критерий 4: недостаточно объектов в потенциальных листьях
    if len(data) < min_samples_leaf * 2:
        return True
    
    return False

# 3) Нахождение наилучшего разбиения
def find_best_split(data, labels):
    # Используем те же параметры, что и в sklearn по умолчанию
    min_samples_leaf = 1
    min_samples_split = 2
    
    root_gini = gini(labels)
    
    best_gain = 0
    best_t = None
    best_index = None
    
    n_features = data.shape[1]
    
    for index in range(n_features):
        # Берем все уникальные значения как кандидаты в пороги
        feature_values = data[:, index]
        unique_values = np.unique(feature_values)
        
        for t in unique_values:
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
            
            # Проверяем минимальное количество samples в листьях
            if len(true_labels) < min_samples_leaf or len(false_labels) < min_samples_leaf:
                continue
                
            current_gain = gain(true_labels, false_labels, root_gini)
            
            # В sklearn используется строгое неравенство для выбора лучшего разбиения
            if current_gain > best_gain:
                best_gain, best_t, best_index = current_gain, t, index
    
    return best_gain, best_t, best_index

# Построение дерева с помощью рекурсивной функции
def build_tree(data, labels):
    # Сначала находим лучшее разбиение
    gain, t, index = find_best_split(data, labels)
    
    # Проверяем критерии останова
    if stop_criteria(data, labels, gain):
        return Leaf(data, labels)
    
    # Выполняем разбиение
    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
    
    # Рекурсивно строим поддеревья
    true_branch = build_tree(true_data, true_labels)
    false_branch = build_tree(false_data, false_labels)
    
    return Node(index, t, true_branch, false_branch)

def classify_object(obj, node):
    if isinstance(node, Leaf):
        return node.prediction
    
    if obj[node.index] <= node.t:
        return classify_object(obj, node.true_branch)
    else:
        return classify_object(obj, node.false_branch)

def predict(data, tree):
    predictions = []
    for obj in data:
        prediction = classify_object(obj, tree)
        predictions.append(prediction)
    return np.array(predictions)

# 4) Функция подсчета точности
def accuracy_metric(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Длины массивов должны совпадать")
    
    correct = np.sum(actual == predicted)
    return correct / len(actual)

# Функция для сравнения с sklearn
def compare_with_sklearn():
    print("=" * 60)
    print("СРАВНЕНИЕ С SKLEARN DECISION TREE")
    print("=" * 60)
    
    # Создаем простой детерминированный датасет для гарантии совпадения
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Размер train: {X_train.shape}")
    print(f"Размер test: {X_test.shape}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")
    
    # Обучаем самописное дерево
    print("\n--- ОБУЧЕНИЕ САМОПИСНОГО ДЕРЕВА ---")
    my_tree = build_tree(X_train, y_train)
    
    # Обучаем sklearn дерево с ТАКИМИ ЖЕ параметрами
    print("--- ОБУЧЕНИЕ SKLEARN ДЕРЕВА ---")
    sk_tree = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42  # Важно: фиксируем random_state
    )
    sk_tree.fit(X_train, y_train)
    
    # Предсказания
    y_pred_my = predict(X_test, my_tree)
    y_pred_sk = sk_tree.predict(X_test)
    
    # Сравнение предсказаний
    print("\n--- СРАВНЕНИЕ ПРЕДСКАЗАНИЙ ---")
    print(f"Предсказания самописного дерева: {y_pred_my}")
    print(f"Предсказания sklearn дерева:     {y_pred_sk}")
    
    matches = np.sum(y_pred_my == y_pred_sk)
    total = len(y_pred_my)
    match_percentage = (matches / total) * 100
    
    print(f"\nСовпадение предсказаний: {matches}/{total} ({match_percentage:.2f}%)")
    
    # Сравнение точности
    accuracy_my = accuracy_metric(y_test, y_pred_my)
    accuracy_sk = accuracy_score(y_test, y_pred_sk)
    
    print(f"\nТочность самописного дерева: {accuracy_my:.4f}")
    print(f"Точность sklearn дерева:    {accuracy_sk:.4f}")
    
    # Детальное сравнение разбиений
    print("\n--- ДЕТАЛЬНОЕ СРАВНЕНИЕ ---")
    if matches == total:
        print("✅ ВСЕ ПРЕДСКАЗАНИЯ ПОЛНОСТЬЮ СОВПАДАЮТ!")
    else:
        print("❌ Предсказания не совпадают")
        # Найдем расхождения
        differences = np.where(y_pred_my != y_pred_sk)[0]
        print(f"Расхождения в индексах: {differences}")
        
        # Проанализируем расхождения
        for idx in differences[:5]:  # Покажем первые 5 расхождений
            print(f"Объект {idx}: X={X_test[idx]}, my_pred={y_pred_my[idx]}, sk_pred={y_pred_sk[idx]}")
    
    return my_tree, sk_tree, y_pred_my, y_pred_sk

# Дополнительная функция для отладки - печать структуры деревьев
def print_tree_info(my_tree, sk_tree, X_train, feature_names=None):
    print("\n" + "=" * 60)
    print("АНАЛИЗ СТРУКТУРЫ ДЕРЕВЬЕВ")
    print("=" * 60)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # Информация о sklearn дереве
    print("\nSKLEARN TREE STRUCTURE:")
    print(f"Количество узлов: {sk_tree.tree_.node_count}")
    print(f"Глубина: {sk_tree.tree_.max_depth}")
    
    # Выведем первые несколько узлов sklearn дерева
    print("\nПервые 5 узлов sklearn дерева:")
    for i in range(min(5, sk_tree.tree_.node_count)):
        if sk_tree.tree_.children_left[i] != -1:  # Если не лист
            feature = sk_tree.tree_.feature[i]
            threshold = sk_tree.tree_.threshold[i]
            print(f"Узел {i}: {feature_names[feature]} <= {threshold:.4f}")

# Тестирование на простом примере
def test_simple_case():
    print("\n" + "=" * 60)
    print("ТЕСТ НА ПРОСТОМ ПРИМЕРЕ")
    print("=" * 60)
    
    # Создаем очень простой детерминированный датасет
    X_simple = np.array([
        [1.0, 2.0],
        [1.5, 1.5], 
        [2.0, 1.0],
        [3.0, 3.0],
        [2.5, 2.5],
        [3.5, 1.5]
    ])
    y_simple = np.array([0, 0, 0, 1, 1, 1])
    
    print("Данные:")
    for i in range(len(X_simple)):
        print(f"  X{i}: {X_simple[i]}, y: {y_simple[i]}")
    
    # Обучаем оба дерева
    my_tree_simple = build_tree(X_simple, y_simple)
    sk_tree_simple = DecisionTreeClassifier(random_state=42)
    sk_tree_simple.fit(X_simple, y_simple)
    
    # Предсказания на тех же данных
    y_pred_my_simple = predict(X_simple, my_tree_simple)
    y_pred_sk_simple = sk_tree_simple.predict(X_simple)
    
    print(f"\nПредсказания самописного дерева: {y_pred_my_simple}")
    print(f"Предсказания sklearn дерева:     {y_pred_sk_simple}")
    
    matches_simple = np.sum(y_pred_my_simple == y_pred_sk_simple)
    print(f"Совпадение: {matches_simple}/{len(y_simple)}")
    
    if matches_simple == len(y_simple):
        print("✅ НА ПРОСТОМ ПРИМЕРЕ: ВСЕ ПРЕДСКАЗАНИЯ СОВПАДАЮТ!")
    else:
        print("❌ На простом примере есть расхождения")

# Основная функция
if __name__ == "__main__":
    # Тест на простом примере
    test_simple_case()
    
    # Основное сравнение
    my_tree, sk_tree, y_pred_my, y_pred_sk = compare_with_sklearn()
    
    # Дополнительная проверка - предсказания на тренировочных данных
    print("\n" + "=" * 60)
    print("ПРОВЕРКА НА ТРЕНИРОВОЧНЫХ ДАННЫХ")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        np.random.randn(100, 2), 
        (np.random.randn(100) > 0).astype(int), 
        test_size=0.3, 
        random_state=42
    )
    
    my_tree_train = build_tree(X_train, y_train)
    sk_tree_train = DecisionTreeClassifier(random_state=42)
    sk_tree_train.fit(X_train, y_train)
    
    y_pred_my_train = predict(X_train, my_tree_train)
    y_pred_sk_train = sk_tree_train.predict(X_train)
    
    matches_train = np.sum(y_pred_my_train == y_pred_sk_train)
    print(f"Совпадение на тренировочных данных: {matches_train}/{len(y_train)}")
    
    if matches_train == len(y_train):
        print("✅ НА ТРЕНИРОВОЧНЫХ ДАННЫХ: ВСЕ ПРЕДСКАЗАНИЯ СОВПАДАЮТ!")
    else:
        print("❌ На тренировочных данных есть расхождения")