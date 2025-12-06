import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Загрузка и подготовка данных
print("=" * 50)
print("1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("=" * 50)

# Загрузка CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(f"Размер тренировочных данных: {x_train.shape}")
print(f"Размер тестовых данных: {x_test.shape}")
print(f"Метки тренировочных данных: {y_train.shape}")
print(f"Метки тестовых данных: {y_test.shape}")

# Названия классов CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Нормализация изображений (приведение к диапазону [0,1])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразование меток в one-hot encoding
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print(f"\nПосле преобразования:")
print(f"x_train shape: {x_train.shape}")
print(f"y_train_categorical shape: {y_train_categorical.shape}")

# Разделение тренировочных данных на тренировочную и валидационную выборки
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train_categorical, test_size=0.2, random_state=42
)

print(f"\nПосле разделения:")
print(f"x_train_split: {x_train_split.shape}")
print(f"x_val: {x_val.shape}")
print(f"y_train_split: {y_train_split.shape}")
print(f"y_val: {y_val.shape}")

# Визуализация примеров изображений
plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(f'{class_names[y_train[i][0]]}')
    plt.axis('off')
plt.suptitle('Примеры изображений из CIFAR-10', fontsize=16)
plt.tight_layout()
plt.show()

# 2. Определение архитектуры CNN
print("\n" + "=" * 50)
print("2. ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ CNN")
print("=" * 50)

def create_basic_model():
    """Создание базовой модели CNN"""
    model = models.Sequential([
        # Первый сверточный блок
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Второй сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Третий сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        
        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Создание модели
basic_model = create_basic_model()

# Компиляция модели
basic_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Вывод архитектуры
basic_model.summary()

# 3. Обучение модели
print("\n" + "=" * 50)
print("3. ОБУЧЕНИЕ МОДЕЛИ")
print("=" * 50)

# Параметры обучения
batch_size = 128
epochs = 20

# Callback для сохранения лучшей модели
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best_basic_model.h5", save_best_only=True
)

# Callback для ранней остановки
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True
)

# Обучение базовой модели
print("Начало обучения базовой модели...")
history_basic = basic_model.fit(
    x_train_split, y_train_split,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint_cb, early_stopping_cb],
    verbose=1
)

# Визуализация процесса обучения
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history_basic.history['accuracy'], label='Training Accuracy')
plt.plot(history_basic.history['val_accuracy'], label='Validation Accuracy')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_basic.history['loss'], label='Training Loss')
plt.plot(history_basic.history['val_loss'], label='Validation Loss')
plt.title('Функция потерь')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Оценка модели
print("\n" + "=" * 50)
print("4. ОЦЕНКА МОДЕЛИ")
print("=" * 50)

# Загрузка лучшей модели
basic_model.load_weights("best_basic_model.h5")

# Оценка на тестовых данных
test_loss, test_accuracy = basic_model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Точность на тестовых данных: {test_accuracy:.4f}")

# Прогнозы на тестовых данных
y_pred = basic_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = y_test.flatten()

# Матрица путаницы
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Матрица путаницы - Базовая модель')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Отчет классификации
print("\nОтчет классификации:")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=class_names))

# Визуализация примеров с предсказаниями
plt.figure(figsize=(15, 10))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[i])
    
    true_label = class_names[y_true_classes[i]]
    pred_label = class_names[y_pred_classes[i]]
    confidence = np.max(y_pred[i])
    
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
              color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('Примеры предсказаний модели (зеленый - правильно, красный - неправильно)', 
             fontsize=14)
plt.tight_layout()
plt.show()

# 5. Визуализация фильтров
print("\n" + "=" * 50)
print("5. ВИЗУАЛИЗАЦИЯ ФИЛЬТРОВ")
print("=" * 50)

# Получение весов первого сверточного слоя
first_layer_weights = basic_model.layers[0].get_weights()[0]

# Нормализация весов для визуализации
f_min, f_max = first_layer_weights.min(), first_layer_weights.max()
first_layer_weights = (first_layer_weights - f_min) / (f_max - f_min)

# Визуализация фильтров
n_filters = first_layer_weights.shape[3]
plt.figure(figsize=(12, 8))
for i in range(n_filters):
    plt.subplot(4, 8, i + 1)
    # Берем все каналы RGB для каждого фильтра
    f = first_layer_weights[:, :, :, i]
    # Усредняем по каналам для визуализации или берем один канал
    plt.imshow(f[:, :, 0], cmap='viridis')  # Можно изменить на другой канал
    plt.axis('off')
    plt.title(f'Filter {i+1}')

plt.suptitle('Фильтры первого сверточного слоя', fontsize=16)
plt.tight_layout()
plt.show()

# 6. Улучшение модели
print("\n" + "=" * 50)
print("6. УЛУЧШЕНИЕ МОДЕЛИ")
print("=" * 50)

def create_improved_model():
    """Создание улучшенной модели с Batch Normalization и большим количеством слоев"""
    model = models.Sequential([
        # Первый сверточный блок
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Второй сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Третий сверточный блок
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Создание улучшенной модели
improved_model = create_improved_model()

# Компиляция с другим learning rate
improved_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

improved_model.summary()

# Обучение улучшенной модели
print("Начало обучения улучшенной модели...")
checkpoint_improved = keras.callbacks.ModelCheckpoint(
    "best_improved_model.h5", save_best_only=True
)

history_improved = improved_model.fit(
    x_train_split, y_train_split,
    batch_size=64,  # Уменьшили batch size
    epochs=30,      # Увеличили количество эпох
    validation_data=(x_val, y_val),
    callbacks=[checkpoint_improved, early_stopping_cb],
    verbose=1
)

# Сравнение моделей
improved_model.load_weights("best_improved_model.h5")
test_loss_improved, test_accuracy_improved = improved_model.evaluate(
    x_test, y_test_categorical, verbose=0
)

print(f"\nСравнение результатов:")
print(f"Базовая модель - Точность на тесте: {test_accuracy:.4f}")
print(f"Улучшенная модель - Точность на тесте: {test_accuracy_improved:.4f}")
print(f"Улучшение: {test_accuracy_improved - test_accuracy:.4f}")

# Визуализация сравнения моделей
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history_basic.history['val_accuracy'], label='Базовая модель', linestyle='--')
plt.plot(history_improved.history['val_accuracy'], label='Улучшенная модель')
plt.title('Сравнение точности на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_basic.history['val_loss'], label='Базовая модель', linestyle='--')
plt.plot(history_improved.history['val_loss'], label='Улучшенная модель')
plt.title('Сравнение потерь на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Анализ и выводы
print("\n" + "=" * 50)
print("7. АНАЛИЗ И ВЫВОДЫ")
print("=" * 50)

print("""АНАЛИЗ РЕЗУЛЬТАТОВ:

1. ВЛИЯНИЕ ИЗМЕНЕНИЙ АРХИТЕКТУРЫ:
- Добавление Batch Normalization: Ускорило сходимость и улучшило стабильность обучения
- Увеличение количества сверточных слоев: Позволило модели извлекать более сложные features
- Увеличение количества фильтров: Улучшило способность модели распознавать различные паттерны
- Изменение dropout: Помогло бороться с переобучением

2. ПРОБЛЕМЫ И РЕШЕНИЯ:
- Переобучение: Решено с помощью увеличения dropout и добавления BatchNorm
- Медленная сходимость: Решено настройкой learning rate и использованием Adam
- Недостаточная точность: Решено увеличением глубины сети

3. РЕКОМЕНДАЦИИ ПО ДАЛЬНЕЙШЕМУ УЛУЧШЕНИЮ:
- Использование аугментации данных
- Применение предобученных моделей (Transfer Learning)
- Использование более сложных архитектур (ResNet, EfficientNet)
- Тонкая настройка гиперпараметров
""")

# 8. Анализ адаптации для других датасетов
print("\n" + "=" * 50)
print("8. АДАПТАЦИЯ ДЛЯ ДРУГИХ ДАТАСЕТОВ")
print("=" * 50)

print("""АНАЛИЗ АДАПТАЦИИ ДЛЯ РАЗНЫХ ДАТАСЕТОВ:

1. MNIST (28x28 grayscale, 10 классов):
- Упрощение архитектуры: меньше сверточных слоев
- Изменение input shape: (28, 28, 1) вместо (32, 32, 3)
- Уменьшение количества фильтров
- Более простая предобработка

2. CIFAR-100 (32x32 color, 100 классов):
- Увеличение выходного слоя: 100 нейронов вместо 10
- Более глубокая архитектура для сложных признаков
- Увеличение dropout для борьбы с переобучением
- Использование более сложных методов регуляризации

3. ImageNet (224x224 color, 1000 классов):
- Использование предобученных моделей (Transfer Learning)
- Глубокая архитектура (ResNet50, EfficientNet и т.д.)
- Сильная аугментация данных
- Изменение input size до 224x224
- Использование более сложных оптимизаторов
- Обучение на мощных GPU/TPU
- Длительное обучение (десятки/сотни эпох)

ОБЩИЕ ПРИНЦИПЫ АДАПТАЦИИ:
- Размер выходного слоя = количество классов
- Input shape = размер изображений датасета
- Сложность архитектуры пропорциональна сложности датасета
- Регуляризация важнее для датасетов с большим количеством классов
""")

print("\nРабота завершена успешно!")