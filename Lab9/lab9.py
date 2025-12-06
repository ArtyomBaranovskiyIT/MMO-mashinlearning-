import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================

print("=" * 60)
print("1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

# Параметры
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 10

# Загрузка данных
print("Загрузка IMDB датасета...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# Анализ данных
print(f"Размер тренировочных данных: {len(X_train)}")
print(f"Размер тестовых данных: {len(X_test)}")
print(f"Пример метки: {y_train[0]}")
print(f"Длина первой последовательности: {len(X_train[0])}")

# Анализ длины последовательностей
seq_lens = [len(seq) for seq in X_train]
print(f"Средняя длина последовательности: {np.mean(seq_lens):.2f}")
print(f"Максимальная длина: {max(seq_lens)}")
print(f"Минимальная длина: {min(seq_lens)}")

# Обрезка/дополнение последовательностей
print("Обработка последовательностей...")
X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Форма тренировочных данных: {X_train.shape}")
print(f"Форма тестовых данных: {X_test.shape}")

# Разделение на тренировочную и валидационную выборки
X_val = X_train[:5000]
y_val = y_train[:5000]
X_train_partial = X_train[5000:]
y_train_partial = y_train[5000:]

print(f"Тренировочные данные: {X_train_partial.shape}")
print(f"Валидационные данные: {X_val.shape}")

# =============================================================================
# 2. СОЗДАНИЕ МОДЕЛЕЙ RNN
# =============================================================================

print("\n" + "=" * 60)
print("2. СОЗДАНИЕ МОДЕЛЕЙ RNN")
print("=" * 60)

def create_rnn_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, 
                    max_len=MAX_LEN, rnn_units=64, dropout_rate=0.5):
    """Создание модели SimpleRNN"""
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=max_len),
        
        SimpleRNN(units=rnn_units, 
                 dropout=dropout_rate, 
                 recurrent_dropout=0.2,
                 return_sequences=False),
        
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    return model

def create_lstm_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, 
                     max_len=MAX_LEN, lstm_units=64, dropout_rate=0.5):
    """Создание модели LSTM"""
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=max_len),
        
        LSTM(units=lstm_units, 
             dropout=dropout_rate, 
             recurrent_dropout=0.2,
             return_sequences=False),
        
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    return model

def create_gru_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, 
                    max_len=MAX_LEN, gru_units=64, dropout_rate=0.5):
    """Создание модели GRU"""
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=max_len),
        
        GRU(units=gru_units, 
            dropout=dropout_rate, 
            recurrent_dropout=0.2,
            return_sequences=False),
        
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    return model

def create_bidirectional_lstm(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, 
                             max_len=MAX_LEN, lstm_units=64, dropout_rate=0.5):
    """Создание Bidirectional LSTM модели"""
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=max_len),
        
        Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=0.2)),
        
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    return model

# =============================================================================
# 3. ОБУЧЕНИЕ И СРАВНЕНИЕ АРХИТЕКТУР
# =============================================================================

print("\n" + "=" * 60)
print("3. ОБУЧЕНИЕ И СРАВНЕНИЕ АРХИТЕКТУР")
print("=" * 60)

def train_and_evaluate_model(model, model_name):
    """Обучение и оценка модели"""
    print(f"\nОбучение {model_name} модели...")
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    history = model.fit(X_train_partial, y_train_partial,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=(X_val, y_val),
                       verbose=1)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return {
        'model': model,
        'history': history,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }

# Создание и обучение моделей
models = {
    'SimpleRNN': create_rnn_model(),
    'LSTM': create_lstm_model(),
    'GRU': create_gru_model(),
    'Bidirectional_LSTM': create_bidirectional_lstm()
}

results = {}

for name, model in models.items():
    results[name] = train_and_evaluate_model(model, name)
    print(f"{name} - Test Accuracy: {results[name]['test_accuracy']:.4f}")

# =============================================================================
# 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================

print("\n" + "=" * 60)
print("4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 60)

# Функция для визуализации истории обучения
def plot_training_history(history_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for name, result in history_dict.items():
        history = result['history']
        # График потерь
        ax1.plot(history.history['loss'], label=f'{name} Train', linestyle='--')
        ax1.plot(history.history['val_loss'], label=f'{name} Val')
        
        # График точности
        ax2.plot(history.history['accuracy'], label=f'{name} Train', linestyle='--')
        ax2.plot(history.history['val_accuracy'], label=f'{name} Val')
    
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Визуализация процесса обучения
plot_training_history(results)

# Сравнение точности моделей
model_names = list(results.keys())
accuracies = [results[name]['test_accuracy'] for name in model_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title('Сравнение точности моделей на тестовых данных')
plt.ylabel('Accuracy')
plt.ylim(0.7, 0.9)

# Добавление значений на столбцы
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{accuracy:.4f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 5. ДЕТАЛЬНАЯ ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ
# =============================================================================

print("\n" + "=" * 60)
print("5. ДЕТАЛЬНАЯ ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
print("=" * 60)

# Выбор лучшей модели
best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']
print(f"Лучшая модель: {best_model_name}")

# Предсказания на тестовых данных
y_pred_proba = best_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Отчет о классификации
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Анализ ошибок
misclassified_indices = np.where(y_pred != y_test)[0]
print(f"Количество неверно классифицированных примеров: {len(misclassified_indices)}")
print(f"Процент ошибок: {len(misclassified_indices)/len(y_test)*100:.2f}%")

# Функция для декодирования текста
def decode_review(encoded_review):
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 0])

# Покажем несколько примеров ошибок
print("\nПримеры ошибок классификации:")
for i in misclassified_indices[:3]:
    print(f"\nПример {i}:")
    print(f"True: {'Positive' if y_test[i] == 1 else 'Negative'}")
    print(f"Predicted: {'Positive' if y_pred[i] == 1 else 'Negative'}")
    print(f"Confidence: {y_pred_proba[i][0]:.4f}")
    review_text = decode_review(X_test[i])
    print(f"Текст: {review_text[:200]}...")

# =============================================================================
# 6. ЭКСПЕРИМЕНТЫ С ГИПЕРПАРАМЕТРАМИ
# =============================================================================

print("\n" + "=" * 60)
print("6. ЭКСПЕРИМЕНТЫ С ГИПЕРПАРАМЕТРАМИ")
print("=" * 60)

def experiment_hyperparameters():
    """Эксперименты с различными гиперпараметрами"""
    experiments = []
    
    # Различные длины последовательностей
    seq_lengths = [100, 200, 300]
    
    for seq_len in seq_lengths:
        print(f"\nЭксперимент с длиной последовательности: {seq_len}")
        
        # Подготовка данных с новой длиной
        X_train_exp = pad_sequences(X_train, maxlen=seq_len, padding='post', truncating='post')
        X_test_exp = pad_sequences(X_test, maxlen=seq_len, padding='post', truncating='post')
        X_val_exp = X_train_exp[:5000]
        X_train_partial_exp = X_train_exp[5000:]
        
        model = create_lstm_model(max_len=seq_len)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_partial_exp, y_train_partial,
                          batch_size=BATCH_SIZE,
                          epochs=5,
                          validation_data=(X_val_exp, y_val),
                          verbose=0)
        
        test_loss, test_accuracy = model.evaluate(X_test_exp, y_test, verbose=0)
        
        experiments.append({
            'parameter': 'sequence_length',
            'value': seq_len,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        })
        
        print(f"Длина {seq_len}: Accuracy = {test_accuracy:.4f}")
    
    # Различные размеры embedding
    embedding_dims = [50, 100, 200]
    
    for emb_dim in embedding_dims:
        print(f"\nЭксперимент с размером embedding: {emb_dim}")
        
        model = create_lstm_model(embedding_dim=emb_dim)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(X_train_partial, y_train_partial,
                          batch_size=BATCH_SIZE,
                          epochs=5,
                          validation_data=(X_val, y_val),
                          verbose=0)
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        experiments.append({
            'parameter': 'embedding_dim',
            'value': emb_dim,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        })
        
        print(f"Embedding dim {emb_dim}: Accuracy = {test_accuracy:.4f}")
    
    return experiments

# Запуск экспериментов
print("Запуск экспериментов с гиперпараметрами...")
experiment_results = experiment_hyperparameters()

# Визуализация результатов экспериментов
seq_len_results = [r for r in experiment_results if r['parameter'] == 'sequence_length']
emb_dim_results = [r for r in experiment_results if r['parameter'] == 'embedding_dim']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# График для длины последовательности
ax1.bar([str(r['value']) for r in seq_len_results], 
        [r['test_accuracy'] for r in seq_len_results])
ax1.set_title('Влияние длины последовательности на точность')
ax1.set_xlabel('Длина последовательности')
ax1.set_ylabel('Accuracy')
ax1.grid(True, alpha=0.3)

# График для размера embedding
ax2.bar([str(r['value']) for r in emb_dim_results], 
        [r['test_accuracy'] for r in emb_dim_results])
ax2.set_title('Влияние размера embedding на точность')
ax2.set_xlabel('Размер embedding')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 7. РЕШЕНИЕ ПРОБЛЕМЫ ИСЧЕЗАЮЩЕГО ГРАДИЕНТА
# =============================================================================

print("\n" + "=" * 60)
print("7. РЕШЕНИЕ ПРОБЛЕМЫ ИСЧЕЗАЮЩЕГО ГРАДИЕНТА")
print("=" * 60)

print("""
ПРЕДЛОЖЕННЫЕ РЕШЕНИЯ ПРОБЛЕМЫ ИСЧЕЗАЮЩЕГО ГРАДИЕНТА:

1. ИСПОЛЬЗОВАНИЕ LSTM/GRU:
   - LSTM: Решает проблему через механизмы вентилей (forget, input, output gates)
   - GRU: Упрощенная версия LSTM с двумя вентилями (update и reset gates)

2. GRADIENT CLIPPING:
   - Ограничение величины градиента во время обратного распространения
   - optimizer = Adam(learning_rate=0.001, clipvalue=1.0)

3. ПРОПУСК СВЯЗЕЙ (SKIP CONNECTIONS):
   - Добавление residual connections между слоями
   - Позволяет градиентам проходить напрямую

4. BIDIRECTIONAL RNN:
   - Обработка последовательности в обоих направлениях
   - Улучшает понимание контекста

5. ПРАВИЛЬНАЯ ИНИЦИАЛИЗАЦИЯ:
   - Использование orthogonal инициализации для рекуррентных весов

6. NORMALIZATION LAYERS:
   - Layer Normalization в рекуррентных слоях
   - Batch Normalization между слоями

7. АЛЬТЕРНАТИВНЫЕ АРХИТЕКТУРЫ:
   - Transformer модели с self-attention
   - CNN + RNN гибридные архитектуры
""")

# Демонстрация решения с gradient clipping
print("\nСоздание модели с Gradient Clipping...")
model_clipped = create_lstm_model()
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
model_clipped.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Быстрое обучение для демонстрации
history_clipped = model_clipped.fit(X_train_partial, y_train_partial,
                                   batch_size=BATCH_SIZE,
                                   epochs=3,
                                   validation_data=(X_val, y_val),
                                   verbose=1)

clipped_test_loss, clipped_test_accuracy = model_clipped.evaluate(X_test, y_test, verbose=0)
print(f"Модель с Gradient Clipping - Test Accuracy: {clipped_test_accuracy:.4f}")

# =============================================================================
# 8. ИТОГОВЫЕ ВЫВОДЫ И АНАЛИЗ
# =============================================================================

print("\n" + "=" * 60)
print("8. ИТОГОВЫЕ ВЫВОДЫ И АНАЛИЗ")
print("=" * 60)

# Сводка результатов
print("\nСВОДКА РЕЗУЛЬТАТОВ:")
print("-" * 50)
for name in results:
    accuracy = results[name]['test_accuracy']
    print(f"{name:<20}: {accuracy:.4f}")

# Анализ лучшей модели
best_result = max(results.values(), key=lambda x: x['test_accuracy'])
best_model_name = [name for name, result in results.items() if result == best_result][0]

print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
print(f"ТОЧНОСТЬ: {best_result['test_accuracy']:.4f}")

# Анализ переобучения
print("\nАНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
for name, result in results.items():
    history = result['history']
    train_final = history.history['accuracy'][-1]
    val_final = history.history['val_accuracy'][-1]
    gap = train_final - val_final
    status = "ПЕРЕОБУЧЕНИЕ" if gap > 0.05 else "НОРМА"
    print(f"{name:<20}: Train={train_final:.4f}, Val={val_final:.4f}, Gap={gap:.4f} ({status})")

# Рекомендации
print("""
ВЫВОДЫ И РЕКОМЕНДАЦИИ:

1. АРХИТЕКТУРЫ:
   - LSTM и GRU значительно превосходят SimpleRNN
   - Bidirectional LSTM часто показывает лучшие результаты
   - SimpleRNN страдает от проблемы исчезающего градиента

2. ГИПЕРПАРАМЕТРЫ:
   - Оптимальная длина последовательности: 200-300 слов
   - Размер embedding: 100-200 измерений
   - Learning rate 0.001 показывает стабильные результаты

3. РЕГУЛЯРИЗАЦИЯ:
   - Dropout критически важен для предотвращения переобучения
   - Recurrent dropout улучшает стабильность RNN
   - Gradient clipping помогает с обучением

4. ПРАКТИЧЕСКИЕ СОВЕТЫ:
   - Начинать с LSTM/GRU вместо SimpleRNN
   - Использовать bidirectional архитектуры для NLP
   - Мониторить разницу между train и validation accuracy
   - Экспериментировать с разными размерами embedding
""")

# Финальная визуализация сравнения всех моделей
final_comparison = {
    'Models': list(results.keys()),
    'Accuracy': [results[name]['test_accuracy'] for name in results.keys()],
    'Loss': [results[name]['test_loss'] for name in results.keys()]
}

x = np.arange(len(final_comparison['Models']))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))

# Столбцы для accuracy
bars1 = ax1.bar(x - width/2, final_comparison['Accuracy'], width, label='Accuracy', color='lightblue')
ax1.set_xlabel('Модели')
ax1.set_ylabel('Accuracy', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0.7, 0.9)

# Вторая ось для loss
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, final_comparison['Loss'], width, label='Loss', color='lightcoral')
ax2.set_ylabel('Loss', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0.3, 0.6)

# Настройки графика
ax1.set_title('Финальное сравнение всех моделей')
ax1.set_xticks(x)
ax1.set_xticklabels(final_comparison['Models'])
ax1.grid(True, alpha=0.3)

# Легенда
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Добавление значений на столбцы
for bar, accuracy in zip(bars1, final_comparison['Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{accuracy:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН!")
print("=" * 60)