import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights = None
        self.loss_history = []
        
    def _initialize_weights(self, n_features):
        """Инициализация весов"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights = np.random.randn(n_features + 1)  # +1 для bias
    
    def _add_bias(self, X):
        """Добавление столбца единиц для bias"""
        return np.c_[np.ones(X.shape[0]), X]
    
    def predict(self, X):
        """Предсказание"""
        X_with_bias = self._add_bias(X)
        return X_with_bias @ self.weights
    
    def compute_gradient(self, X, y, weights):
        """Вычисление градиента"""
        X_with_bias = self._add_bias(X)
        n_samples = X.shape[0]
        predictions = X_with_bias @ weights
        errors = predictions - y
        gradient = (X_with_bias.T @ errors) / n_samples
        return gradient
    
    def compute_loss(self, X, y, weights):
        """Вычисление MSE"""
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)

class BatchGradientDescent(GradientDescent):
    """Пакетный градиентный спуск"""
    
    def fit(self, X, y):
        """Обучение модели"""
        self._initialize_weights(X.shape[1])
        X_with_bias = self._add_bias(X)
        n_samples = X.shape[0]
        
        for i in range(self.max_iter):
            # Расчет градиента
            predictions = X_with_bias @ self.weights
            errors = predictions - y
            gradient = (X_with_bias.T @ errors) / n_samples
            
            # Обновление весов
            new_weights = self.weights - self.learning_rate * gradient
            
            # Проверка сходимости
            if np.linalg.norm(new_weights - self.weights) < self.tol:
                print(f"Сходимость достигнута на итерации {i+1}")
                break
                
            self.weights = new_weights
            current_loss = self.compute_loss(X, y, self.weights)
            self.loss_history.append(current_loss)
            
        return self

class StochasticGradientDescent(GradientDescent):
    """Стохастический градиентный спуск"""
    
    def fit(self, X, y):
        """Обучение модели"""
        self._initialize_weights(X.shape[1])
        n_samples = X.shape[0]
        
        for i in range(self.max_iter):
            # Выбор случайного объекта
            random_idx = np.random.randint(n_samples)
            X_random = X[random_idx:random_idx+1]
            y_random = y[random_idx:random_idx+1]
            
            # Расчет градиента для одного объекта
            gradient = self.compute_gradient(X_random, y_random, self.weights)
            
            # Обновление весов
            new_weights = self.weights - self.learning_rate * gradient
            
            # Проверка сходимости (реже для SGD)
            if i % 100 == 0 and np.linalg.norm(new_weights - self.weights) < self.tol:
                print(f"Сходимость достигнута на итерации {i+1}")
                break
                
            self.weights = new_weights
            current_loss = self.compute_loss(X, y, self.weights)
            self.loss_history.append(current_loss)
            
        return self

class RegularizedGradientDescent(GradientDescent):
    """Градиентный спуск с L2 регуляризацией"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, 
                 random_state=None, alpha=0.01):
        super().__init__(learning_rate, max_iter, tol, random_state)
        self.alpha = alpha  # коэффициент регуляризации
    
    def fit(self, X, y):
        """Обучение модели с L2 регуляризацией"""
        self._initialize_weights(X.shape[1])
        X_with_bias = self._add_bias(X)
        n_samples = X.shape[0]
        
        for i in range(self.max_iter):
            # Расчет градиента с регуляризацией
            gradient = self.compute_gradient(X, y, self.weights)
            
            # Обновление весов
            new_weights = self.weights - self.learning_rate * gradient
            
            # Проверка сходимости
            if np.linalg.norm(new_weights - self.weights) < self.tol:
                print(f"Сходимость достигнута на итерации {i+1}")
                break
                
            self.weights = new_weights
            current_loss = self.compute_loss(X, y, self.weights)
            self.loss_history.append(current_loss)
            
        return self
    
    def compute_gradient(self, X, y, weights):
        """Вычисление градиента с L2 регуляризацией"""
        X_with_bias = self._add_bias(X)
        n_samples = X.shape[0]
        predictions = X_with_bias @ weights
        errors = predictions - y
        
        # Градиент MSE
        gradient_mse = (X_with_bias.T @ errors) / n_samples
        
        # Градиент L2 регуляризации (не применяем к bias)
        gradient_reg = self.alpha * weights
        gradient_reg[0] = 0  # не регуляризуем bias
        
        return gradient_mse + gradient_reg
    
    def compute_loss(self, X, y, weights):
        """Вычисление MSE с L2 регуляризацией"""
        X_with_bias = self._add_bias(X)
        n_samples = X.shape[0]
        predictions = X_with_bias @ weights
        mse = np.mean((predictions - y) ** 2)
        
        # L2 регуляризация (не применяем к bias)
        l2_penalty = self.alpha * np.sum(weights[1:] ** 2)
        
        return mse + l2_penalty

def experiment_learning_rates(X_train, y_train, X_test, y_test):
    """Эксперимент с разными скоростями обучения"""
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        # Пакетный градиентный спуск
        bgd = BatchGradientDescent(learning_rate=lr, max_iter=1000)
        bgd.fit(X_train, y_train)
        
        plt.subplot(2, 2, i+1)
        plt.plot(bgd.loss_history)
        plt.title(f'BGD, learning_rate={lr}\nFinal MSE: {bgd.loss_history[-1]:.4f}')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_gd_sgd(X_train, y_train, X_test, y_test):
    """Сравнение градиентного и стохастического градиентного спуска"""
    # Пакетный градиентный спуск
    bgd = BatchGradientDescent(learning_rate=0.01, max_iter=1000)
    bgd.fit(X_train, y_train)
    
    # Стохастический градиентный спуск
    sgd = StochasticGradientDescent(learning_rate=0.01, max_iter=1000)
    sgd.fit(X_train, y_train)
    
    # График сравнения
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(bgd.loss_history, label='BGD', alpha=0.7)
    plt.plot(sgd.loss_history, label='SGD', alpha=0.7)
    plt.title('Сравнение BGD и SGD')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Сглаживание для лучшей визуализации
    window = 50
    sgd_smooth = np.convolve(sgd.loss_history, np.ones(window)/window, mode='valid')
    plt.plot(bgd.loss_history[:len(sgd_smooth)], label='BGD', alpha=0.7)
    plt.plot(sgd_smooth, label='SGD (сглаженный)', alpha=0.7)
    plt.title('Сравнение BGD и SGD (сглаженный)')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод результатов
    bgd_test_mse = bgd.compute_loss(X_test, y_test, bgd.weights)
    sgd_test_mse = sgd.compute_loss(X_test, y_test, sgd.weights)
    
    print(f"BGD Final Train MSE: {bgd.loss_history[-1]:.6f}")
    print(f"SGD Final Train MSE: {sgd.loss_history[-1]:.6f}")
    print(f"BGD Test MSE: {bgd_test_mse:.6f}")
    print(f"SGD Test MSE: {sgd_test_mse:.6f}")

def l2_regularization_experiment(X_train, y_train):
    """Эксперимент с L2 регуляризацией"""
    # Широкий диапазон коэффициентов регуляризации
    alphas = np.logspace(-3, 3, 20)  # от 0.001 до 1000
    weights_history = []
    
    for alpha in alphas:
        gd = RegularizedGradientDescent(learning_rate=0.01, max_iter=1000, alpha=alpha)
        gd.fit(X_train, y_train)
        weights_history.append(gd.weights.copy())
    
    weights_history = np.array(weights_history)
    
    # График изменения весов
    plt.figure(figsize=(12, 8))
    
    for i in range(weights_history.shape[1]):
        if i == 0:
            label = f'Bias (w0)'
        else:
            label = f'w{i}'
        plt.plot(alphas, weights_history[:, i], label=label, linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Коэффициент регуляризации (alpha)')
    plt.ylabel('Значение веса')
    plt.title('Изменение весов в зависимости от коэффициента L2 регуляризации')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return alphas, weights_history

def main():
    """Основная функция"""
    print("=== Лабораторная работа: Градиентный спуск и регуляризация ===\n")
    
    # 1. Генерация датасета
    print("1. Генерация датасета...")
    X, y = make_regression(n_samples=100000, n_features=5, noise=0.1, random_state=42)
    
    # Добавим еще один признак для разнообразия
    X_extra = np.random.randn(100000, 2)
    X = np.hstack([X, X_extra])
    
    print(f"Размерность данных: {X.shape}")
    print(f"Количество признаков: {X.shape[1]}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Масштабирование данных выполнено")
    
    # 2. Эксперимент со скоростями обучения
    print("\n2. Эксперимент со скоростями обучения...")
    experiment_learning_rates(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 3. Сравнение BGD и SGD
    print("\n3. Сравнение градиентного и стохастического градиентного спуска...")
    compare_gd_sgd(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 4. Эксперимент с L2 регуляризацией
    print("\n4. Эксперимент с L2 регуляризацией...")
    alphas, weights_history = l2_regularization_experiment(X_train_scaled, y_train)
    
    # Дополнительный анализ регуляризации
    plt.figure(figsize=(12, 6))
    
    # Нормы весов
    weights_norms = np.linalg.norm(weights_history[:, 1:], axis=1)  # исключаем bias
    
    plt.subplot(1, 2, 1)
    plt.plot(alphas, weights_norms)
    plt.xscale('log')
    plt.xlabel('Коэффициент регуляризации (alpha)')
    plt.ylabel('L2 норма весов')
    plt.title('L2 норма весов vs alpha')
    plt.grid(True, alpha=0.3)
    
    # MSE на тестовой выборке
    test_errors = []
    for i, alpha in enumerate(alphas):
        gd = RegularizedGradientDescent(learning_rate=0.01, max_iter=1000, alpha=alpha)
        gd.fit(X_train_scaled, y_train)
        test_error = gd.compute_loss(X_test_scaled, y_test, gd.weights)
        test_errors.append(test_error)
    
    plt.subplot(1, 2, 2)
    plt.plot(alphas, test_errors)
    plt.xscale('log')
    plt.xlabel('Коэффициент регуляризации (alpha)')
    plt.ylabel('MSE на тестовой выборке')
    plt.title('Качество модели vs alpha')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Эксперимент завершен ===")

main()