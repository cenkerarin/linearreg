import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class LinearRegression:
    """
    NumPy kullanarak Linear Regression denemesi
    Optimizasyon için gradient descent kullanılıyor videoda bu kısımdan bahsetmedim ekstra bir video çekeceğim gradient descent ile ilgili
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Modeli başlatır.

        Parametreler:
            learning_rate: Gradient descent için learning rate değeri
            n_iterations: Gradient descent için iterasyon sayısı
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Modeli gradient descent kullanarak eğitir.

        Parametreler:
            X: Training features (m x n matrix)
            y: Training targets (m x 1 vektör)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []

        for i in range(self.n_iterations):
            y_predicted = self.predict(X)

            cost = self._compute_cost(y, y_predicted)
            self.cost_history.append(cost)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Eğitilmiş modeli kullanarak prediction yapar.

        Parametreler:
            X: Input features

        Dönüş:
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias

    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error cost function hesaplar.

        Parametreler:
            y_true: Actual values
            y_pred: Predicted values

        Dönüş:
            Mean squared error değeri
        """
        return np.mean((y_true - y_pred) ** 2)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        R-squared (coefficient of determination) değerini hesaplar.

        Parametreler:
            X: Input features
            y: True values

        Dönüş:
            R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def generate_sample_data(n_samples: int = 100, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear regression için sample data oluşturur.

    Parametreler:
        n_samples: Oluşturulacak sample sayısı
        noise: Eklenecek noise miktarı

    Dönüş:
        X: Feature matrix
        y: Target vector
    """
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X.flatten() + noise * np.random.randn(n_samples)
    return X, y


def plot_results(X: np.ndarray, y: np.ndarray, model: LinearRegression) -> None:
    """
    Parametreler:
        X: Feature matrix
        y: Target vector
        model: Eğitilmiş linear regression modeli
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, color='blue', label='Data points')

    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history, color='green', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost Function Over Time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """
    ana fonksiyon
    """
    print("Linear Regression using NumPy")
    print("=" * 50)

    X, y = generate_sample_data(n_samples=100, noise=0.5)

    print(f"Generated data points: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)

    print("\nTraining the model...")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    final_cost = model.cost_history[-1]

    print("\nModel Results:")
    print(f"Final weight: {model.weights[0]:.4f}")
    print(f"Final bias: {model.bias:.4f}")
    print(f"Final cost (MSE): {final_cost:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    plot_results(X_train, y_train, model)

    print(f"\nExample prediction:")
    example_input = np.array([[1.5]])
    prediction = model.predict(example_input)
    print(f"Input: {example_input[0][0]:.2f} -> Prediction: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()