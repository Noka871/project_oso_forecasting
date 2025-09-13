"""
Модуль для обучения нейронной сети
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.history = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Обучение модели
        """
        print("Начало обучения модели...")

        self.history = self.model.fit(
            X_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1,
            shuffle=False  # Для временных рядов не перемешиваем данные
        )

        print("Обучение завершено!")
        return self.history

    def plot_training_history(self, filename='results/training_plot.png'):
        """
        Визуализация процесса обучения
        """
        if self.history is None:
            print("История обучения отсутствует")
            return

        plt.style.use(config.PLOT_STYLE)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.FIGURE_SIZE)

        # График потерь
        ax1.plot(self.history.history['loss'], label='Обучающая выборка')
        ax1.plot(self.history.history['val_loss'], label='Валидационная выборка')
        ax1.set_title('Функция потерь (MSE)')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Потери')
        ax1.legend()
        ax1.grid(True)

        # График MAE
        ax2.plot(self.history.history['mae'], label='Обучающая выборка')
        ax2.plot(self.history.history['val_mae'], label='Валидационная выборка')
        ax2.set_title('Средняя абсолютная ошибка (MAE)')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"График обучения сохранен в {filename}")

    def evaluate_model(self, X_test, y_test):
        """
        Оценка качества модели на тестовых данных
        """
        predictions = self.model.predict(X_test)

        # Расчет метрик
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
        print("=" * 50)
        print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
        print(f"Средняя квадратичная ошибка (MSE): {mse:.4f}")
        print(f"Корень из MSE (RMSE): {rmse:.4f}")
        print(f"Коэффициент детерминации (R²): {r2:.4f}")
        print("=" * 50)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }