import numpy as np
import pandas as pd
from src.data_processing.load_data import load_data
from src.data_processing.preprocess import prepare_time_series
from src.data_processing.split_data import split_data
from src.model.build_model import build_lstm_model
from src.model.train import train_model
from src.model.evaluate import evaluate_model
from src.visualization.plot_results import plot_predictions
from src.config import Config


def main():
    try:
        print("1. Загрузка данных...")
        data = load_data()

        print("2. Подготовка временных рядов...")
        X, y, x_scaler, y_scaler = prepare_time_series(data)

        print("3. Разделение данных...")
        X_train, X_test, y_train, y_test = split_data(X, y)

        print("4. Построение модели...")
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        print("5. Обучение модели...")
        history = train_model(model, X_train, y_train)

        print("6. Оценка модели...")
        metrics, y_test_orig, y_pred = evaluate_model(model, X_test, y_test, y_scaler)

        print("\nМетрики модели:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("7. Визуализация результатов...")
        # Получаем годы для тестовой выборки
        test_years = data[Config.TIME_COLUMN].values[-len(y_test_orig):]
        plot_predictions(y_test_orig.flatten(), y_pred.flatten(), test_years)

        print("8. Сохранение прогнозов...")
        predictions_df = pd.DataFrame({
            "Year": test_years,
            "Actual": y_test_orig.flatten(),
            "Predicted": y_pred.flatten()
        })
        predictions_df.to_csv(Config.PREDICTIONS_PATH, index=False)

        print("Готово! Результаты сохранены в папке results.")

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()