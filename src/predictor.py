# src/predictor.py
import os
import pandas as pd
from tensorflow.keras.models import load_model
import logging

logger = logging.getLogger(__name__)


def make_predictions(model, data_loader, output_path=None):
    """
    Выполнение прогнозов и сохранение результатов
    """
    if output_path is None:
        # Автоматически определяем путь для сохранения
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(current_dir, "..")
        output_path = os.path.join(base_dir, "data", "raw", "ОСО_predict.dat")

    logger.info("Выполнение прогнозов...")

    try:
        # Получаем тестовые данные
        X_test, y_test = data_loader.get_test_data()

        # Прогнозируем
        y_pred_scaled = model.predict(X_test, verbose=0)

        # Преобразуем обратно в исходный масштаб
        y_test_original = data_loader.inverse_transform_y(y_test)
        y_pred_original = data_loader.inverse_transform_y(y_pred_scaled)

        # Создаем DataFrame с результатами
        results_df = pd.DataFrame({
            'Actual': y_test_original.flatten(),
            'Predicted': y_pred_original.flatten()
        })

        # Сохраняем результаты
        results_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
        logger.info(f"Прогнозы сохранены в {output_path}")

        return results_df

    except Exception as e:
        logger.error(f"Ошибка при прогнозировании: {e}")
        return None


def load_trained_model(model_path=None):
    """Загрузка обученной модели"""
    if model_path is None:
        # Автоматически определяем путь к модели
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(current_dir, "..")
        model_path = os.path.join(base_dir, "models", "best_model.h5")

    try:
        logger.info(f"Загрузка модели из {model_path}")
        model = load_model(model_path)
        logger.info("Модель успешно загружена")
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return None