# main.py
import argparse
import os
import sys
import logging  # Добавляем этот импорт
from src.logger_config import main_logger, data_logger, model_logger, training_logger
from src.data_loader import DataLoader
from src.model import create_model
from src.trainer import train_model, evaluate_model
from src.predictor import make_predictions, load_trained_model


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Прогнозирование временных рядов ОСО')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                        help='Режим работы: train, predict или full')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Уровень логирования')
    args = parser.parse_args()

    # Устанавливаем уровень логирования
    log_level = getattr(logging, args.log_level)
    main_logger.setLevel(log_level)
    data_logger.setLevel(log_level)
    model_logger.setLevel(log_level)
    training_logger.setLevel(log_level)

    main_logger.info("=" * 60)
    main_logger.info("ЗАПУСК ПРОГРАММЫ ПРОГНОЗИРОВАНИЯ ОСО")
    main_logger.info("=" * 60)
    main_logger.info(f"Режим работы: {args.mode}")
    main_logger.info(f"Уровень логирования: {args.log_level}")

    try:
        # Инициализация загрузчика данных
        data_logger.info("Инициализация DataLoader...")
        data_loader = DataLoader()

        if args.mode in ['train', 'full']:
            main_logger.info("=== ЗАПУСК РЕЖИМА ОБУЧЕНИЯ ===")

            # Загрузка и подготовка данных
            data_logger.info("Подготовка данных для обучения...")
            if not data_loader.prepare_data():
                main_logger.error("Не удалось подготовить данные для обучения!")
                return 1

            # Создание и обучение модели
            model_logger.info("Создание модели...")
            model = create_model()

            X_train, y_train = data_loader.get_train_data()
            X_test, y_test = data_loader.get_test_data()

            training_logger.info("Начало обучения модели...")
            history = train_model(model, X_train, y_train, X_test, y_test)

            if history is not None:
                training_logger.info("Оценка обученной модели...")
                test_results = evaluate_model(model, X_test, y_test)
                if test_results:
                    training_logger.info("Результаты обучения успешны!")

        if args.mode in ['predict', 'full']:
            main_logger.info("=== ЗАПУСК РЕЖИМА ПРОГНОЗИРОВАНИЯ ===")

            # Загрузка данных если еще не загружены
            if data_loader.data is None:
                data_logger.info("Загрузка данных для прогнозирования...")
                if not data_loader.prepare_data():
                    main_logger.error("Не удалось подготовить данные для прогнозирования!")
                    return 1

            # Загрузка модели
            model_logger.info("Загрузка обученной модели...")
            model = load_trained_model()
            if model is None:
                main_logger.error("Не удалось загрузить модель для прогнозирования!")
                return 1

            # Выполнение прогнозов
            data_logger.info("Выполнение прогнозов...")
            results = make_predictions(model, data_loader)
            if results is not None:
                main_logger.info("Прогнозы успешно выполнены и сохранены!")
                main_logger.info(f"Размер результатов: {results.shape}")
                main_logger.info("Первые 5 прогнозов:")
                for i, (actual, predicted) in enumerate(zip(results['Actual'].head(), results['Predicted'].head())):
                    main_logger.info(f"  {i + 1}. Actual: {actual:.4f}, Predicted: {predicted:.4f}")

        main_logger.info("Программа успешно завершена!")
        return 0

    except Exception as e:
        main_logger.critical(f"Критическая ошибка в main: {e}")
        main_logger.exception("Трассировка стека:")
        return 1


if __name__ == "__main__":
    # Создаем папку для логов
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Запускаем программу
    exit_code = main()
    sys.exit(exit_code)