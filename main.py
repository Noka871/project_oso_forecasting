# main.py
import argparse
import logging
from src.data_loader import DataLoader
from src.model import create_model
from src.trainer import train_model, evaluate_model
from src.predictor import make_predictions, load_trained_model

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Прогнозирование временных рядов ОСО')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                        help='Режим работы: train, predict или full')
    args = parser.parse_args()

    # Инициализация загрузчика данных
    data_loader = DataLoader()

    if args.mode in ['train', 'full']:
        logger.info("=== РЕЖИМ ОБУЧЕНИЯ ===")

        # Загрузка и подготовка данных
        if not data_loader.load_data() or not data_loader.prepare_data():
            logger.error("Не удалось подготовить данные для обучения!")
            return

        # Создание и обучение модели
        model = create_model()
        X_train, y_train = data_loader.get_train_data()
        X_test, y_test = data_loader.get_test_data()

        history = train_model(model, X_train, y_train, X_test, y_test)

        # Оценка модели
        evaluate_model(model, X_test, y_test)

    if args.mode in ['predict', 'full']:
        logger.info("=== РЕЖИМ ПРОГНОЗИРОВАНИЯ ===")

        # Загрузка данных если еще не загружены
        if data_loader.data is None:
            if not data_loader.load_data() or not data_loader.prepare_data():
                logger.error("Не удалось подготовить данные для прогнозирования!")
                return

        # Загрузка модели и выполнение прогнозов
        model = load_trained_model()
        if model is None:
            logger.error("Не удалось загрузить модель для прогнозирования!")
            return

        results = make_predictions(model, data_loader)
        if results is not None:
            logger.info("Первые 5 прогнозов:")
            print(results.head())

    logger.info("Программа завершена!")


if __name__ == "__main__":
    main()
