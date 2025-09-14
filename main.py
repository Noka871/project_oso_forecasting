#!/usr/bin/env python3
"""
Главный скрипт для прогнозирования общего содержания озона (ОСО)
"""

import sys
import os
import logging

# Добавляем папку src в путь импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Основная функция"""
    logger = setup_logging()

    try:
        logger.info("=" * 60)
        logger.info("🌍 ЗАПУСК ПРОГРАММЫ ПРОГНОЗИРОВАНИЯ ОСО")
        logger.info("=" * 60)

        # Проверяем наличие данных
        data_path = "data/ОСО_индекс_12.dat"
        if not os.path.exists(data_path):
            logger.error(f"Файл с данными не найден: {data_path}")
            logger.info("Поместите файл ОСО_индекс_12.dat в папку data/")
            return

        # Импортируем модули
        try:
            from modules.data_loader import DataLoader
            from modules.model import create_model
            from modules.trainer import ModelTrainer
            from modules.predictor import Predictor
            from utils.config import Config

            logger.info("✅ Все модули успешно импортированы")

        except ImportError as e:
            logger.error(f"Ошибка импорта модулей: {e}")
            logger.info("Проверьте структуру проекта и зависимости")
            return

        # Загрузка конфигурации
        try:
            config = Config()
            logger.info("✅ Конфигурация загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return

        # Загрузка данных
        try:
            data_loader = DataLoader(config)
            if data_loader.load_data(data_path):
                logger.info("✅ Данные успешно загружены")
            else:
                logger.error("❌ Не удалось загрузить данные")
                return

        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return

        logger.info("=" * 60)
        logger.info("🚀 ПРОГРАММА ГОТОВА К РАБОТЕ!")
        logger.info("=" * 60)

        # Здесь будет основной код прогнозирования

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()