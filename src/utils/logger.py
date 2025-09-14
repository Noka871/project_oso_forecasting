"""
Логирование для проекта
"""

import logging
from pathlib import Path
import sys


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Настройка логгера"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Консольный handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый handler (если указан)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Глобальный логгер
logger = setup_logger('oso_forecasting', 'logs/app.log')