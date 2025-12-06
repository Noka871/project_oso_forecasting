import logging
import os
from datetime import datetime


def setup_logger(name='oso_forecasting', log_level=logging.INFO):
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join('logs', f'app_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger