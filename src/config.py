import os
from pathlib import Path

# Базовый путь к проекту
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # Пути к данным (теперь абсолютные)
    DATA_PATH = os.path.join(BASE_DIR, "data", "ОСО_индекс_12.dat")
    PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "ОСО_predict.dat")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.h5")

    # Остальные настройки остаются без изменений...
    TIME_COLUMN = "TO"
    TARGET_COLUMN = "OCO_среднее_годичное"
    FEATURES = [
        "OCO_январь", "OCO_февраль", "OCO_март", "OCO_апрель",
        "OCO_май", "OCO_июнь", "OCO_июль", "OCO_август",
        "OCO_сентябрь", "OCO_октябрь", "OCO_ноябрь", "OCO_декабрь"
    ]
    # ... остальной конфиг