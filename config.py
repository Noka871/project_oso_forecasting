"""
Конфигурационные параметры проекта прогнозирования ОСО
"""


class Config:
    # Пути к файлам
    DATA_PATH = 'data/ОСО_индекс_12.dat'
    OUTPUT_PATH = 'data/ОСО_predict.dat'

    # Параметры данных
    FEATURE_COLUMNS = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    TARGET_COLUMN = 'среднее'

    # Параметры нейронной сети
    HIDDEN_LAYERS = [24, 12]  # Количество нейронов в скрытых слоях
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'linear'
    OPTIMIZER = 'adam'
    LOSS = 'mse'
    METRICS = ['mae', 'mse']

    # Параметры обучения
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.2
    EPOCHS = 200
    BATCH_SIZE = 8
    RANDOM_STATE = 42

    # Параметры визуализации
    PLOT_STYLE = 'ggplot'
    FIGURE_SIZE = (12, 6)