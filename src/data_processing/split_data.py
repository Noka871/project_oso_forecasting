from sklearn.model_selection import train_test_split
from src.config import Config


def split_data(X, y):
    """
    Разделение данных на обучающую и тестовую выборки
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        shuffle=False  # для временных рядов не перемешиваем!
    )

    return X_train, X_test, y_train, y_test