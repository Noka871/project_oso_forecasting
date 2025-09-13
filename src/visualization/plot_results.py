import matplotlib.pyplot as plt
import os
from src.config import Config


def plot_predictions(y_true, y_pred, years):
    """
    Визуализация реальных и предсказанных значений
    """
    plt.figure(figsize=(12, 6))
    plt.plot(years, y_true, label="Реальные значения", marker='o')
    plt.plot(years, y_pred, label="Предсказанные", marker='x')

    plt.title("Прогнозирование общего содержания озона")
    plt.xlabel("Год")
    plt.ylabel("ОСО (стандартизированные значения)")
    plt.legend()
    plt.grid(True)

    # Сохранение графика
    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/predictions.png")
    plt.close()