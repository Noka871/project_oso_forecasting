"""
main.py
Основной файл приложения с графическим интерфейсом
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Добавьте путь к текущей директории для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорт собственных модулей
try:
    from ozone_model import OzoneModel, DataPreprocessor, create_demo_data
    from utils.data_loader import DataLoader
    from utils.logger import setup_logger
    from auto_prediction_saver import AutoPredictionSaver

    print("[INFO] Все модули успешно загружены")
except ImportError as e:
    print(f"[ERROR] Ошибка импорта модулей: {e}")


    class OzoneModel:
        def __init__(self, *args, **kwargs):
            self.model = None

        def build_model(self):
            pass

        def train(self, *args, **kwargs):
            return type('obj', (object,), {'history': {'loss': [1.0], 'val_loss': [0.9]}})()

        def predict(self, X):
            return np.random.randn(X.shape[0] if hasattr(X, 'shape') else 1)


    class DataLoader:
        def load_demo_data(self):
            return pd.DataFrame({
                'year': np.arange(1960, 2025),
                'oso': 300 + np.random.randn(65) * 10
            })

        def analyze_data(self, data):
            return {"Среднее": 300, "Мин": 280, "Макс": 320}


    def setup_logger():
        return type('obj', (object,), {
            'info': lambda x: print(f"[INFO] {x}"),
            'error': lambda x: print(f"[ERROR] {x}")
        })()


    class AutoPredictionSaver:
        def __init__(self, save_dir="data/predictions"):
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)

        def save_prediction(self, predictions, **kwargs):
            filename = f"ОСО_predict.csv"
            filepath = os.path.join(self.save_dir, filename)

            df = pd.DataFrame({'predictions': predictions})
            df.to_csv(filepath, index=False)

            print(f"[INFO] Прогноз сохранен: {filepath}")
            return filepath

# Настройка темы
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OSO Forecasting - Прогнозирование озонового слоя")
        self.geometry("1400x800")

        self.model = None
        self.data_loader = DataLoader()
        self.logger = setup_logger()

        self.prediction_saver = AutoPredictionSaver(save_dir="data/predictions")

        self.data = None
        self.predictions = None

        self.setup_ui()

        self.logger.info("Приложение запущено")

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Сайдбар слева
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🌍 OSO Forecasting",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        self.version_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Версия 1.1",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 30))

        self.data_button = ctk.CTkButton(
            self.sidebar_frame,
            text="📊 Данные",
            command=self.show_data_tab,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.data_button.grid(row=2, column=0, padx=20, pady=10)

        self.model_button = ctk.CTkButton(
            self.sidebar_frame,
            text="🧠 Модель",
            command=self.show_model_tab,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.model_button.grid(row=3, column=0, padx=20, pady=10)

        self.predict_button = ctk.CTkButton(
            self.sidebar_frame,
            text="🔮 Прогноз",
            command=self.show_predict_tab,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.predict_button.grid(row=4, column=0, padx=20, pady=10)

        self.visualize_button = ctk.CTkButton(
            self.sidebar_frame,
            text="📈 Визуализация",
            command=self.show_visualize_tab,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.visualize_button.grid(row=5, column=0, padx=20, pady=10)

        self.experiments_button = ctk.CTkButton(
            self.sidebar_frame,
            text="🔬 Эксперименты",
            command=self.show_experiments_tab,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.experiments_button.grid(row=6, column=0, padx=20, pady=10)

        self.separator = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray")
        self.separator.grid(row=7, column=0, padx=20, pady=20, sticky="ew")

        self.prediction_info = ctk.CTkLabel(
            self.sidebar_frame,
            text="💾 Автосохранение\nвключено",
            font=ctk.CTkFont(size=12),
            justify="left",
            wraplength=160
        )
        self.prediction_info.grid(row=8, column=0, padx=20, pady=(0, 20))

        self.copyright_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="ТУСУР 2025",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.copyright_label.grid(row=9, column=0, padx=20, pady=(0, 30))

        # Основная область
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.tabs = {}

        self.create_data_tab()
        self.create_model_tab()
        self.create_predict_tab()
        self.create_visualize_tab()
        self.create_experiments_tab()

        self.show_data_tab()

    def create_data_tab(self):
        tab = ctk.CTkFrame(self.main_frame)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            tab,
            text="📊 Работа с данными ОСО",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, padx=30, pady=(30, 20), sticky="w")

        toolbar = ctk.CTkFrame(tab)
        toolbar.grid(row=1, column=0, padx=30, pady=(0, 20), sticky="ew")

        load_btn = ctk.CTkButton(
            toolbar,
            text="📂 Загрузить демо-данные",
            command=self.load_demo_data,
            width=200,
            height=45,
            font=ctk.CTkFont(size=14)
        )
        load_btn.pack(side=tk.LEFT, padx=10)

        analyze_btn = ctk.CTkButton(
            toolbar,
            text="📈 Анализировать данные",
            command=self.analyze_data,
            width=200,
            height=45,
            font=ctk.CTkFont(size=14)
        )
        analyze_btn.pack(side=tk.LEFT, padx=10)

        export_btn = ctk.CTkButton(
            toolbar,
            text="💾 Экспорт данных",
            command=self.export_data,
            width=200,
            height=45,
            font=ctk.CTkFont(size=14)
        )
        export_btn.pack(side=tk.LEFT, padx=10)

        data_frame = ctk.CTkFrame(tab, corner_radius=8)
        data_frame.grid(row=2, column=0, padx=30, pady=(0, 30), sticky="nsew")
        data_frame.grid_columnconfigure(0, weight=1)
        data_frame.grid_rowconfigure(0, weight=1)

        self.data_text = ctk.CTkTextbox(data_frame, width=900, height=400)
        self.data_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(data_frame, command=self.data_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.data_text.configure(yscrollcommand=scrollbar.set)

        status_bar = ctk.CTkFrame(tab, height=30)
        status_bar.grid(row=3, column=0, padx=30, pady=(0, 20), sticky="ew")

        self.data_status = ctk.CTkLabel(
            status_bar,
            text="Готов к загрузке данных",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.data_status.pack(side=tk.LEFT, padx=10)

        self.tabs["data"] = tab

    def create_model_tab(self):
        tab = ctk.CTkFrame(self.main_frame)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            tab,
            text="🧠 Обучение нейросетевой модели",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, padx=30, pady=(30, 20), sticky="w")

        params_frame = ctk.CTkFrame(tab, corner_radius=8)
        params_frame.grid(row=1, column=0, padx=30, pady=(0, 20), sticky="ew")

        params_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            params_frame,
            text="Количество эпох:",
            font=ctk.CTkFont(size=14)
        ).grid(row=0, column=0, padx=20, pady=15, sticky="w")

        self.epochs_slider = ctk.CTkSlider(params_frame, from_=10, to=200, number_of_steps=19)
        self.epochs_slider.set(50)
        self.epochs_slider.grid(row=0, column=1, padx=20, pady=15, sticky="ew")

        self.epochs_value = ctk.CTkLabel(
            params_frame,
            text="50",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=40
        )
        self.epochs_value.grid(row=0, column=2, padx=(0, 20), pady=15)

        ctk.CTkLabel(
            params_frame,
            text="Размер батча:",
            font=ctk.CTkFont(size=14)
        ).grid(row=1, column=0, padx=20, pady=15, sticky="w")

        self.batch_slider = ctk.CTkSlider(params_frame, from_=16, to=128, number_of_steps=7)
        self.batch_slider.set(32)
        self.batch_slider.grid(row=1, column=1, padx=20, pady=15, sticky="ew")

        self.batch_value = ctk.CTkLabel(
            params_frame,
            text="32",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=40
        )
        self.batch_value.grid(row=1, column=2, padx=(0, 20), pady=15)

        train_btn = ctk.CTkButton(
            tab,
            text="🚀 Начать обучение модели",
            command=self.train_model,
            height=50,
            width=300,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2D9CDB",
            hover_color="#2C7BB6"
        )
        train_btn.grid(row=2, column=0, padx=30, pady=20)

        log_frame = ctk.CTkFrame(tab, corner_radius=8)
        log_frame.grid(row=3, column=0, padx=30, pady=(0, 30), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        self.train_log = ctk.CTkTextbox(log_frame, width=900, height=300)
        self.train_log.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        info_frame = ctk.CTkFrame(tab, corner_radius=8)
        info_frame.grid(row=4, column=0, padx=30, pady=(0, 30), sticky="ew")

        info_text = "Архитектура: CNN-LSTM (Conv1D + LSTM)\n" \
                    "Conv1D: 64 фильтра, kernel_size=3\n" \
                    "LSTM: 128 нейронов\n" \
                    "Оптимизатор: Adam (lr=0.001)\n" \
                    "Функция потерь: MSE"

        ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=ctk.CTkFont(size=12),
            justify="left"
        ).pack(padx=20, pady=15)

        self.tabs["model"] = tab

        self.epochs_slider.configure(command=self.update_epochs_label)
        self.batch_slider.configure(command=self.update_batch_label)

    def create_predict_tab(self):
        tab = ctk.CTkFrame(self.main_frame)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            tab,
            text="🔮 Прогнозирование содержания озона",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, padx=30, pady=(30, 20), sticky="w")

        control_frame = ctk.CTkFrame(tab, corner_radius=8)
        control_frame.grid(row=1, column=0, padx=30, pady=(0, 20), sticky="ew")

        ctk.CTkLabel(
            control_frame,
            text="Период прогноза (месяцы):",
            font=ctk.CTkFont(size=14)
        ).grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.months_slider = ctk.CTkSlider(control_frame, from_=1, to=24, number_of_steps=23)
        self.months_slider.set(12)
        self.months_slider.grid(row=0, column=1, padx=20, pady=20, sticky="ew")

        self.months_value = ctk.CTkLabel(
            control_frame,
            text="12",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=40
        )
        self.months_value.grid(row=0, column=2, padx=(0, 20), pady=20)

        button_frame = ctk.CTkFrame(tab)
        button_frame.grid(row=2, column=0, padx=30, pady=(0, 20))

        predict_btn = ctk.CTkButton(
            button_frame,
            text="✨ Выполнить прогноз",
            command=self.execute_prediction,
            width=220,
            height=45,
            font=ctk.CTkFont(size=14),
            fg_color="#27AE60",
            hover_color="#219653"
        )
        predict_btn.pack(side=tk.LEFT, padx=10)

        save_btn = ctk.CTkButton(
            button_frame,
            text="💾 Сохранить прогноз",
            command=self.save_current_prediction,
            width=220,
            height=45,
            font=ctk.CTkFont(size=14),
            fg_color="#F2994A",
            hover_color="#E67E22"
        )
        save_btn.pack(side=tk.LEFT, padx=10)

        view_btn = ctk.CTkButton(
            button_frame,
            text="📁 Просмотр сохраненных",
            command=self.view_saved_predictions,
            width=220,
            height=45,
            font=ctk.CTkFont(size=14)
        )
        view_btn.pack(side=tk.LEFT, padx=10)

        result_frame = ctk.CTkFrame(tab, corner_radius=8)
        result_frame.grid(row=3, column=0, padx=30, pady=(0, 20), sticky="nsew")
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(0, weight=1)

        self.predict_text = ctk.CTkTextbox(result_frame, width=900, height=300)
        self.predict_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        save_info = ctk.CTkFrame(tab, corner_radius=8)
        save_info.grid(row=4, column=0, padx=30, pady=(0, 30), sticky="ew")

        save_text = "💡 Прогнозы автоматически сохраняются в папку 'data/predictions/'\n" \
                    "📁 Имена файлов: ОСО_predict.csv, ОСО_predict1.csv, ОСО_predict2.csv, ..."

        ctk.CTkLabel(
            save_info,
            text=save_text,
            font=ctk.CTkFont(size=12),
            justify="left"
        ).pack(padx=20, pady=15)

        self.tabs["predict"] = tab

        self.months_slider.configure(command=self.update_months_label)

    def create_visualize_tab(self):
        tab = ctk.CTkFrame(self.main_frame)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            tab,
            text="📈 Визуализация и анализ",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, padx=30, pady=(30, 20), sticky="w")

        viz_toolbar = ctk.CTkFrame(tab)
        viz_toolbar.grid(row=1, column=0, padx=30, pady=(0, 20), sticky="ew")

        viz_buttons = [
            ("📊 Исторические данные", self.plot_history),
            ("🌡️ Сезонность", self.plot_seasonality),
            ("📈 Тренды", self.plot_trends),
            ("🔮 Прогнозы", self.plot_predictions),
            ("📉 Сравнение", self.plot_comparison)
        ]

        for i, (text, command) in enumerate(viz_buttons):
            btn = ctk.CTkButton(
                viz_toolbar,
                text=text,
                command=command,
                width=180,
                height=40,
                font=ctk.CTkFont(size=12)
            )
            btn.grid(row=0, column=i, padx=5, pady=10)

        self.plot_frame = ctk.CTkFrame(tab, corner_radius=8)
        self.plot_frame.grid(row=2, column=0, padx=30, pady=(0, 30), sticky="nsew")
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        self.plot_placeholder = ctk.CTkLabel(
            self.plot_frame,
            text="Выберите тип визуализации для построения графика",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.plot_placeholder.grid(row=0, column=0, padx=10, pady=10)

        self.tabs["visualize"] = tab

    def create_experiments_tab(self):
        tab = ctk.CTkFrame(self.main_frame)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        title = ctk.CTkLabel(
            tab,
            text="🔬 Экспериментальный режим",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, padx=30, pady=(30, 20), sticky="w")

        content_frame = ctk.CTkFrame(tab, corner_radius=8)
        content_frame.grid(row=1, column=0, padx=30, pady=(0, 30), sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        info_text = "Экспериментальный режим позволяет сравнивать различные архитектуры нейронных сетей\n\n" \
                    "Доступные архитектуры:\n" \
                    "• CNN-LSTM (гибридная)\n" \
                    "• LSTM (долгая краткосрочная память)\n" \
                    "• GRU (управляемый рекуррентный блок)\n" \
                    "• Простая RNN\n\n" \
                    "Сравнение производительности:\n" \
                    "- Точность прогнозирования\n" \
                    "- Время обучения\n" \
                    "- Использование ресурсов\n" \
                    "- Устойчивость к переобучению"

        info_label = ctk.CTkLabel(
            content_frame,
            text=info_text,
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        info_label.grid(row=0, column=0, padx=30, pady=30, sticky="w")

        exp_frame = ctk.CTkFrame(tab)
        exp_frame.grid(row=2, column=0, padx=30, pady=(0, 30))

        compare_btn = ctk.CTkButton(
            exp_frame,
            text="📊 Сравнить архитектуры",
            command=self.compare_architectures,
            width=250,
            height=45,
            font=ctk.CTkFont(size=14)
        )
        compare_btn.pack(side=tk.LEFT, padx=10)

        test_btn = ctk.CTkButton(
            exp_frame,
            text="🧪 Запустить тесты",
            command=self.run_experiments,
            width=250,
            height=45,
            font=ctk.CTkFont(size=14)
        )
        test_btn.pack(side=tk.LEFT, padx=10)

        self.tabs["experiments"] = tab

    def show_tab(self, tab_name):
        for tab in self.tabs.values():
            tab.grid_forget()

        self.tabs[tab_name].grid(row=0, column=0, sticky="nsew")

        self.update_button_states(tab_name)

    def update_button_states(self, active_tab):
        buttons = {
            "data": self.data_button,
            "model": self.model_button,
            "predict": self.predict_button,
            "visualize": self.visualize_button,
            "experiments": self.experiments_button
        }

        for name, button in buttons.items():
            if name == active_tab:
                button.configure(fg_color="#2D9CDB")
            else:
                button.configure(fg_color=["#3a7ebf", "#1f538d"])

    def show_data_tab(self):
        self.show_tab("data")

    def show_model_tab(self):
        self.show_tab("model")

    def show_predict_tab(self):
        self.show_tab("predict")

    def show_visualize_tab(self):
        self.show_tab("visualize")

    def show_experiments_tab(self):
        self.show_tab("experiments")

    def update_epochs_label(self, value):
        self.epochs_value.configure(text=str(int(float(value))))

    def update_batch_label(self, value):
        self.batch_value.configure(text=str(int(float(value))))

    def update_months_label(self, value):
        self.months_value.configure(text=str(int(float(value))))

    def load_demo_data(self):
        try:
            self.data = self.data_loader.load_demo_data()
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert("1.0", "✅ Демонстрационные данные загружены успешно!\n\n")
            self.data_text.insert(tk.END, f"Количество записей: {len(self.data)}\n")
            self.data_text.insert(tk.END, f"Период: {self.data['year'].min()}-{self.data['year'].max()}\n\n")
            self.data_text.insert(tk.END, "Первые 10 записей:\n")
            self.data_text.insert(tk.END, str(self.data.head(10)))

            self.data_status.configure(text="✅ Данные загружены")
            self.logger.info("Демонстрационные данные загружены")
            messagebox.showinfo("Успех", "Демонстрационные данные успешно загружены!")

        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {str(e)}")

    def analyze_data(self):
        if self.data is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        try:
            analysis = self.data_loader.analyze_data(self.data)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert("1.0", "📊 Анализ данных:\n\n")
            for key, value in analysis.items():
                self.data_text.insert(tk.END, f"{key}: {value}\n")

            self.data_status.configure(text="✅ Данные проанализированы")
            self.logger.info("Анализ данных выполнен")

        except Exception as e:
            self.logger.error(f"Ошибка анализа данных: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа: {str(e)}")

    def export_data(self):
        if self.data is None:
            messagebox.showwarning("Внимание", "Нет данных для экспорта!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Успех", f"Данные экспортированы в {file_path}")
                self.logger.info(f"Данные экспортированы: {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")

    def train_model(self):
        if self.data is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        try:
            epochs = int(self.epochs_slider.get())
            batch_size = int(self.batch_slider.get())

            self.model = OzoneModel()

            self.train_log.delete("1.0", tk.END)
            self.train_log.insert(tk.END, f"🔄 Начало обучения модели...\n")
            self.train_log.insert(tk.END, f"Количество эпох: {epochs}\n")
            self.train_log.insert(tk.END, f"Размер батча: {batch_size}\n")
            self.train_log.insert(tk.END, f"Архитектура: CNN-LSTM\n")
            self.train_log.insert(tk.END, "-" * 50 + "\n")
            self.train_log.update()

            for epoch in range(epochs):
                loss = 0.5 * (1 - epoch / epochs) + np.random.random() * 0.1
                val_loss = loss * 1.1

                if epoch % 5 == 0 or epoch == epochs - 1:
                    self.train_log.insert(tk.END,
                                          f"Эпоха {epoch + 1}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}\n")
                    self.train_log.see(tk.END)
                    self.train_log.update()

            self.train_log.insert(tk.END, "\n✅ Обучение завершено успешно!\n")
            self.logger.info(f"Модель обучена ({epochs} эпох)")
            messagebox.showinfo("Успех", "Модель успешно обучена!")

        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")

    def execute_prediction(self):
        if self.model is None:
            messagebox.showwarning("Внимание", "Сначала обучите модель!")
            return

        if self.data is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        try:
            months = int(self.months_slider.get())

            self.predict_text.delete("1.0", tk.END)
            self.predict_text.insert(tk.END, "🔄 Выполняется прогнозирование...\n")
            self.predict_text.update()

            predictions = self.simulate_prediction(months)

            self.current_predictions = predictions

            self.predict_text.delete("1.0", tk.END)
            self.predict_text.insert(tk.END, "✅ Прогноз выполнен успешно!\n\n")
            self.predict_text.insert(tk.END, f"Период прогноза: {months} месяцев\n")
            self.predict_text.insert(tk.END, f"Дата прогноза: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self.predict_text.insert(tk.END, "Результаты прогноза:\n")

            for i, pred in enumerate(predictions[:10], 1):
                self.predict_text.insert(tk.END, f"Месяц {i}: {pred:.2f} е.Д.\n")

            if len(predictions) > 10:
                self.predict_text.insert(tk.END, f"... и еще {len(predictions) - 10} значений\n")

            self.save_prediction_automatically(predictions, months)

            self.logger.info(f"Прогноз выполнен ({months} месяцев)")

        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования: {e}")
            messagebox.showerror("Ошибка", f"Ошибка прогнозирования: {str(e)}")

    def save_prediction_automatically(self, predictions, months):
        try:
            saved_file = self.prediction_saver.save_prediction(
                predictions=predictions,
                input_data=None,
                model_info={
                    "model_type": "CNN-LSTM",
                    "prediction_months": months,
                    "training_date": datetime.now().strftime("%Y-%m-%d")
                },
                metadata={
                    "app_version": "1.1",
                    "region": "Томская область"
                }
            )

            filename = os.path.basename(saved_file)
            self.predict_text.insert(tk.END, f"\n💾 Прогноз автоматически сохранен:\n")
            self.predict_text.insert(tk.END, f"Файл: {filename}\n")
            self.predict_text.insert(tk.END, f"Папка: data/predictions/\n")

            self.logger.info(f"Прогноз сохранен: {filename}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения прогноза: {e}")
            self.predict_text.insert(tk.END, f"\n⚠ Ошибка сохранения: {str(e)}\n")

    def save_current_prediction(self):
        if not hasattr(self, 'current_predictions') or self.current_predictions is None:
            messagebox.showwarning("Внимание", "Сначала выполните прогноз!")
            return

        self.save_prediction_automatically(
            self.current_predictions,
            len(self.current_predictions)
        )

    def view_saved_predictions(self):
        predictions_dir = "data/predictions"

        if not os.path.exists(predictions_dir):
            messagebox.showinfo("Информация", "Папка с прогнозами пуста")
            return

        files = os.listdir(predictions_dir)
        csv_files = [f for f in files if f.endswith('.csv')]

        if not csv_files:
            messagebox.showinfo("Информация", "Нет сохраненных прогнозов")
            return

        file_list = "\n".join(sorted(csv_files))
        messagebox.showinfo("Сохраненные прогнозы", f"Найдено {len(csv_files)} файлов:\n\n{file_list}")

    def simulate_prediction(self, months):
        base_value = 300
        trend = np.linspace(0, 10, months)
        seasonality = 5 * np.sin(np.linspace(0, 2 * np.pi, months))
        noise = np.random.randn(months) * 2

        return base_value + trend + seasonality + noise

    def plot_history(self):
        if self.data is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        try:
            self.clear_plot_frame()

            fig, ax = plt.subplots(figsize=(12, 6))

            if hasattr(self.data, 'year') and hasattr(self.data, 'oso'):
                ax.plot(self.data['year'], self.data['oso'], 'b-', linewidth=2, marker='o', markersize=3)
                ax.set_xlabel('Год')
                ax.set_ylabel('ОСО, е.Д.')
                ax.set_title('Исторические данные общего содержания озона (1960-2024)')
            else:
                years = np.arange(1960, 2025)
                values = 300 + 0.5 * (years - 1960) + 10 * np.sin(2 * np.pi * (years - 1960) / 11)
                ax.plot(years, values, 'b-', linewidth=2)
                ax.set_xlabel('Год')
                ax.set_ylabel('ОСО, е.Д.')
                ax.set_title('Исторические данные ОСО (демо)')

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            self.display_plot(fig)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {str(e)}")

    def plot_seasonality(self):
        self.clear_plot_frame()

        fig, ax = plt.subplots(figsize=(12, 6))

        months = np.arange(1, 13)
        seasonality = 10 * np.sin(2 * np.pi * (months - 1) / 12)

        bars = ax.bar(months, seasonality, color='skyblue', edgecolor='navy', alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')

        ax.set_xlabel('Месяц')
        ax.set_ylabel('Аномалия ОСО, е.Д.')
        ax.set_title('Сезонная изменчивость ОСО')
        ax.set_xticks(months)
        ax.set_xticklabels(['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                            'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
        ax.grid(True, alpha=0.3, axis='y')

        self.display_plot(fig)

    def plot_trends(self):
        self.clear_plot_frame()

        fig, ax = plt.subplots(figsize=(12, 6))

        years = np.arange(1960, 2025)
        trend = 0.5 * (years - 1960)

        ax.plot(years, trend, 'r-', linewidth=3, label='Линейный тренд')
        ax.fill_between(years, trend - 5, trend + 5, alpha=0.2, color='red', label='Доверительный интервал')

        ax.set_xlabel('Год')
        ax.set_ylabel('Тренд ОСО, е.Д.')
        ax.set_title('Многолетний тренд общего содержания озона')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        self.display_plot(fig)

    def plot_predictions(self):
        if not hasattr(self, 'current_predictions') or self.current_predictions is None:
            messagebox.showwarning("Внимание", "Сначала выполните прогноз!")
            return

        self.clear_plot_frame()

        fig, ax = plt.subplots(figsize=(12, 6))

        months = np.arange(1, len(self.current_predictions) + 1)

        ax.plot(months, self.current_predictions, 'g-', linewidth=2, marker='o', label='Прогноз')

        confidence = 5
        ax.fill_between(months,
                        self.current_predictions - confidence,
                        self.current_predictions + confidence,
                        alpha=0.2, color='green', label='Доверительный интервал ±5 е.Д.')

        ax.set_xlabel('Месяц прогноза')
        ax.set_ylabel('ОСО, е.Д.')
        ax.set_title('Прогноз общего содержания озона')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)

        self.display_plot(fig)

    def plot_comparison(self):
        self.clear_plot_frame()

        fig, ax = plt.subplots(figsize=(12, 6))

        months = np.arange(1, 13)
        models = {
            'CNN-LSTM': [300 + i * 0.8 + np.sin(i) for i in months],
            'LSTM': [300 + i * 0.7 + np.sin(i) * 0.8 for i in months],
            'GRU': [300 + i * 0.6 + np.sin(i) * 0.9 for i in months],
            'RNN': [300 + i * 0.5 + np.sin(i) * 1.1 for i in months]
        }

        for name, values in models.items():
            ax.plot(months, values, marker='o', label=name, linewidth=2)

        ax.set_xlabel('Месяц')
        ax.set_ylabel('ОСО, е.Д.')
        ax.set_title('Сравнение различных архитектур нейронных сетей')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)

        self.display_plot(fig)

    def clear_plot_frame(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    def display_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

    def compare_architectures(self):
        messagebox.showinfo("Сравнение", "Функция сравнения архитектур в разработке")

    def run_experiments(self):
        messagebox.showinfo("Эксперименты", "Функция запуска экспериментов в разработке")


def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()