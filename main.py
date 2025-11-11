import customtkinter as ctk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ozone_model import OzoneHybridModel
import threading
import os
import traceback
from utils.data_loader import OzoneDataLoader
from utils.logger import logger, log_function_call

# Настройка темы
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ModernOzoneApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        logger.info("🚀 Запуск приложения OSO Forecasting")

        # Настройка шрифтов
        self.title_font = ctk.CTkFont(family="Arial", size=20, weight="bold")
        self.subtitle_font = ctk.CTkFont(family="Arial", size=14, weight="bold")
        self.normal_font = ctk.CTkFont(family="Arial", size=12)
        self.small_font = ctk.CTkFont(family="Arial", size=10)

        # Настройка главного окна
        self.title("🌍 OSO Forecasting - Прогнозирование озонового слоя")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Инициализация компонентов
        self.data_loader = OzoneDataLoader()
        self.model = OzoneHybridModel()
        self.oso_data = None
        self.forecast = None
        self.current_step = 0

        # Создание интерфейса
        self.create_sidebar()
        self.create_main_content()
        self.create_status_bar()

        logger.info("✅ Интерфейс приложения инициализирован")

    def create_sidebar(self):
        """Создание боковой панели"""
        logger.debug("Создание боковой панели")

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Заголовок
        title_label = ctk.CTkLabel(
            self.sidebar,
            text="🌍 OSO Forecasting",
            font=self.title_font
        )
        title_label.pack(pady=(30, 10), padx=20)

        # Шаги работы
        steps_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        steps_frame.pack(fill="x", padx=20, pady=10)

        steps = [
            ("📥 Загрузить данные", "Загрузка данных ОСО"),
            ("🧠 Обучить модель", "Обучение нейросети"),
            ("🔮 Выполнить прогноз", "Прогнозирование"),
            ("💾 Сохранить результаты", "Экспорт данных")
        ]

        self.step_buttons = []
        for i, (title, desc) in enumerate(steps):
            step_btn = ctk.CTkButton(
                steps_frame,
                text=f"{i + 1}. {title}",
                font=self.normal_font,
                height=50,
                anchor="w",
                command=lambda idx=i: self.set_current_step(idx),
                state="disabled" if i > 0 else "normal"
            )
            step_btn.pack(fill="x", pady=5)
            self.step_buttons.append(step_btn)

        self.step_buttons[0].configure(fg_color="#2E8B57")

    def create_main_content(self):
        """Создание основного контента"""
        logger.debug("Создание основного контента")

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Вкладки
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True)

        self.tab_data = self.tabview.add("📊 Данные")
        self.tab_model = self.tabview.add("🧠 Модель")
        self.tab_forecast = self.tabview.add("🔮 Прогноз")
        self.tab_visualization = self.tabview.add("📈 Визуализация")

        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_forecast_tab()
        self.setup_visualization_tab()

    def setup_data_tab(self):
        """Настройка вкладки данных"""
        title_label = ctk.CTkLabel(
            self.tab_data,
            text="Загрузка и анализ данных озонового слоя",
            font=self.title_font
        )
        title_label.pack(pady=20)

        button_frame = ctk.CTkFrame(self.tab_data, fg_color="transparent")
        button_frame.pack(pady=20)

        load_btn = ctk.CTkButton(
            button_frame,
            text="📥 Загрузить демо-данные",
            command=self.load_demo_data,
            font=self.normal_font,
            height=40,
            width=200
        )
        load_btn.pack(pady=10)

        self.data_info_text = ctk.CTkTextbox(self.tab_data, height=200)
        self.data_info_text.pack(fill="both", expand=True, padx=20, pady=10)
        self.data_info_text.insert("1.0",
                                   "Данные не загружены.\n\nНажмите кнопку выше для загрузки демонстрационных данных.")
        self.data_info_text.configure(state="disabled")

    def setup_model_tab(self):
        """Настройка вкладки модели"""
        title_label = ctk.CTkLabel(
            self.tab_model,
            text="Обучение гибридной нейросетевой модели",
            font=self.title_font
        )
        title_label.pack(pady=20)

        arch_frame = ctk.CTkFrame(self.tab_model)
        arch_frame.pack(fill="x", padx=20, pady=10)

        arch_label = ctk.CTkLabel(
            arch_frame,
            text="🏗️ Архитектура модели:",
            font=self.subtitle_font
        )
        arch_label.pack(pady=10)

        arch_text = """• Conv1D: 64 фильтра, ядро=3, ReLU
• LSTM: 128 нейронов  
• Dense: 64 → 32 нейрона
• Dropout: 0.3
• Оптимизатор: Adam (lr=0.001)"""
        arch_desc = ctk.CTkLabel(
            arch_frame,
            text=arch_text,
            font=self.normal_font,
            justify="left"
        )
        arch_desc.pack(pady=10)

        self.train_btn = ctk.CTkButton(
            self.tab_model,
            text="🚀 Начать обучение модели",
            command=self.train_model,
            font=self.normal_font,
            height=50,
            fg_color="#2E8B57",
            state="disabled"
        )
        self.train_btn.pack(pady=30)

        self.progress_bar = ctk.CTkProgressBar(self.tab_model, height=20)
        self.progress_bar.pack(fill="x", padx=50, pady=10)
        self.progress_bar.set(0)

        self.training_results = ctk.CTkTextbox(self.tab_model, height=150)
        self.training_results.pack(fill="x", padx=20, pady=20)
        self.training_results.insert("1.0", "Результаты обучения появятся здесь...")
        self.training_results.configure(state="disabled")

    def setup_forecast_tab(self):
        """Настройка вкладки прогноза"""
        title_label = ctk.CTkLabel(
            self.tab_forecast,
            text="Прогнозирование содержания озона",
            font=self.title_font
        )
        title_label.pack(pady=20)

        settings_frame = ctk.CTkFrame(self.tab_forecast)
        settings_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(settings_frame, text="Период прогноза (месяцев):", font=self.normal_font).pack(pady=5)
        self.forecast_period = ctk.CTkEntry(settings_frame, placeholder_text="12")
        self.forecast_period.pack(pady=5)
        self.forecast_period.insert(0, "12")

        self.forecast_btn = ctk.CTkButton(
            self.tab_forecast,
            text="🔮 Выполнить прогноз",
            command=self.run_forecast,
            font=self.normal_font,
            height=50,
            state="disabled"
        )
        self.forecast_btn.pack(pady=20)

        self.forecast_results = ctk.CTkTextbox(self.tab_forecast, height=200)
        self.forecast_results.pack(fill="both", expand=True, padx=20, pady=10)
        self.forecast_results.insert("1.0", "Результаты прогноза появятся здесь...")
        self.forecast_results.configure(state="disabled")

    def setup_visualization_tab(self):
        """Настройка вкладки визуализации"""
        title_label = ctk.CTkLabel(
            self.tab_visualization,
            text="Визуализация данных и прогнозов",
            font=self.title_font
        )
        title_label.pack(pady=10)

        controls_frame = ctk.CTkFrame(self.tab_visualization, fg_color="transparent")
        controls_frame.pack(fill="x", padx=20, pady=10)

        buttons = [
            ("📊 Исторические данные", self.show_historical),
            ("📈 Сезонность", self.show_seasonality),
            ("📉 Тренды", self.show_trends),
            ("🔮 Прогноз", self.show_forecast_plot)
        ]

        for text, command in buttons:
            btn = ctk.CTkButton(
                controls_frame,
                text=text,
                command=command,
                font=self.small_font,
                width=150
            )
            btn.pack(side="left", padx=5)

        self.viz_frame = ctk.CTkFrame(self.tab_visualization)
        self.viz_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor='#2b2b2b')
        self.canvas = FigureCanvasTkAgg(self.figure, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.show_welcome_plot()

    def create_status_bar(self):
        """Создание статус бара"""
        logger.debug("Создание статус бара")

        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Готов к работе",
            font=self.small_font
        )
        self.status_label.pack(side="left", padx=10, pady=5)

    def set_current_step(self, step_index):
        """Установка текущего шага"""
        self.current_step = step_index
        for i, btn in enumerate(self.step_buttons):
            if i == step_index:
                btn.configure(fg_color="#2E8B57")
            else:
                btn.configure(fg_color=("gray75", "gray25"))

        tabs = ["📊 Данные", "🧠 Модель", "🔮 Прогноз", "📈 Визуализация"]
        self.tabview.set(tabs[step_index])

    def update_status(self, message):
        """Обновление статуса"""
        self.status_label.configure(text=message)
        self.update()

    @log_function_call
    def load_demo_data(self):
        """Загрузка демо-данных"""
        self.update_status("Создание демонстрационных данных...")
        logger.info("Пользователь запросил загрузку демонстрационных данных")

        thread = threading.Thread(target=self._load_demo_data_thread)
        thread.daemon = True
        thread.start()

    def _load_demo_data_thread(self):
        """Поток загрузки демо-данных"""
        try:
            self.oso_data = self.data_loader.create_demo_oso_data()
            self.after(0, self._on_data_loaded)
        except Exception as e:
            logger.error(f"Ошибка создания демо-данных: {str(e)}")
            self.after(0, lambda: self._on_data_error(str(e)))

    def _on_data_loaded(self):
        """После загрузки данных"""
        logger.info("Демо-данные успешно созданы и загружены в интерфейс")
        self.update_status("Демо-данные успешно созданы")

        info_text = f"""✅ ДЕМО-ДАННЫЕ УСПЕШНО СОЗДАНЫ

📅 Период: 1960-2024 гг.
📊 Записей: {len(self.oso_data):,}
📍 Регион: Томская область

Данные готовы для обучения модели!"""

        self.data_info_text.configure(state="normal")
        self.data_info_text.delete("1.0", "end")
        self.data_info_text.insert("1.0", info_text)
        self.data_info_text.configure(state="disabled")

        self.step_buttons[1].configure(state="normal")
        self.train_btn.configure(state="normal")

        self.show_historical()
        messagebox.showinfo("Успех", "Демонстрационные данные успешно созданы!")

    def _on_data_error(self, error_msg):
        """Ошибка загрузки данных"""
        logger.error(f"Ошибка загрузки данных: {error_msg}")
        self.update_status("Ошибка загрузки данных")
        messagebox.showerror("Ошибка", f"Не удалось загрузить данные:\n{error_msg}")

    @log_function_call
    def train_model(self):
        """Обучение модели"""
        if self.oso_data is None:
            logger.warning("Попытка обучения модели без данных")
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        logger.info("Начало процесса обучения модели")
        self.update_status("Обучение гибридной модели...")
        self.train_btn.configure(state="disabled")
        self.progress_bar.set(0)

        thread = threading.Thread(target=self._training_thread)
        thread.daemon = True
        thread.start()

    def _training_thread(self):
        """Поток обучения"""
        try:
            logger.info("Запуск потока обучения модели")
            for i in range(101):
                self.after(0, lambda val=i: self.progress_bar.set(val / 100))
                threading.Event().wait(0.05)

            self.model.train(self.oso_data)
            self.after(0, self._on_training_complete)

        except Exception as e:
            logger.error(f"Ошибка в потоке обучения: {str(e)}")
            self.after(0, lambda: self._on_training_error(str(e)))

    def _on_training_complete(self):
        """После обучения"""
        logger.info("Обучение модели завершено успешно")
        self.update_status("Модель успешно обучена")
        self.train_btn.configure(state="normal")

        results_text = f"""✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА!

📊 МЕТРИКИ КАЧЕСТВА:
• MAE: {self.model.metrics['mae']:.3f} е.Д.
• RMSE: {self.model.metrics['rmse']:.3f} е.Д. 
• Точность: {self.model.metrics['accuracy']:.1%}

Модель готова к прогнозированию!"""

        self.training_results.configure(state="normal")
        self.training_results.delete("1.0", "end")
        self.training_results.insert("1.0", results_text)
        self.training_results.configure(state="disabled")

        self.step_buttons[2].configure(state="normal")
        self.forecast_btn.configure(state="normal")

        messagebox.showinfo("Успех", "Гибридная модель успешно обучена!")

    def _on_training_error(self, error_msg):
        """Ошибка обучения"""
        logger.error(f"Ошибка обучения модели: {error_msg}")
        self.update_status("Ошибка обучения")
        self.train_btn.configure(state="normal")
        messagebox.showerror("Ошибка", f"Ошибка обучения модели:\n{error_msg}")

    @log_function_call
    def run_forecast(self):
        """Запуск прогноза"""
        if not self.model.is_trained:
            logger.warning("Попытка прогнозирования без обученной модели")
            messagebox.showwarning("Внимание", "Сначала обучите модель!")
            return

        try:
            periods = int(self.forecast_period.get())
            logger.info(f"Запуск прогноза на {periods} месяцев")
            self.update_status(f"Выполнение прогноза на {periods} месяцев...")

            self.forecast = self.model.forecast(periods)
            self._on_forecast_complete(periods)

        except Exception as e:
            logger.error(f"Ошибка выполнения прогноза: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка прогноза: {str(e)}")

    def _on_forecast_complete(self, periods):
        """После выполнения прогноза"""
        logger.info(f"Прогноз на {periods} месяцев успешно выполнен")
        self.update_status(f"Прогноз на {periods} месяцев выполнен")

        forecast_text = f"""📈 ПРОГНОЗ ОСО НА {periods} МЕСЯЦЕВ:

"""
        for i, value in enumerate(self.forecast[:8], 1):
            trend = "↗️" if value > 300 else "↘️" if value < 280 else "➡️"
            forecast_text += f"Месяц {i:2d}: {value:6.1f} е.Д. {trend}\n"

        if periods > 8:
            forecast_text += f"... и ещё {periods - 8} месяцев\n\n"

        forecast_text += f"""📊 СТАТИСТИКА ПРОГНОЗА:
• Среднее: {np.mean(self.forecast):.1f} е.Д.
• Минимум: {np.min(self.forecast):.1f} е.Д.
• Максимум: {np.max(self.forecast):.1f} е.Д."""

        self.forecast_results.configure(state="normal")
        self.forecast_results.delete("1.0", "end")
        self.forecast_results.insert("1.0", forecast_text)
        self.forecast_results.configure(state="disabled")

        self.step_buttons[3].configure(state="normal")
        self.show_forecast_plot()
        messagebox.showinfo("Успех", f"Прогноз на {periods} месяцев успешно выполнен!")

    def show_welcome_plot(self):
        """Показать приветственный график"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        x = np.linspace(0, 10, 100)
        y = 300 + 20 * np.sin(x) + 5 * np.cos(2 * x)

        ax.plot(x, y, 'cyan', linewidth=2, alpha=0.8, label='Пример данных ОСО')
        ax.fill_between(x, y - 10, y + 10, alpha=0.2, color='cyan')

        ax.set_title('🌍 Система прогнозирования озонового слоя',
                     color='white', fontsize=14, pad=20)
        ax.set_xlabel('Время', color='white')
        ax.set_ylabel('ОСО (е.Д.)', color='white')

        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')

        ax.text(0.5, 0.5, 'Загрузите данные для начала работы',
                transform=ax.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#3CB371", alpha=0.8),
                color='white')

        self.figure.tight_layout()
        self.canvas.draw()

    def show_historical(self):
        """Показать исторические данные"""
        if self.oso_data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            dates = pd.date_range('1960-01-01', '2024-12-31', freq='M')[:len(self.oso_data)]
            values = self.oso_data['oso'].values

            ax.plot(dates, values, 'lightblue', alpha=0.7, linewidth=1, label='Данные ОСО')

            window = 12
            if len(values) > window:
                rolling_mean = pd.Series(values).rolling(window=window).mean()
                ax.plot(dates[window - 1:], rolling_mean[window - 1:], 'yellow',
                        linewidth=2, label=f'Скользящее среднее ({window} мес.)')

            ax.set_title('Исторические данные ОСО (1960-2024)', color='white', fontsize=14)
            ax.set_xlabel('Год', color='white')
            ax.set_ylabel('ОСО (е.Д.)', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()

    def show_seasonality(self):
        """Показать сезонность"""
        if self.oso_data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            seasonal_data = []
            for year in range(1960, 2025):
                year_data = self.oso_data[self.oso_data['year'] == year]
                if len(year_data) == 12:
                    seasonal_data.append(year_data['oso'].values)

            if seasonal_data:
                seasonal_avg = np.mean(seasonal_data, axis=0)
                months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                          'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

                ax.plot(months, seasonal_avg, 'limegreen', linewidth=3,
                        marker='o', markersize=6, label='Средняя сезонность')
                ax.fill_between(months, seasonal_avg - 5, seasonal_avg + 5,
                                alpha=0.2, color='limegreen')

                ax.set_title('Сезонность ОСО (средние значения по месяцам)',
                             color='white', fontsize=14)
                ax.set_xlabel('Месяц', color='white')
                ax.set_ylabel('ОСО (е.Д.)', color='white')
                ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
                ax.grid(True, alpha=0.3, color='gray')
                ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()

    def show_trends(self):
        """Показать тренды"""
        if self.oso_data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            yearly_avg = self.oso_data.groupby('year')['oso'].mean()

            ax.plot(yearly_avg.index, yearly_avg.values, 'orange',
                    linewidth=2, marker='o', markersize=3, label='Среднегодовые значения')

            z = np.polyfit(yearly_avg.index, yearly_avg.values, 1)
            p = np.poly1d(z)
            ax.plot(yearly_avg.index, p(yearly_avg.index), "red", linewidth=2,
                    label=f'Тренд: {z[0]:.3f}/год')

            ax.set_title('Многолетние тренды ОСО (1960-2024)', color='white', fontsize=14)
            ax.set_xlabel('Год', color='white')
            ax.set_ylabel('ОСО (е.Д.)', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()

    def show_forecast_plot(self):
        """Показать прогноз"""
        if self.oso_data is not None and self.forecast is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            historical = self.oso_data.tail(36)
            dates_hist = pd.date_range('2022-01-01', '2024-12-31', freq='M')[:len(historical)]
            values_hist = historical['oso'].values

            forecast_dates = pd.date_range('2025-01-01', periods=len(self.forecast), freq='M')

            ax.plot(dates_hist, values_hist, 'lightblue', linewidth=2, label='Исторические данные')
            ax.plot(forecast_dates, self.forecast, 'magenta', linewidth=2, label='Прогноз')
            ax.fill_between(forecast_dates, self.forecast - 3, self.forecast + 3,
                            alpha=0.2, color='magenta')

            ax.set_title('Прогноз общего содержания озона', color='white', fontsize=14)
            ax.set_xlabel('Дата', color='white')
            ax.set_ylabel('ОСО (е.Д.)', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()


def main():
    try:
        logger.info("=" * 50)
        logger.info("🌍 ЗАПУСК ПРИЛОЖЕНИЯ OSO FORECASTING")
        logger.info("=" * 50)

        app = ModernOzoneApp()
        app.mainloop()

        logger.info("✅ Приложение завершило работу нормально")

    except Exception as e:
        error_msg = f"💥 КРИТИЧЕСКАЯ ОШИБКА: {str(e)}\n{traceback.format_exc()}"
        logger.critical(error_msg)
        messagebox.showerror("Критическая ошибка",
                             f"Приложение завершилось с ошибкой:\n{str(e)}\n\nПодробности в лог-файле.")


if __name__ == "__main__":
    main()