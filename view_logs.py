import os
import glob
from datetime import datetime


def view_latest_log():
    log_files = glob.glob('logs/*.log')

    if not log_files:
        print("Нет лог-файлов")
        return

    latest_log = max(log_files, key=os.path.getctime)

    print(f"Последний лог-файл: {latest_log}")
    print("=" * 80)

    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")


def view_all_logs():
    log_files = glob.glob('logs/*.log')

    if not log_files:
        print("Нет лог-файлов")
        return

    print(f"Всего лог-файлов: {len(log_files)}")
    print("=" * 80)

    for log_file in sorted(log_files):
        size = os.path.getsize(log_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
        print(f"{os.path.basename(log_file)} - {size:,} байт - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        view_all_logs()
    else:
        view_latest_log()