import subprocess
import os
import csv
import re
from pathlib import Path
import argparse
import sys
import traceback

def run_test_with_threads(executable_path, matrix_size, repeats, threads_list, output_csv="timing_results.csv"):
    """
    Запускает программу с разными значениями OMP_NUM_THREADS и записывает результаты в CSV.
    
    Args:
        executable_path: путь к исполняемому файлу программы
        matrix_size: размер матрицы для теста
        repeats: количество повторов теста
        threads_list: список значений потоков для тестирования
        output_csv: путь к выходному CSV файлу
    """
    
    results = []
    headers = ["Threads", "Init_Time_ms", "LU_Time_ms", "Total_Time_ms", "Correct_Count", "Incorrect_Count", "Success_Rate"]
    
    # Преобразуем относительный путь в абсолютный
    executable_path = str(Path(executable_path).absolute())
    print(f"Абсолютный путь к программе: {executable_path}")
    
    # Проверяем, существует ли файл
    if not Path(executable_path).exists():
        print(f"ОШИБКА: Файл {executable_path} не существует!")
        return
    
    for threads in threads_list:
        print(f"\n{'='*60}")
        print(f"Запуск с OMP_NUM_THREADS={threads}")
        print(f"{'='*60}")
        
        # Устанавливаем переменную окружения
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        
        # Формируем аргументы командной строки
        args = [executable_path, str(matrix_size), str(repeats)]
        
        try:
            print(f"Запуск команды: {' '.join(args)}")
            
            # Запускаем процесс БЕЗ shell=True для Git Bash
            process = subprocess.Popen(
                args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,  # Важно: False для Git Bash
                cwd=os.path.dirname(executable_path)  # Рабочая директория - где находится exe
            )
            
            stdout, stderr = process.communicate()
            
            # Пробуем разные кодировки
            try:
                stdout_text = stdout.decode('utf-8', errors='ignore')
            except:
                stdout_text = stdout.decode('cp1251', errors='ignore')
            
            try:
                stderr_text = stderr.decode('utf-8', errors='ignore')
            except:
                stderr_text = stderr.decode('cp1251', errors='ignore')
            
            print(f"Код возврата: {process.returncode}")
            
            # Сохраняем полный вывод в файл
            output_file = f"output_threads_{threads}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"=== STDOUT ===\n{stdout_text}\n")
                f.write(f"=== STDERR ===\n{stderr_text}\n")
            print(f"Полный вывод сохранён в {output_file}")
            
            if process.returncode != 0:
                print(f"Ошибка выполнения (код {process.returncode}):")
                print(f"STDOUT (первые 1000 символов):\n{stdout_text[:1000]}")
                print(f"STDERR:\n{stderr_text}")
                continue
            
            # Выводим первые 500 символов для отладки
            print(f"Начало вывода программы:")
            print(stdout_text[:500])
            print("..." if len(stdout_text) > 500 else "")
            
            # Парсим нужные значения - улучшенные регулярки
            lu_time = parse_value(stdout_text, r"Minimum time for LU decomposition:\s*(\d+)\s*ms")
            if lu_time is None:
                # Попробуем другой формат
                lu_time = parse_value(stdout_text, r"LU decomposition.*?(\d+)\s*ms")
            
            init_time = parse_value(stdout_text, r"Minimum time for init random matrix:\s*(\d+)\s*ms")
            if init_time is None:
                init_time = parse_value(stdout_text, r"init random matrix.*?(\d+)\s*ms")
            
            total_time = parse_value(stdout_text, r"Minimum total time:\s*(\d+)\s*ms")
            if total_time is None:
                total_time = parse_value(stdout_text, r"total time.*?(\d+)\s*ms")
            
            correct_count = parse_value(stdout_text, r"Correct count:\s*(\d+)")
            incorrect_count = parse_value(stdout_text, r"Incorrect count:\s*(\d+)")
            
            print(f"Найденные значения:")
            print(f"  LU время: {lu_time}")
            print(f"  Инициализация: {init_time}")
            print(f"  Общее время: {total_time}")
            print(f"  Успешные: {correct_count}")
            print(f"  Ошибки: {incorrect_count}")
            
            # Вычисляем процент успеха
            if correct_count is not None and incorrect_count is not None:
                total = correct_count + incorrect_count
                success_rate = (correct_count / total * 100) if total > 0 else 0
            else:
                success_rate = None
            
            # Добавляем в результаты только если есть данные
            if any([lu_time is not None, init_time is not None, total_time is not None,
                   correct_count is not None, incorrect_count is not None]):
                results.append({
                    "Threads": threads,
                    "Init_Time_ms": init_time if init_time is not None else "",
                    "LU_Time_ms": lu_time if lu_time is not None else "",
                    "Total_Time_ms": total_time if total_time is not None else "",
                    "Correct_Count": correct_count if correct_count is not None else "",
                    "Incorrect_Count": incorrect_count if incorrect_count is not None else "",
                    "Success_Rate": f"{success_rate:.2f}" if success_rate is not None else ""
                })
                
                print(f"\nРезультаты для {threads} потоков:")
                print(f"  Время инициализации: {init_time if init_time is not None else 'N/A'} мс")
                print(f"  Время LU разложения: {lu_time if lu_time is not None else 'N/A'} мс")
                print(f"  Общее время: {total_time if total_time is not None else 'N/A'} мс")
                print(f"  Успешные: {correct_count if correct_count is not None else 'N/A'}, Ошибки: {incorrect_count if incorrect_count is not None else 'N/A'}")
                print(f"  Успешность: {success_rate:.1f}%" if success_rate is not None else "  Успешность: N/A")
            else:
                print("Не удалось извлечь данные из вывода программы")
            
        except FileNotFoundError:
            print(f"Ошибка: файл {executable_path} не найден!")
            print(f"Текущая директория: {os.getcwd()}")
            return
        except Exception as e:
            print(f"Ошибка при выполнении: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
    
    # Записываем результаты в CSV, если есть данные
    if results:
        save_to_csv(results, output_csv, headers)
        print(f"\nВсе результаты сохранены в {output_csv}")
        
        # Выводим сводную таблицу
        print("\n" + "="*80)
        print("Сводная таблица результатов:")
        print("="*80)
        print(f"{'Потоки':<8} {'LU время (мс)':<15} {'Ускорение':<12} {'Успешность':<12}")
        print("-"*50)
        
        base_time = None
        for row in results:
            threads = row["Threads"]
            lu_time_str = row["LU_Time_ms"]
            
            try:
                lu_time = float(lu_time_str) if lu_time_str else None
            except:
                lu_time = None
            
            if lu_time is not None:
                if base_time is None and threads == 1:
                    base_time = lu_time
                
                if base_time and lu_time:
                    speedup = base_time / lu_time
                else:
                    speedup = None
                
                success_rate = row["Success_Rate"]
                success_str = success_rate if success_rate else "N/A"
                speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
                lu_str = f"{lu_time}" if lu_time is not None else "N/A"
                
                print(f"{threads:<8} {lu_str:<15} {speedup_str:<12} {success_str:<12}")
    else:
        print("\nНет данных для сохранения в CSV!")

def parse_value(text, pattern):
    """
    Парсит значение из текста с помощью регулярного выражения.
    Возвращает None, если значение не найдено.
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            try:
                return float(match.group(1))
            except ValueError:
                return match.group(1)
    return None

def save_to_csv(data, filename, headers):
    """Сохраняет данные в CSV файл."""
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Запуск тестов LU разложения с разным количеством потоков')
    parser.add_argument('--executable', '-e', 
                       help='Путь к исполняемому файлу программы')
    parser.add_argument('--size', '-s', type=int, default=1000,
                       help='Размер матрицы (по умолчанию: 1000)')
    parser.add_argument('--repeats', '-r', type=int, default=1,
                       help='Количество повторений теста (по умолчанию: 1)')
    parser.add_argument('--output', '-o', default='omp_threads_timing.csv',
                       help='Имя выходного CSV файла (по умолчанию: omp_threads_timing.csv)')
    parser.add_argument('--threads', '-t', type=int, nargs='+',
                       default=[1, 2, 4, 6, 8, 10, 12, 16],
                       help='Список значений потоков для тестирования')
    
    args = parser.parse_args()
    
    # Автоматически определяем путь к программе, если не указан
    if not args.executable:
        # Попробуем найти стандартные пути
        possible_paths = [
            "../build/intel_oneAPI/Release/Matrix_Project.exe",
            "./Matrix_Project.exe",
            "Matrix_Project.exe",
            "build/intel_oneAPI/Release/Matrix_Project.exe",
        ]
        
        for path in possible_paths:
            abs_path = Path(path).absolute()
            if abs_path.exists():
                args.executable = str(abs_path)
                print(f"Найден исполняемый файл: {args.executable}")
                break
        
        if not args.executable:
            print("ОШИБКА: Исполняемый файл не найден!")
            print("Укажите путь с помощью --executable")
            print("Пример: python run_omp_tests.py --executable path/to/Matrix_Project.exe")
            return
    
    # Проверяем существование файла
    abs_path = Path(args.executable).absolute()
    if not abs_path.exists():
        print(f"ОШИБКА: Файл {abs_path} не существует!")
        print(f"Текущая директория: {os.getcwd()}")
        
        # Покажем содержимое папки build
        build_dir = Path(".") / "build"
        if build_dir.exists():
            print(f"\nСодержимое папки build:")
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    if file.endswith('.exe'):
                        print(f"  {Path(root) / file}")
        return
    
    print(f"Исполняемый файл: {abs_path}")
    print(f"Размер матрицы: {args.size}")
    print(f"Повторений: {args.repeats}")
    print(f"Тестируемые потоки: {args.threads}")
    print(f"Выходной файл: {args.output}")
    print(f"Текущая рабочая директория: {os.getcwd()}")
    
    # Запускаем тесты
    run_test_with_threads(
        executable_path=str(abs_path),
        matrix_size=args.size,
        repeats=args.repeats,
        threads_list=args.threads,
        output_csv=args.output
    )

if __name__ == "__main__":
    main()
