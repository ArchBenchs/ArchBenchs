#pragma once

#include "square_matrix.h"

#include <vector>
#include <string>
#include <chrono>
#include <fstream>

using namespace std::chrono;

// структура дл€ возврата времени из теста на оптимизацию
struct ReturnedResults {
	milliseconds InitTime{0};
	milliseconds LUTime{0};
	milliseconds TotalTime{0};
	bool is_correct;
};

// указатель на функцию теста на работоспособность
using WorkabilityTestPtr = bool (*)();

class TestSystem {
private:
	// флаг, определ€ющий, производитс€ ли проверка результата в test_time (по умолчанию true)
	static bool do_accuracy_check;
	// вектор тестов работоспособности
	static std::vector<WorkabilityTestPtr> workability_tests;

	// сюда пошел обший код дл€ тестов на работоспособность
	static bool test_LU(SquareMatrix& A, std::string test_num,
		bool print_a = 0, bool print_lu = 0, bool print_res = 0);

	// тесты на работоспособность

	static bool test1(); 
	static bool test2(); 
	static bool test3();
	static bool test4();

	// тест на оптимизацию и врем€ выполнени€, n - размер кв. матрицы, how_many_times - сколько раз запускать
	static void test_time(size_t _n, size_t how_many_times = 1);
	// внутренн€€ функци€ test_time
	static ReturnedResults single_test_time(size_t n, size_t iter);

	// функции дл€ вывода

	static void print_test_start(std::string s = "");
	static void print_test_end(std::string s = "");

	static std::ofstream file_out;
	static std::ostream* out;

	template<typename T>
	static __forceinline void print(const T& value) { *out << value; }

	static __forceinline void p_endl() { *out << std::endl; }

	// функци€, анализирующа€ входное число как число обусловленности матрицы
	static void analyze_cond(double cond);
public:
	// функци€, добавл€юща€ все тесты работоспособности в workability_tests
	static void enable_workability_tests();
	// отключение проверки результата в test_time
	static void disable_accuracy_check();

	static void run_all_tests(size_t n = 5000, size_t count = 1, std::string filename = "");
};