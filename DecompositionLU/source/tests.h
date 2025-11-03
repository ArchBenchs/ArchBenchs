#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <fstream>

using namespace std::chrono;

class SquareMatrix;

// структура дл€ возврата времени из теста на оптимизацию
struct ReturnedTimes {
	milliseconds InitTime{0};
	milliseconds LUTime{0};
	milliseconds TotalTime{0};
};

// массив указателей на функции тестов на работоспособность
using WorkTestPtr = bool (*)();

class TestSystem {
private:
	// вектор тестов работоспособности
	static std::vector<WorkTestPtr> work_tests;

	// сюда пошел обший код дл€ тестов на работоспособность
	static bool test_LU(SquareMatrix& A, size_t n, std::string test_num);

	// тесты на работоспособность

	static bool test1(); 
	static bool test2(); 
	static bool test3(); 

	// тест на оптимизацию и врем€ выполнени€, n - размер кв. матрицы, 
	// how_many_times - сколько раз прогнать
	static void test_time(size_t _n, size_t how_many_times = 1);
	// внутренн€€ функци€ test_time
	static ReturnedTimes single_test_time(size_t n);

	// функци€ добавлени€ всех тестов работоспособности в work_tests
	static void add_tests();

	// функции дл€ вывода

	static void print_test_start(std::string s = "");
	static void print_test_end(std::string s = "");

	static std::ofstream file_out;
	static std::ostream* out;

	template<typename T>
	static inline void print(const T& value) { *out << value; }

	static inline void p_endl() { *out << std::endl; }
public:
	static void run_all_tests(std::string filename = "");
};