#pragma once
#include <vector>
#include <string>

class TestSystem {
private:
	using TestPtr = bool (*)();
	static std::vector<TestPtr> tests;

	// тесты на работоспособность
	static bool test1(); 
	static bool test2();
	static bool test3();
	// тест на оптимизацию и время выполнения
	static bool test_time(size_t how_many_times = 1, size_t _n = 1000); 

	// функция добавления всех тестов в tests
	static void add_tests();

	// функции для вывода
	static void print_test_start(std::string s = "");
	static void print_test_end(std::string s = "");
public:
	static void run_all_tests();
};