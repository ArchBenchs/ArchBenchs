#include "tests.h"
#include "square_matrix.h"

#include <random>
#include <chrono>
using namespace chrono;

#define TP steady_clock::time_point 
#define NOW steady_clock::now()

std::vector<bool(*)()> TestSystem::tests;

void TestSystem::add_tests() {
	tests.push_back(TestSystem::test1);
	tests.push_back(TestSystem::test2);
}

void TestSystem::run_all_tests() {
	std::cout << "TestSystem:\n\n";
	add_tests(); bool last_res;
	for (auto TestPtr : tests) {
		last_res = (*TestPtr)();
		std::cout << ((last_res) ? "true" : "false") << endl;
	} std::cout << endl;
	/*test_time(10, 1000);*/
	test_time(1, 3000);
	/*test_time(10, 7000);*/
	/*test_time(1, 10000);*/
}

bool TestSystem::test1() {
	std::cout << endl;
	const size_t n = 3;
	Type arr[n * n]{ 
		2, 3, 1, 
		4, 7, 7, 
		6, 18, 22 
	};
	SquareMatrix A(n, arr);
	SquareMatrix LU(A);

	get_LU(LU);

	SquareMatrix L(n), U(n);
	LU.decompose_LU(L, U);
	SquareMatrix Res = L * U;

	print_LU(LU);
	std::cout << "Matrix Res = L * U:\n" << Res
		<< "Test1: ";
	return A == Res;
}

bool TestSystem::test2() {
	std::cout << endl;
	const size_t n = 4;
	Type arr[n * n]{
		2, 3, 1, 1,
		4, 7, 7, 1,
		6, 18, 22, 1,
		1, 1, 1, 1
	};
	SquareMatrix A(n, arr);
	SquareMatrix LU(A);

	get_LU(LU);

	SquareMatrix L(n), U(n);
	LU.decompose_LU(L, U);
	SquareMatrix Res = L * U;

	print_LU(LU);
	std::cout << "Matrix Res = L * U:\n" << Res 
		<< "Test2: ";
	return A == Res;
}

bool TestSystem::test_time(size_t how_many_times, size_t _n) {
	chrono::milliseconds time_init{0}, time_LU{0}, total_time{0};
	const size_t n = _n;
	std::cout << "\nTesting with n = " << _n << ":";
	for (size_t iter = 0; iter < how_many_times; iter++) {
		TP start_init = NOW;
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> double_generator(-1e6, 1e6);

		SquareMatrix A(n);
		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < n; j++) {
				A(i, j) = double_generator(gen);
			}
		} time_init += duration_cast<milliseconds>(NOW - start_init);

		TP start_LU = NOW;
		get_LU(A);
		time_LU += duration_cast<milliseconds>(NOW - start_LU);

		total_time += duration_cast<milliseconds>(NOW - start_init);
	}
	std::cout << "\nArithmetic mean of time for init random matrix: ~"
		<< time_init.count() / how_many_times << "ms";
	std::cout << "\nArithmetic mean of time for LU decomposition: ~"
		<< time_LU.count() / how_many_times << "ms";
	std::cout << "\nArithmetic mean of total time: ~"
		<< total_time.count() / how_many_times << "ms\n";
	return true;
}
