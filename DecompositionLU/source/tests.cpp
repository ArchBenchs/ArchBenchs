#include "tests.h"
#include "square_matrix.h"

#include <random>

#define TP steady_clock::time_point 
#define NOW steady_clock::now()

std::vector<WorkTestPtr> TestSystem::work_tests;

void TestSystem::add_tests() {
	work_tests.push_back(TestSystem::test1);
	work_tests.push_back(TestSystem::test2);
	work_tests.push_back(TestSystem::test3);
}

void TestSystem::print_test_start(std::string s) {
	std::cout
		<< "\n------------------------------ Test " 
		<< s << " ------------------------------"
		<< endl;
}

void TestSystem::print_test_end(std::string s) {
	std::cout 
		<< "--------------------------------------------------------------------"
		<< "\nTest " << s << ": ";
}

void TestSystem::run_all_tests() {
	std::cout << "TestSystem:\n";
	add_tests(); bool last_res;
	for (auto TestPtr : work_tests) {
		last_res = (*TestPtr)();
		std::cout << ((last_res) ? "true" : "false") << endl << endl;
	} std::cout << endl;

	test_several_times(1000, 10);
	test_several_times(3000, 1);
	//test_several_times(7000, 1);
	//test_several_times(10000, 1);
}

bool TestSystem::test_LU(SquareMatrix& A, size_t n, std::string test_num) {
	print_test_start(test_num);
	SquareMatrix LU(A);

	get_LU(LU);
	SquareMatrix L(n), U(n);
	LU.decompose_LU(L, U);
	print_LU(LU);
	SquareMatrix Res = L * U;
	
	std::cout << "Matrix Res = L * U:\n" << Res;
	print_test_end(test_num);
	return A == Res;
}

bool TestSystem::test1() {
	const size_t n = 3;
	Type arr[n * n]{ 
		2, 3, 1, 
		4, 7, 7, 
		6, 18, 22 
	};
	SquareMatrix A(n, arr);
	return test_LU(A, n, "1");
}

bool TestSystem::test2() {
	const size_t n = 4;
	Type arr[n * n]{
		2, 3, 1, 1,
		4, 7, 7, 1,
		6, 18, 22, 1,
		1, 1, 1, 1
	};
	SquareMatrix A(n, arr);
	return test_LU(A, n, "2");
}

bool TestSystem::test3() {
	const size_t n = 4;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> double_generator(-1e6, 1e6);

	SquareMatrix A(n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			A(i, j) = double_generator(gen);
		}
	} std::cout << "Matrix A:\n" << A << endl;
	return test_LU(A, n, "3");
}

ReturnedTimes TestSystem::single_test_time(size_t n) {
	ReturnedTimes result_times;
	TP start_init = NOW;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> double_generator(-1e6, 1e6);

	SquareMatrix A(n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			A(i, j) = double_generator(gen);
		}
	} result_times.InitTime = duration_cast<milliseconds>(NOW - start_init);

	TP start_LU = NOW;
	get_LU(A);
	result_times.LUTime = duration_cast<milliseconds>(NOW - start_LU);

	result_times.TotalTime += duration_cast<milliseconds>(NOW - start_init);
	return result_times;
}

void TestSystem::test_time(size_t _n, size_t how_many_times) {
	print_test_start("time");
	chrono::milliseconds time_init{ 0 }, time_LU{ 0 }, total_time{ 0 };
	const size_t n = _n;
	std::cout << "Testing with n = " << _n << ", " << how_many_times << " times:";
	for (size_t iter = 0; iter < how_many_times; iter++) {
		ReturnedTimes res = test_time(n);
		time_init += res.InitTime;
		time_LU += res.LUTime;
		total_time += res.TotalTime;
	}
	std::cout << "\nArithmetic mean of time for init random matrix: ~"
		<< time_init.count() / how_many_times << "ms";
	std::cout << "\nArithmetic mean of time for LU decomposition: ~"
		<< time_LU.count() / how_many_times << "ms";
	std::cout << "\nArithmetic mean of total time: ~"
		<< total_time.count() / how_many_times << "ms\n";
	std::cout
		<< "-----------------------------------------------------------------------";
}
