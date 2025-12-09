#include "tests.h"
#include "square_matrix.h"

#include <random>

#define TP steady_clock::time_point 
#define NOW steady_clock::now()

std::vector<WorkTestPtr> TestSystem::work_tests;
std::ofstream TestSystem::file_out;
std::ostream* TestSystem::out = &std::cout;

void TestSystem::add_tests() {
	work_tests.push_back(TestSystem::test1);
	work_tests.push_back(TestSystem::test2);
	work_tests.push_back(TestSystem::test3);
	work_tests.push_back(TestSystem::test4);
}

void TestSystem::print_test_start(std::string s) {
	print("\n------------------------------ Test ");
	print(s); print(" ------------------------------");
	p_endl();
}

void TestSystem::print_test_end(std::string s) {
	print("--------------------------------------------------------------------\nTest");
	print(s); print(": ");
}

void TestSystem::run_all_tests(size_t n, size_t count, std::string filename) {
	if (filename != "") { 
		file_out.open(filename);
		if (!file_out.is_open()) {
			std::cerr << "Failed to open file: " << filename << std::endl;
			out = &std::cout;
		}
		else { out = &file_out; }
	}
	print("TestSystem:\n");
	add_tests(); bool last_res;
	for (auto TestPtr : work_tests) {
		last_res = (*TestPtr)();
		print((last_res) ? "true\n\n" : "false\n\n");
	} p_endl();

	/*test_time(1000, 10);
	test_time(3000, 6);
	test_time(7000, 4);*/
	test_time(n, count);

	if (filename != "") { file_out.close(); }
}

bool TestSystem::test_LU(SquareMatrix& A, std::string test_num,
	bool print_a, bool print_lu, bool print_res) {

	const size_t n = A.get_size();
	SquareMatrix LU(A);
	block_get_LU(LU.get_array(), n, n);
	SquareMatrix L(n), U(n);
	LU.decompose_LU(L, U);
	SquareMatrix Res = L * U;

	print_test_start(test_num);
	/*if (test_num != "4") { print(LU); p_endl(); }*/
	if (print_a) { print("Matrix A:\n"); print(A); p_endl(); }
	if (print_lu) { print_LU(LU, *out); }
	if (print_res) { print("Matrix Res = L * U:\n"); print(Res); }
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
	return test_LU(A, "1", 1, 1, 1);
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
	return test_LU(A, "2", 1, 1, 1);
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
	} 

	return test_LU(A, "3", 1, 1, 1);
}

bool TestSystem::test4() {
	const size_t n = 100;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> double_generator(-1e6, 1e6);

	SquareMatrix A(n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			A(i, j) = double_generator(gen);
		}
	}

	return test_LU(A, "4");
}

ReturnedResults TestSystem::single_test_time(size_t n) {
	ReturnedResults results;
	TP start_init = NOW;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> double_generator(-1e6, 1e6);

	SquareMatrix A(n), LU(n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			double val = double_generator(gen);
			A(i, j) = val; LU(i, j) = val;
		}
	} results.InitTime = duration_cast<milliseconds>(NOW - start_init);

	TP start_LU = NOW;
	block_get_LU(LU.get_array(), n, n);
	results.LUTime = duration_cast<milliseconds>(NOW - start_LU);

	results.TotalTime = duration_cast<milliseconds>(NOW - start_init);
	SquareMatrix L(n), U(n);
	LU.decompose_LU(L, U);
	results.is_correct = (A == L * U);
	return results;
}

void TestSystem::test_time(size_t _n, size_t how_many_times) {
	print_test_start("time");
	chrono::milliseconds time_init{ 1000000000 }, time_LU{ 1000000000 }, total_time{ 1000000000 };
	const size_t n = _n;
	int cc = 0, incc = 0; bool all_correct = true;
	print("Testing with n = "); print(_n); print(", "); 
	print(how_many_times); print(" times:");
	for (size_t iter = 0; iter < how_many_times; iter++) {
		ReturnedResults res = single_test_time(n);
		time_init = (time_init > res.InitTime) ? res.InitTime : time_init;
		time_LU = (time_LU > res.LUTime) ? res.LUTime : time_LU;
		total_time = (total_time > res.TotalTime) ? res.TotalTime : total_time;
		all_correct &= res.is_correct;
		if (res.is_correct) { ++cc; } else { ++incc; }
	}
	print("\nMinimum time for init random matrix: ");
	print(time_init.count()); 

	print(" ms\nMinimum time for LU decomposition: ");
	print(time_LU.count());

	print(" ms\nMinimum total time: ");
	print(total_time.count()); 

	print(" ms\n\nTotal test result: "); print(all_correct);
	print("\nCorrect count: "); print(cc);
	print("\nIncorrect count: "); print(incc);
	print("\n-----------------------------------------------------------------------");
}
