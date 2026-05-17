#pragma once

#include "square_matrix.h"

#include <vector>
#include <string>
#include <chrono>
#include <fstream>


#define mkl 1
#define eigen 2

using namespace std::chrono;

// structure for returning timing results from optimization test
struct ReturnedResults {
	milliseconds InitTime{ 0 };
	milliseconds LUTime{ 0 };
	milliseconds TotalTime{ 0 };
	bool is_correct;
};

// function pointer for workability test functions
using WorkabilityTestPtr = bool (*)();

class TestSystem {
private:
	// flag determining whether result verification is performed in test_time (default true)
	static bool do_accuracy_check;
	// flag determining whether the matrix is filled with random values or
	// computed values that depend on the matrix size
	static bool random_initialization;
	// vector of workability tests
	static std::vector<WorkabilityTestPtr> workability_tests;

	// common code for workability tests goes here
	static bool test_LU(SquareMatrix& A, std::string test_num,
		bool print_a = 0, bool print_lu = 0, bool print_res = 0);

	// workability tests

	static bool test1();
	static bool test2();
	static bool test3();
	static bool test4();

	// optimization and execution time test, n - size of square matrix, how_many_times - number of runs
	static void test_time(size_t _n, size_t how_many_times = 1);

#if defined REFERENCE_TEST && (REFERENCE_TEST==eigen || REFERENCE_TEST==mkl)
	// internal function for test_time, matpoint - pointer to inited matrix, 
	// which can be used in single_reference_test function.
	static ReturnedResults single_test_time(size_t n, size_t iter, SquareMatrix*& A);
#if REFERENCE_TEST == eigen
	// internal function for test_time, measures execution time using Eigen::PartialPivLU<Eigen::MatrixXd>
	static ReturnedResults single_reference_test(size_t n, size_t iter, const SquareMatrix& A);
#elif REFERENCE_TEST == mkl
	// internal function for test_time, measures execution time using LAPACKE_dgetrf from mkl
	static ReturnedResults single_reference_test(size_t n, size_t iter, const SquareMatrix& A);
#endif
#else
	// internal function for test_time
	static ReturnedResults single_test_time(size_t n, size_t iter);
#endif

	// output functions

	static void print_test_start(std::string s = "");
	static void print_test_end(std::string s = "");

	static std::ofstream file_out;
	static std::ostream* out;

	template<typename T>
	static inline void print(const T& value) { *out << value; }

	static inline void p_endl() { *out << std::endl; }

	// function that interprets the input number as the matrix condition number
	static void analyze_cond(double cond);
public:
	// function that adds all workability tests to workability_tests
	static void enable_workability_tests();
	// disables result verification in test_time
	static void disable_accuracy_check();
	// enables initialization with random values
	static void enable_random_initialization();

	static void run_all_tests(size_t n = 5000, size_t count = 1, std::string filename = "");
};