#include "tests.h"
#include "decomposer_lu.h"

// ----------------------------------------< workability tests >---------------------------------------------------

std::vector<WorkabilityTestPtr> TestSystem::workability_tests;

bool TestSystem::test_LU(SquareMatrix& A, std::string test_num,
	bool print_a, bool print_lu, bool print_res) {

	const size_t n = A.get_size();
	SquareMatrix LU(A);
	DecomposerLU::block_get_LU(LU.get_array(), n, n);
	SquareMatrix L(n), U(n);
	DecomposerLU::decompose_LU(LU, L, U);
	SquareMatrix Res = L * U;
	double infinite_cond_A = (Res - A).get_infinite_norm() /
		(A.get_infinite_norm() * SquareMatrix::mashine_eps);

	print_test_start(test_num);
	analyze_cond(infinite_cond_A); p_endl(); p_endl();
	if (print_a) { print("Matrix A:\n"); print(A); p_endl(); }
	if (print_lu) { DecomposerLU::print_LU(LU, *out); }
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
	SquareMatrix A(n, -1e6, 1e6);
	return test_LU(A, "3", 1, 1, 1);
}

bool TestSystem::test4() {
	const size_t n = 100;
	SquareMatrix A(n, -1e6, 1e6);
	return test_LU(A, "4");
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< time tests >----------------------------------------------------------

#ifndef TP
#define TP steady_clock::time_point 
#endif
#ifndef NOW
#define NOW steady_clock::now()
#endif

bool TestSystem::do_accuracy_check = true;
bool TestSystem::random_initialization = false;

#if defined REFERENCE_TEST && (REFERENCE_TEST == eigen || REFERENCE_TEST == mkl)
ReturnedResults TestSystem::single_test_time(size_t n, size_t iter, SquareMatrix*& A) {
	ReturnedResults results;

	TP start_init = NOW;
	if (random_initialization) { A = new SquareMatrix(n, 1e-6, 1e6); }
	else { A = new SquareMatrix(n, true); }
	SquareMatrix LU(*A);

	results.InitTime = duration_cast<milliseconds>(NOW - start_init);

	TP start_LU = NOW;
	DecomposerLU::block_get_LU(LU.get_array(), n, n);
	results.LUTime = duration_cast<milliseconds>(NOW - start_LU);

	results.TotalTime = duration_cast<milliseconds>(NOW - start_init);

	if (do_accuracy_check) {
		SquareMatrix L(n), U(n);
		DecomposerLU::decompose_LU(LU, L, U);
		SquareMatrix Res = L * U;
		double infinite_cond_A = (Res - *A).get_infinite_norm() /
			(A->get_infinite_norm() * SquareMatrix::mashine_eps);

		results.is_correct = (*A == Res);

		p_endl();
		print(iter + 1); print(") ");
		analyze_cond(infinite_cond_A);
		print(" Test result: ");
		bool consol = (out == &cout);
		if (results.is_correct) { if (consol) *out << "\033[32m"; print("true"); }
		else { if (consol) *out << "\033[31m"; print("false"); }
		if (consol) *out << "\033[0m";
		print(". LU Time: ");
		print(results.LUTime.count());
		p_endl();
	}
	else { results.is_correct = false; }
	return results;
}
#if REFERENCE_TEST == eigen

#include <Eigen/Dense>      
#include <Eigen/LU>    
ReturnedResults TestSystem::single_reference_test(size_t n, size_t iter, const SquareMatrix& sqmtr) {
	ReturnedResults results;
	Eigen::MatrixXd A(n, n);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			A(i, j) = sqmtr(i, j);
		}
	}

	TP start_LU = NOW;
	Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
	results.LUTime = duration_cast<milliseconds>(NOW - start_LU);

	if (do_accuracy_check) {
		results.is_correct = true;
		bool consol = (out == &cout);
		if (consol) { *out << "\033[33m"; }
		print("   Reference test "); print(iter + 1);
		print(". LU Time: ");
		print(results.LUTime.count());
		if (consol) { *out << "\033[0m"; }
		p_endl();
	}
	else { results.is_correct = false; }
	return results;
}

#else

#include <mkl_lapacke.h>
ReturnedResults TestSystem::single_reference_test(size_t n, size_t iter, const SquareMatrix& sqmtr) {
	ReturnedResults results;
	SquareMatrix A(sqmtr);
	int64_t* ipiv_ptr = new int64_t[n];

	TP start_LU = NOW;
	int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A.get_array(), n, ipiv_ptr);
	results.LUTime = duration_cast<milliseconds>(NOW - start_LU);

	if (do_accuracy_check) {
		results.is_correct = true;
		bool consol = (out == &cout);
		if (consol) { *out << "\033[33m"; }
		print("   Reference test "); print(iter + 1);
		print(". LU Time: ");
		print(results.LUTime.count());
		if (consol) { *out << "\033[0m"; }
		p_endl();
	}
	else { results.is_correct = false; }
	return results;
}

#endif

#else
ReturnedResults TestSystem::single_test_time(size_t n, size_t iter) {
	ReturnedResults results;
	TP start_init = NOW;

	SquareMatrix A(n);
	if (random_initialization) { A = SquareMatrix(n, 1e-6, 1e6); }
	else { A = SquareMatrix(n, true); }
	SquareMatrix LU(A);

	results.InitTime = duration_cast<milliseconds>(NOW - start_init);

	TP start_LU = NOW;
	DecomposerLU::block_get_LU(LU.get_array(), n, n);
	results.LUTime = duration_cast<milliseconds>(NOW - start_LU);

	results.TotalTime = duration_cast<milliseconds>(NOW - start_init);

	if (do_accuracy_check) {
		SquareMatrix L(n), U(n);
		DecomposerLU::decompose_LU(LU, L, U);
		SquareMatrix Res = L * U;
		double infinite_cond_A = (Res - A).get_infinite_norm() /
			(A.get_infinite_norm() * SquareMatrix::mashine_eps);

		results.is_correct = (A == Res);

		p_endl();
		print(iter + 1); print(") ");
		analyze_cond(infinite_cond_A);
		print(" Test result: ");
		bool consol = (out == &cout);
		if (results.is_correct) { if (consol) *out << "\033[32m"; print("true"); }
		else { if (consol) *out << "\033[31m"; print("false"); }
		if (consol) *out << "\033[0m";
		print(". LU Time: ");
		print(results.LUTime.count());
		p_endl();
	}
	else { results.is_correct = false; }
	return results;
}
#endif

void TestSystem::test_time(size_t _n, size_t how_many_times) {
	print_test_start("time");
	chrono::milliseconds time_init{ 1000000000 }, total_time{ 1000000000 }, time_LU{ 1000000000 };
#if defined REFERENCE_TEST && (REFERENCE_TEST == eigen || REFERENCE_TEST == mkl)
	chrono::milliseconds time_LU_ref{ 1000000000 };
#endif
	const size_t n = _n;
	double cc = 0, incc = 0;
	print("Testing with n = "); print(_n); print(", ");
	print(how_many_times); print(" times:"); p_endl();
	for (size_t iter = 0; iter < how_many_times; ++iter) {
#if defined REFERENCE_TEST && (REFERENCE_TEST == eigen || REFERENCE_TEST == mkl)
		SquareMatrix* A = new SquareMatrix(n);
		ReturnedResults res = single_test_time(n, iter, A);
#else
		ReturnedResults res = single_test_time(n, iter);
#endif
		time_init = (time_init > res.InitTime) ? res.InitTime : time_init;
		time_LU = (time_LU > res.LUTime) ? res.LUTime : time_LU;
		total_time = (total_time > res.TotalTime) ? res.TotalTime : total_time;
		if (do_accuracy_check) {
			if (res.is_correct) { ++cc; }
			else { ++incc; }
		}
#if defined REFERENCE_TEST && (REFERENCE_TEST == eigen || REFERENCE_TEST == mkl)
		ReturnedResults resref = single_reference_test(n, iter, *A);
		time_LU_ref = (time_LU_ref > resref.LUTime) ? resref.LUTime : time_LU_ref;
#endif
	}
	print("\nMinimum time for init random matrix: ");
	print(time_init.count());

	print(" ms\nMinimum time for LU decomposition: ");
	print(time_LU.count());
#if defined REFERENCE_TEST && (REFERENCE_TEST == eigen || REFERENCE_TEST == mkl)
	print(" ms\nMinimum time for reference LU decomposition: ");
	print(time_LU_ref.count());
#endif
	print(" ms\nMinimum total time: ");
	print(total_time.count());

	if (do_accuracy_check) {
		print(" ms\n\nTotal test result: "); print(cc / (cc + incc) * 100);
		print("%\nCorrect count: "); print(cc);
		print("\nIncorrect count: "); print(incc);
	}
	print("\n-------------------------------------------------------------------------------------------------\n");
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< printing >------------------------------------------------------------

std::ofstream TestSystem::file_out;
std::ostream* TestSystem::out = &std::cout;

void TestSystem::print_test_start(std::string s) {
	print("\n------------------------------------------- Test ");
	print(s); print(" -------------------------------------------");
	p_endl();
}

void TestSystem::print_test_end(std::string s) {
	print("----------------------------------------------------------------------------------------------\nTest");
	print(s); print(": ");
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< other functions >----------------------------------------------------

void TestSystem::enable_workability_tests() {
	workability_tests.push_back(TestSystem::test1);
	workability_tests.push_back(TestSystem::test2);
	workability_tests.push_back(TestSystem::test3);
	workability_tests.push_back(TestSystem::test4);
}

void TestSystem::disable_accuracy_check() { do_accuracy_check = false; }

void TestSystem::enable_random_initialization() { random_initialization = true; }

void TestSystem::run_all_tests(size_t n, size_t count, std::string filename) {
	if (filename != "") {
		file_out.open(filename);
		if (!file_out.is_open()) {
			std::cerr << "Failed to open file: " << filename << std::endl;
			out = &std::cout;
		}
		else { out = &file_out; }
	}
	print("\nTestSystem:\n");
	bool last_res;
	for (auto TestPtr : workability_tests) {
		last_res = (*TestPtr)();
		bool consol = (out == &cout);
		if (consol) {
			if (last_res) *out << "\033[32m";
			else *out << "\033[31m";
		}
		print((last_res) ? "true\n\n" : "false\n\n");
		if (consol) *out << "\033[0m";
	} p_endl();
	test_time(n, count);

	if (filename != "") { file_out.close(); }
}

enum cond_quality { good, ill, singular };
void TestSystem::analyze_cond(double cond) {
	print("Infinite cond(A): "); print(cond); print("; ");
	cond_quality cq;
	if (cond < 1e+3) { cq = cond_quality::good; }
	else {
		if (cond >= 1e+3 && cond <= 1e+6) { cq = cond_quality::ill; }
		else { cq = cond_quality::singular; }
	}
	bool consol = out == &cout;
	switch (cq) {
	case cond_quality::good:
		if (consol) { *out << "\033[32m"; }
		print("Matrix is good-conditioned."); break;
	case cond_quality::ill:
		if (consol) { *out << "\033[33m"; }
		print("WARNING! Matrix is ill-conditioned."); break;
	case cond_quality::singular:
		if (consol) { *out << "\033[31m"; }
		print("CRITICAL! Matrix is singular-conditioned!"); break;
	} if (consol) { *out << "\033[0m"; }
}

// ----------------------------------------------------------------------------------------------------------------
