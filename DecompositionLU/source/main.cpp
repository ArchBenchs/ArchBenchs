#include "tests.h"
#include <iostream>
#include <typeinfo> 
using namespace std;

static double bytes_to_Gb(double val) {
	return val / 268435456.0;
}

int main(int argc, char* argv[])
{
	string fname; 
	size_t size, count;
	bool hsz = false, hcnt = false;
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--help") == 0) {
			cout << "CMD Options:\n";
			cout << "  --help					Show help.\n";
			cout << "  --size [VALUE]				Set size of matrix used in time tests equal to VALUE.\n";
			cout << "  --count [VALUE]				Program do the time test VALUE times (new matrix every time).\n";
			cout << "  --out [PATH]					All the program output will be write in file [PATH] (.txt file).\n";
			cout << "  --workability_tests (--wt)			Enables workability tests.\n";
			cout << "  --disable_accuracy_check (--dac)		Disables result checking in time-measuring tests.\n";
			cout << "  --random_initialization (--ri)		Enables random matrix initialization (in range [1e-6; 1e6]).\n";
			cout << "Compilation options:\n";
			cout << "  -DBLOCK_SIZE=[VALUE]				Set size of block used in block LU algorithm (default 64).\n";
			cout << "  -DPRINT_BLOCK_TIMES=1				Enables measuring part's times of block LU algorithm:\n"
				<< "							L11xU11: A11 -> L11 and U11;\n"
				<< "							L21: A21->L21;\n"
				<< "							U12: A12->U12;\n"
				<< "							L22xU22: A22->L22* U22.\n";
			cout << "  -REFERENCE_TEST=[ eigen | mkl ]		Set library to compare results with. Works with exactly same matrixes.\n";
			return 0;
		}
		else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
			size = stoull(argv[i + 1]); hsz = true; ++i;
		}
		else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc){
			count = stoull(argv[i + 1]); hcnt = true; ++i;
		}
		else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
			fname = argv[i + 1]; ++i;
		}
		else if (strcmp(argv[i], "--wt") == 0 ||
			strcmp(argv[i], "--workability_tests") == 0) 
		{
			TestSystem::enable_workability_tests();
		}
		else if (strcmp(argv[i], "--dac") == 0 ||
			strcmp(argv[i], "--disable_accuracy_check") == 0) 
		{
			TestSystem::disable_accuracy_check();
		}
		else if (strcmp(argv[i], "--ri") == 0 ||
			strcmp(argv[i], "--random_initialization")) 
		{
			TestSystem::enable_random_initialization();
		}
	}
	cout << "Use \"--help\" to see additional options.\n\n";
	size_t arg1 = (hsz) ? size : 1000;
	size_t arg2 = (hcnt) ? count : 1;

	cout << "Requires " << bytes_to_Gb((double)(arg1 * arg1 * sizeof(Type))) << "Gb of RAM" << endl;
	cout << "Testing with values type: " << typeid(Type).name() << endl;

	TestSystem::run_all_tests(arg1, arg2, fname);
	return 0; 
}