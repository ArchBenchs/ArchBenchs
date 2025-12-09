#include "tests.h"
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
	size_t arg1 = (argc == 1) ? 5000 : stoull(argv[1]);
	size_t arg2 = (argc == 1) ? 1 : stoull(argv[2]);

	cout << "Requires " << (double)(arg1 * arg1 * sizeof(Type)) / 1073741824 * 3 << "Gb of RAM" << endl;

	//string fname = "../../docs/last_output.txt";
	string fname = "C:/Users/Maxim/source/repos/practice/SECOND_COURSE/FOR_ITLAB/ArchBenchs/DecompositionLU/docs/last_output.txt";
	TestSystem::run_all_tests(arg1, arg2, fname);
	return 0;
}

// QWEN