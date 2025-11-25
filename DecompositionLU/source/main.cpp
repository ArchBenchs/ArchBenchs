#include "tests.h"
using namespace std;

int main(int argc, char* argv[])
{
	size_t arg1 = (argc == 1) ? 3000 : stoull(argv[1]);
	size_t arg2 = (argc == 1) ? 2 : stoull(argv[2]);

	//string fname = "../../docs/last_output.txt";
	string fname = "C:/Users/Maxim/source/repos/practice/SECOND_COURSE/FOR_ITLAB/ArchBenchs/DecompositionLU/docs/last_output.txt";
	TestSystem::run_all_tests(arg1, arg2, fname);
	return 0;
}