#include "tests.h"
using namespace std;

int main(int argc, char* argv[])
{
	size_t arg1 = (argc == 1) ? 5000 : stoull(argv[1]);
	size_t arg2 = (argc == 1) ? 1 : stoull(argv[2]);
	TestSystem::run_all_tests(arg1, arg2, "../../docs/last_output.txt");
	return 0;
}