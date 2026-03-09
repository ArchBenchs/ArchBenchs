#include "tests.h"
#include <iostream>
#include <typeinfo> 
using namespace std;

int main(int argc, char* argv[])
{
	string fname; 
	size_t size, count;
	bool hsz = false, hcnt = false;
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
			size = stoull(argv[i + 1]); hsz = true; ++i;
		}
		else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc){
			count = stoull(argv[i + 1]); hcnt = true; ++i;
		}
		else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
			fname = argv[i + 1]; ++i;
		}
		else if (strcmp(argv[i], "--workability_tests") == 0) {
			TestSystem::enable_workability_tests();
		}
		else if (strcmp(argv[i], "--help") == 0) {
			cout << "Options:\n";
			cout << " --help                 Show help.\n";
			cout << " --size [VALUE]         Set size of matrix used in time tests equal to VALUE.\n";
			cout << " --count [VALUE]        Program do the time test VALUE times (new matrix every time).\n";
			cout << " --out [PATH]           All the program output will be write in file [PATH] (.txt file).\n";
			cout << " --workability_tests    Enables workability tests.\n";
			return 0;
		}
	}
	size_t arg1 = (hsz) ? size : 1000;
	size_t arg2 = (hcnt) ? count : 1;

	cout << "Requires " << (double)(arg1 * arg1 * sizeof(Type)) / 268435456.0 << "Gb of RAM" << endl;
	cout << "Testing with values type: " << typeid(Type).name() << endl;

	//string fname = "../../docs/last_output.txt";
	TestSystem::run_all_tests(arg1, arg2, fname);
	return 0; 
}