#include "tests.h"

int main()
{
	// здесь путь прописан так, потому что build 
	// находится в директории проекта (build в gitignore)

	TestSystem::run_all_tests("../../docs/last_output.txt");
	return 0;
}