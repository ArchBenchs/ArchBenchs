#include "tests.h"

int main()
{
	// здесь путь прописан так, потому что build 
	// находится в директории проекта (build в gitignore)

	// C:/Users/Maxim/source/repos/practice/SECOND_COURSE/FOR_ITLAB/ArchBenchs/DecompositionLU

	TestSystem::run_all_tests("C:/Users/stud-itmm/Desktop/for_del/ArchBenchs/DecompositionLU/docs/last_output.txt");
	return 0;
}