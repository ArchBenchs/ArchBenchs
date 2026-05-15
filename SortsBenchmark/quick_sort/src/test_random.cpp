#include "quick_sort.cpp"
#include <chrono>
#include <iostream>
#include <fstream>


bool check_sorted(double* arr, int64_t N) {
	for (int64_t i = 1; i < N; i++)
		if (arr[i - 1] > arr[i])
			return false;

	return true;
}


void test_random() {
	auto start_total = std::chrono::high_resolution_clock::now();

	const int64_t N = 1000000000;

	double* Array = new double[N];


	auto start_read = std::chrono::high_resolution_clock::now();
	try {
		std::ifstream file("random_data.bin", std::ios::binary);
		file.read(reinterpret_cast<char*>(Array), N * sizeof(double));
		file.close();
	}

	catch(...) {
		std::cout << "Data reading error" << std::endl;
		return;
	}
	auto end_read = std::chrono::high_resolution_clock::now();

	auto start_sort = std::chrono::high_resolution_clock::now();
	parallel_quick_sort(Array, N);
	auto end_sort = std::chrono::high_resolution_clock::now();

	auto start_check = std::chrono::high_resolution_clock::now();
	bool sorted = check_sorted(Array, N);
	auto end_check = std::chrono::high_resolution_clock::now();

	auto end_total = std::chrono::high_resolution_clock::now();

	auto read_time = std::chrono::duration<double>(end_read - start_read).count();
	auto sort_time = std::chrono::duration<double>(end_sort - start_sort).count();
	auto check_time = std::chrono::duration<double>(end_check - start_check).count();
	auto total_time = std::chrono::duration<double>(end_total - start_total).count();

	std::cout << "Read time: " << read_time << " s" << std::endl;
	std::cout << "Sort time: " << sort_time << " s" << std::endl;
	std::cout << "Check time: " << check_time << " s" << std::endl << std::endl;
	std::cout << "Total time: " << total_time << " s" << std::endl;

	if (sorted)
		std::cout << "\nArray is sorted" << std::endl;

	else
		std::cout << "\nArray is not sorted!!!" << std::endl;

	delete[] Array;
}


int main() {
	test_random();

	return 0;
}
