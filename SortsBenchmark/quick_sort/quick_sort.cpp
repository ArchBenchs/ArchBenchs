#include <algorithm>
#include <omp.h>

#define INSERTION_THRESHOLD (int64_t)96
#define SEQUENTIAL_THRESHOLD (int64_t)75000


void insertion_sort(double* arr, int64_t left, int64_t right) {
	for (int i = left + 1; i <= right; i++) {
		double key = arr[i];
		int64_t j = i - 1;

		while (j >= left && arr[j] > key) {
			arr[j + 1] = arr[j];
			j--;
		}

		arr[j + 1] = key;
	}
}


int64_t partition(double* arr, int64_t left, int64_t right) {
	const int64_t mid = (left + right) / 2;

	if (arr[mid] < arr[left])
		std::swap(arr[left], arr[mid]);

	if (arr[right] < arr[left])
		std::swap(arr[left], arr[right]);

	if (arr[right] < arr[mid])
		std::swap(arr[right], arr[mid]);

	const double pivot = arr[mid];

	int64_t i = left - 1, j = right + 1;


	while (true) {
		do
			i++;
		while (arr[i] < pivot);

		do
			j--;
		while (arr[j] > pivot);


		if (i >= j)
			return j;

		std::swap(arr[i], arr[j]);
	}
}


void quick_sort(double* arr, int64_t left, int64_t right) {
	int64_t size = right - left;

	if (size <= 0) 
		return;
	
	if (size < INSERTION_THRESHOLD) {
		insertion_sort(arr, left, right);
		return;
	}

	else if (size < SEQUENTIAL_THRESHOLD) {
		int64_t partition_index = partition(arr, left, right);

		if (partition_index - left < right - partition_index) {
			quick_sort(arr, left, partition_index);
			quick_sort(arr, partition_index + 1, right);
		}

		else {
			quick_sort(arr, partition_index + 1, right);
			quick_sort(arr, left, partition_index);
		}
	}

	else {
		int64_t partition_index = partition(arr, left, right);

		#pragma omp task
		quick_sort(arr, left, partition_index);

		#pragma omp task
		quick_sort(arr, partition_index + 1, right);

		#pragma omp taskwait
	}
}


void parallel_quick_sort(double* arr, int64_t size) {
	#pragma omp parallel
	{
		#pragma omp single
		{
			quick_sort(arr, 0, size - 1);
		}
	}
}
