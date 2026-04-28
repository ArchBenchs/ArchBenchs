#include <algorithm>
#include <omp.h>
#include <iostream>

// порог, с которого начинает запускаться сортировка вставками
#define INSERTION_THRESHOLD (int64_t)128

// порог, с которого начинает запускаться последовательная версия
#define SEQUENTIAL_THRESHOLD (int64_t)100000


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


inline int64_t min(int64_t first, int64_t second) {
	if (first < second)
		return first;
	return second;
}

inline int64_t max(int64_t first, int64_t second) {
	if (first > second)
		return first;
	return second;
}

double sum = 0.0;
uint64_t counter = 0;

void quick_sort(double* arr, int64_t left, int64_t right) {
	uint64_t size = right - left;

	if (size <= 0) 
		return;
	
	if (size < INSERTION_THRESHOLD) {
		insertion_sort(arr, left, right);
		return;
	}

	else if (size < SEQUENTIAL_THRESHOLD) {
		int64_t partition_index = partition(arr, left, right);

		int64_t left_size = partition_index - left + 1;
		int64_t right_size = right - partition_index;

		if (left_size > 0 && right_size > 0) {
			double balance = (double)min(left_size, right_size) / max(left_size, right_size);
			sum += balance;
			counter++;
		}


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

		int64_t left_size = partition_index - left + 1;
		int64_t right_size = right - partition_index;

		if (left_size > 0 && right_size > 0) {
			double balance = (double)min(left_size, right_size) / max(left_size, right_size);
			sum += balance;
			counter++;
		}

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

	std::cout << sum / counter << std::endl;
}
