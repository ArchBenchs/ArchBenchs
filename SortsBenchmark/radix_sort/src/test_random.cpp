#include "radix_sort.cpp"
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
    const int64_t N = 10000000;

    double* Array = new double[N];

    try {
        std::ifstream file("random_data.bin", std::ios::binary);
        file.read(reinterpret_cast<char*>(Array), N * sizeof(double));
        file.close();
    }

    catch(...) {
        std::cout << "Data reading error" << std::endl;
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    radix_sort(Array, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    if (check_sorted(Array, N))
        std::cout << "Array is sorted" << std::endl;

    else
        std::cout << "Array is not sorted!!!" << std::endl;

    delete[] Array;
}


int main() {
    test_random();

    return 0;
}
