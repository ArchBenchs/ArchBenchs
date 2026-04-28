#include <cstdint>
#include <omp.h>

using std::size_t;


inline uint64_t get_bit(uint64_t x, int pos) {
    return (x >> pos) & 1ULL;
}


inline uint64_t transform(uint64_t bits) {
    if (bits & (1ULL << 63))
        return ~bits;
    else
        return bits ^ (1ULL << 63);
}


inline uint64_t untransform(uint64_t bits) {
    if (bits & (1ULL << 63))
        return bits ^ (1ULL << 63);
    else
        return ~bits;
}


inline void counting_sort(uint64_t* arr_bits, uint64_t* temp_bits, size_t size, int bit) {
    size_t count[2] = {0, 0};
    
    for (size_t i = 0; i < size; i++)
        count[get_bit(arr_bits[i], bit)]++;
    
    count[1] += count[0];

    for (long int i = size - 1; i >= 0; i--) {
        int bit_value = get_bit(arr_bits[i], bit);
        int index = --count[bit_value];
        temp_bits[index] = arr_bits[i];
    }
}


void msd_radix_sort(uint64_t* arr_bits, uint64_t* temp_bits, size_t size, int bit) {
    if (size <= 1 || bit < 0)
        return;
    
    counting_sort(arr_bits, temp_bits, size, bit);
    
    for (size_t i = 0; i < size; i++)
        arr_bits[i] = temp_bits[i];
    

    size_t mid = 0;

    while (mid < size && get_bit(arr_bits[mid], bit) == 0)
        mid++;
    
    #pragma omp task
    msd_radix_sort(arr_bits, temp_bits, mid, bit - 1);
    
    #pragma omp task
    msd_radix_sort(arr_bits + mid, temp_bits + mid, size - mid, bit - 1);

    #pragma omp taskwait
}


void radix_sort(double* arr, size_t size) {
    if (size <= 1) 
        return;

    uint64_t* arr_bits = reinterpret_cast<uint64_t*>(arr);
    double* temp = new double[size];
    uint64_t* temp_bits = reinterpret_cast<uint64_t*>(temp);
    
    for (size_t i = 0; i < size; i++)
        arr_bits[i] = transform(arr_bits[i]);


    #pragma omp parallel
    {
        #pragma omp single
        msd_radix_sort(arr_bits, temp_bits, size, 63);
    }
    

    for (size_t i = 0; i < size; i++)
        arr_bits[i] = untransform(arr_bits[i]);
            
    delete[] temp;
}
