#include "square_matrix.h"
#include "limits.h"
#include <omp.h>

// ----------------------------------------< allocation >----------------------------------------------------------

#ifdef _WIN32
#include <malloc.h>  
#endif

static inline size_t round_up(size_t n, size_t align) {
	return (n + align - 1) & ~(align - 1);
}

void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
	return _aligned_malloc(size, alignment);
#else
	size_t alloc_size = round_up(size, alignment);
	return aligned_alloc(alignment, alloc_size);
#endif
}
void aligned_free(void* ptr) {
#ifdef _WIN32
	_aligned_free(ptr);
#else
	free(ptr);
#endif
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< constructors & destructor >-------------------------------------------

SquareMatrix::SquareMatrix(size_t s, bool init_diag_dominant) {
	size = s;
	size_t sz_sqr = size * size;
	size_t bytes = sz_sqr * TypeSize;

	array = (Type*)aligned_malloc(bytes, BLOCK_SIZE);
	if (!array) throw bad_alloc();

	if (init_diag_dominant) {
		Type val = static_cast<Type>(sz_sqr);
		std::memset(array, 1, bytes);
		for (size_t i = 0; i < size; ++i) {
			array[i * size + i] = val;
		}
	}
	else { std::memset(array, 0, bytes); }
}
SquareMatrix::SquareMatrix(size_t s, Type* in_arr) {
	size = s;
	size_t bytes = size * size * TypeSize;

	array = (Type*)aligned_malloc(bytes, BLOCK_SIZE);
	if (!array) throw bad_alloc();

	if (in_arr != nullptr) { std::memcpy(array, in_arr, bytes); }
	else throw bad_alloc();//std::invalid_argument("Initial array is nullptr!");
}
SquareMatrix::SquareMatrix(size_t s, Type min, Type max) {
	size = s;
	size_t bytes = size * size * TypeSize;

	array = (Type*)aligned_malloc(bytes, BLOCK_SIZE);
	if (!array) throw bad_alloc();

	random_device rd; mt19937 gen(rd());
	uniform_real_distribution<Type> common_generator(min, max);
	uniform_real_distribution<Type> positive_generator(0, max / 2);

	for (size_t i = 0; i < s; i++) {
		Type sum = 0;
		Type* str = array + i * size;
		for (size_t j = 0; j < s; j++) {
			Type val = common_generator(gen);
			str[j] = val; sum += fabs(val);
		}
		str[i] = sum + positive_generator(gen);
	}
}

SquareMatrix::~SquareMatrix() { aligned_free(array); }

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< copy semantics >------------------------------------------------------

SquareMatrix::SquareMatrix(const SquareMatrix& m) {
	size = m.size;
	size_t bytes = size * size * TypeSize;

	array = (Type*)aligned_malloc(bytes, BLOCK_SIZE);
	if (!array) throw bad_alloc();

	memcpy(array, m.array, bytes);
}
SquareMatrix& SquareMatrix::operator=(const SquareMatrix& m) {
	if (this == &m) return *this;
	aligned_free(array);

	size = m.size;
	size_t bytes = size * size * TypeSize;

	array = (Type*)aligned_malloc(bytes, BLOCK_SIZE);
	if (!array) throw bad_alloc();

	memcpy(array, m.array, bytes);
	return *this;
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< move semantics >------------------------------------------------------

SquareMatrix::SquareMatrix(SquareMatrix&& m) noexcept {
	size = m.size;
	array = m.array;
	m.size = 0;
	m.array = nullptr;
}
SquareMatrix& SquareMatrix::operator=(SquareMatrix&& m) noexcept {
	if (this == &m) return *this;
	aligned_free(array);
	size = m.size;
	array = m.array;
	m.size = 0;
	m.array = nullptr;
	return *this;
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< arithmetic >----------------------------------------------------------

SquareMatrix SquareMatrix::operator+(const SquareMatrix& m) {
	SquareMatrix res(m);
#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < m.size; i++) {
		for (size_t j = 0; j < m.size; j++) {
			res(i, j) += this->operator()(i, j);
		}
	} return res;
}

SquareMatrix SquareMatrix::operator-(const SquareMatrix& m) {
	SquareMatrix res(m);
#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < m.size; i++) {
		for (size_t j = 0; j < m.size; j++) {
			res(i, j) -= this->operator()(i, j);
		}
	} return res;
}

SquareMatrix SquareMatrix::operator*(const SquareMatrix& m)
{
	const size_t n = size;
	const size_t block_size = BLOCK_SIZE;

	SquareMatrix res(n);

	Type* res_arr = res.array;
	Type* this_arr = this->array;
	Type* m_arr = m.array;

#pragma omp parallel for collapse(2)
	for (int i0 = 0; i0 < n; i0 += block_size) {
		for (int j0 = 0; j0 < n; j0 += block_size) {
			int i1 = std::min(i0 + block_size, n);
			int j1 = std::min(j0 + block_size, n);
			for (int k0 = 0; k0 < n; k0 += block_size) {
				int k1 = std::min(k0 + block_size, n);
				for (int i = i0; i < i1; ++i) {
					Type* this_row = this_arr + i * n;
					Type* res_row = res_arr + i * n;
					for (int k = k0; k < k1; ++k) {
						Type this_val = this_row[k];
						Type* m_row = m_arr + k * n;
#pragma omp simd
						for (int j = j0; j < j1; ++j) {
							res_row[j] += this_val * m_row[j];
						}
					}
				}
			}
		}
	}
	return res;
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< comparison >----------------------------------------------------------

// compare values using absolute error
inline bool compare_eps(const Type& arg1, const Type& arg2, const Type& eps) {
	return fabs(arg1 - arg2) <= eps;
}
// compare values using relative error
inline bool compare_rel(const Type& arg1, const Type& arg2, const Type& eps) {
	Type abs1 = fabs(arg1), abs2 = fabs(arg2);
	Type max = (abs1 > abs2) ? abs1 : abs2;
	return fabs(arg1 - arg2) <= max * eps;
}

bool SquareMatrix::operator==(const SquareMatrix& m) {
	if (size != m.size) return false;
	double eps = 1e-9;
	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			size_t index = i * size + j;

			Type a = array[index];
			Type b = m.array[index];
			if (a == b) continue;
			if ((a > -1.0 && a < 1.0) && (b > -1.0 && a < 1.0)) {
				if (compare_eps(a, b, eps)) continue;
			}
			else { if (compare_rel(a, b, eps)) continue; }

			return false;
		}
	}
	return true;
}

bool SquareMatrix::operator!=(const SquareMatrix& m) {
	return !(*this == m);
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< norm calculation >----------------------------------------------------

Type SquareMatrix::get_infinite_norm() const {
	Type curr_max = numeric_limits<Type>::lowest();
#pragma omp parallel for reduction(max:curr_max)
	for (int i = 0; i < size; ++i) {
		Type sum = 0.0;
		for (int j = 0; j < size; ++j) {
			sum += fabs(array[i * size + j]);
		}
		if (sum > curr_max) curr_max = sum;
	}
	return curr_max;
}

double SquareMatrix::get_frobenius_norm() const {
	Type sum = 0.0; size_t sz = size * size;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < sz; ++i) { sum = sum + array[i] * array[i]; }
	return sqrt(sum);
}

Type SquareMatrix::get_one_norm() const {
	Type curr_max = numeric_limits<Type>::lowest();
#pragma omp parallel reduction(max:curr_max)
	for (int i = 0; i < size; ++i) {
		Type sum = 0.0;
		for (int j = 0; j < size; ++j) {
			sum += fabs(array[i + j * size]);
		}
		curr_max = (curr_max < sum) ? sum : curr_max;
	}
	return curr_max;
}

// ----------------------------------------------------------------------------------------------------------------

// ----------------------------------------< I/O Stream >----------------------------------------------------------

istream& operator>>(istream& istr, SquareMatrix& m) {
	size_t n = m.size;
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			istr >> m(i, j);
	return istr;
}
ostream& operator<<(ostream& ostr, const SquareMatrix& m) noexcept {
	size_t n = m.size;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++)
			ostr << m(i, j) << " ";
		ostr << endl;
	}
	return ostr;
}

// ----------------------------------------------------------------------------------------------------------------
