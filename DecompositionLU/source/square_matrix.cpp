#include "square_matrix.h"
#include <omp.h>

SquareMatrix::SquareMatrix(size_t s, Type* in_arr) {
	size = s;
	array = new Type[size * size]{};
	if (in_arr != nullptr)
		std::copy(in_arr, in_arr + size * size, array);
}
SquareMatrix::~SquareMatrix() { delete[] array; }

SquareMatrix::SquareMatrix(const SquareMatrix& m) {
	size = m.size;
	array = new Type[size * size];
	memcpy(array, m.array, size * size * TypeSize);
}
SquareMatrix& SquareMatrix::operator=(const SquareMatrix& m) {
	if (this == &m) return *this;
	delete[] array;
	size = m.size;
	array = new Type[size * size];
	memcpy(array, m.array, size * size * TypeSize);
	return *this;
}

SquareMatrix::SquareMatrix(SquareMatrix&& m) noexcept {
	size = m.size;
	array = m.array;
	m.size = 0;
	m.array = nullptr;
}
SquareMatrix& SquareMatrix::operator=(SquareMatrix&& m) noexcept {
	if (this == &m) return *this;
	delete[] array;
	size = m.size;
	array = m.array;
	m.size = 0;
	m.array = nullptr;
	return *this;
}

Type& SquareMatrix::at(size_t i, size_t j) {
	if (i < 0 || i >= size || j < 0 || j >= size)
		throw out_of_range("Index should be between 0 and matrix size");
	return array[i * size + j];
}
const Type& SquareMatrix::at(size_t i, size_t j) const {
	if (i < 0 || i >= size || j < 0 || j >= size)
		throw out_of_range("Index should be between 0 and matrix size");
	return array[i * size + j];
}

SquareMatrix SquareMatrix::operator*(const SquareMatrix& m) {
	SquareMatrix res(size);
	const size_t n = size;

	Type* res_arr = res.array;
	Type* this_arr = this->array;
	Type* m_arr = m.array;

	#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < n; ++i) {
		//Type* this_row = this_arr + i * n;
		//Type* res_row = res_arr + i * n;
		for (size_t k = 0; k < n; ++k) {
			//Type this_val = this_row[k];
			//Type* m_row = m_arr + k * n;
			#pragma omp simd
			for (size_t j = 0; j < n; ++j) {
				res_arr[i * n + j] += this_arr[i * n + k] * m_arr[k * n + j];
				/*res_arr[i * n + j] += this_arr[i * n + k] * m_arr[k * n + j];*/
			}
		}
	}
	return res;
}

bool SquareMatrix::operator==(const SquareMatrix& m) {
	if (size != m.size) return false;
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++)
			if (array[i * size + j] != m.array[i * size + j]) return false;
	return true;
}

void SquareMatrix::decompose_LU(SquareMatrix& L, SquareMatrix& U) {
	const size_t n = size;
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++) {
			if (j < i) L.at(i, j) = array[i * n + j];
			if (j == i) L.at(i, j) = 1;
		}
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			if (j >= i) U.at(i, j) = array[i * n + j];
}

void print_LU(const SquareMatrix& m) {
	const size_t n = m.size;
	cout << "Matrix L:\n";
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			if (j < i) cout << m.array[i * n + j];
			else cout << (int)(i == j);
			cout << " ";
		}
		cout << endl;
	}
	cout << "Matrix U:\n";
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			if (j >= i) cout << m.array[i * n + j];
			else cout << 0;
			cout << " ";
		}
		cout << endl;
	}
}
istream& operator>>(istream& istr, SquareMatrix& m) {
	size_t n = m.size;
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			istr >> m.array[i * n + j];
	return istr;
}
ostream& operator<<(ostream& ostr, SquareMatrix& m) noexcept {
	size_t n = m.size;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++)
			ostr << m.array[i * n + j] << " ";
		ostr << endl;
	}
	return ostr;
}

 void get_LU(SquareMatrix& matrix_pointer) {
	Type*& m = matrix_pointer.get_array();
	const size_t size = matrix_pointer.get_size();
	size_t k_iter_max = size - 1;
	for (size_t k = 0; k < k_iter_max; k++) {
		Type* A_ik_p = m + k;
		Type* U_ki_p = m + k * size;
		Type A_kk = m[k * size + k];
		#pragma omp parallel for schedule(dynamic, 8)
		for (int i = k + 1; i < size; i++) {
			Type* A_k_p = A_ik_p + i * size;
			Type* A_irow = m + i * size;
			(*A_k_p) /= A_kk;
			#pragma omp simd
			for (int j = k + 1; j < size; j++)
				A_irow[j] -= (*A_k_p) * U_ki_p[j];
		}
	}
}