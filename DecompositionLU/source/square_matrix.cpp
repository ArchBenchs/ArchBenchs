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
	/*if (i < 0 || i >= size || j < 0 || j >= size)
		throw out_of_range("Index should be between 0 and matrix size");*/
	return array[i * size + j];
}
const Type& SquareMatrix::at(size_t i, size_t j) const {
	/*if (i < 0 || i >= size || j < 0 || j >= size)
		throw out_of_range("Index should be between 0 and matrix size");*/
	return array[i * size + j];
}

SquareMatrix SquareMatrix::operator+(const SquareMatrix& m) {
	SquareMatrix res(m);
#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < m.size; i++) {
		for (size_t j = 0; j < m.size; j++) {
			res.at(i, j) += this->at(i, j);
		}
	} return res;
}

SquareMatrix SquareMatrix::operator*(const SquareMatrix& m) 
{
	const size_t n = size;
	const size_t block_size = 64;

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

SquareMatrix SquareMatrix::recursive_mult(const SquareMatrix& m) {
	const size_t n = size;
	SquareMatrix res(n);
	if (n == 2) {
		Type* res_arr = res.array;
		Type* this_arr = this->array;
		Type* m_arr = m.array;
		#pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			Type* this_row = this_arr + i * n;
			Type* res_row = res_arr + i * n;
			for (int k = 0; k < n; ++k) {
				Type this_val = this_row[k];
				Type* m_row = m_arr + k * n;
				#pragma omp simd
				for (int j = 0; j < n; ++j) {
					res_row[j] += this_val * m_row[j];
				}
			}
		}
		return res;
	} else {
		SquareMatrix
			A11 = this->crop(0, 0, n), A12 = this->crop(n, 0, n),
			A21 = this->crop(0, n, n), A22 = this->crop(n, n, n);
		SquareMatrix
			B11 = m.crop(0, 0, n), B12 = m.crop(n, 0, n),
			B21 = m.crop(0, n, n), B22 = m.crop(n, n, n);
		return SquareMatrix(
			A11.recursive_mult(B11) + A12.recursive_mult(B21),
			A11.recursive_mult(B12) + A12.recursive_mult(B22),
			A21.recursive_mult(B11) + A22.recursive_mult(B21),
			A21.recursive_mult(B12) + A22.recursive_mult(B22)
		);
	}
}
SquareMatrix SquareMatrix::crop(size_t csi, size_t rsi, size_t sz) const {
	Type* arr = new Type[sz * sz];
	size_t thsz = this->size;
	Type* tharr = this->array + rsi * thsz + csi;
	for (size_t i = 0; i < sz; ++i) {
		Type* this_str = tharr + i * thsz;
		Type* arr_str = arr + i * sz;
		std::copy(this_str, this_str + sz, arr_str);
	} return SquareMatrix(sz, arr);
}
SquareMatrix::SquareMatrix(const SquareMatrix& A11, const SquareMatrix& A12,
	const SquareMatrix& A21, const SquareMatrix& A22) 
{
	size_t sms = A11.size, bgs = sms * 2;
	size = bgs; array = new Type[bgs * bgs];

	Type* A11_st_str = array;
	Type* A12_st_str = array + sms;
	Type* A21_st_str = array + bgs * sms;
	Type* A22_st_str = A21_st_str + sms;

	Type* A11_ld_str = A11.array;
	Type* A12_ld_str = A12.array;
	Type* A21_ld_str = A21.array;
	Type* A22_ld_str = A22.array;

	for (size_t i = 0; i < sms - 1; ++i) {
		std::copy(A11_ld_str, A11_ld_str + sms, A11_st_str); A11_ld_str += sms; A11_st_str += bgs;
		std::copy(A12_ld_str, A12_ld_str + sms, A12_st_str); A12_ld_str += sms; A12_st_str += bgs;
		std::copy(A21_ld_str, A21_ld_str + sms, A21_st_str); A21_ld_str += sms; A21_st_str += bgs;
		std::copy(A22_ld_str, A22_ld_str + sms, A22_st_str); A22_ld_str += sms; A22_st_str += bgs;
	}
	std::copy(A11_ld_str, A11_ld_str + sms, A11_st_str);
	std::copy(A12_ld_str, A12_ld_str + sms, A12_st_str);
	std::copy(A21_ld_str, A21_ld_str + sms, A21_st_str);
	std::copy(A22_ld_str, A22_ld_str + sms, A22_st_str);
}

bool SquareMatrix::operator==(const SquareMatrix& m) {
	if (size != m.size) return false;
	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			size_t index = i * size + j;

			Type a = array[index];
			Type b = m.array[index];

			if (std::fabs(a - b) <= 1e-12) continue;

			Type max_val = std::max(std::fabs(a), std::fabs(b));
			if (max_val > 1e-12 && std::fabs(a - b) / max_val <= 1e-10) continue;

			return false;
		}
	}
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

void print_LU(const SquareMatrix& m, ostream& out) {
	const size_t n = m.size;
	out << "Matrix L:\n";
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			if (j < i) out << m.array[i * n + j];
			else out << (int)(i == j);
			out << " ";
		}
		out << endl;
	} out << endl;
	out << "Matrix U:\n";
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			if (j >= i) out << m.array[i * n + j];
			else out << 0;
			out << " ";
		}
		out << endl;
	} out << endl;
}
istream& operator>>(istream& istr, SquareMatrix& m) {
	size_t n = m.size;
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			istr >> m.array[i * n + j];
	return istr;
}
ostream& operator<<(ostream& ostr, const SquareMatrix& m) noexcept {
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
#pragma omp parallel for //schedule(dynamic, 8)
		for (int i = k + 1; i < size; i++) {
			Type* A_k_p = A_ik_p + i * size;
			Type* A_irow = m + i * size;
			(*A_k_p) /= A_kk;
#pragma omp simd //safelen(16)
			for (int j = k + 1; j < size; j++)
				A_irow[j] -= (*A_k_p) * U_ki_p[j];
		}
	}
}