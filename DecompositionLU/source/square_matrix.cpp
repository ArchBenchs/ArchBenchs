#include "square_matrix.h"
#include "limits.h"
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
	const size_t block_size = 64;

	SquareMatrix res(n);

	Type* res_arr = res.array;
	Type* this_arr = this->array;
	Type* m_arr = m.array;

#pragma omp parallel for collapse(2)
	for (int i0 = 0; i0 < n; i0 += block_size) {
		for (int j0 = 0; j0 < n; j0 += block_size) {
			int i1 = std::min(i0 + block_size, n);
			int j1 = std::min(j0 + block_size, n);  // замерить сравнить
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

SquareMatrix SquareMatrix::old_multi(const SquareMatrix& m)
{
	const size_t n = size;

	SquareMatrix res(n);

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
}

Type SquareMatrix::get_infinite_norm() const {
	Type curr_max = numeric_limits<Type>::lowest();
	Type sum = 0.0; //Type* str = array;
#pragma omp parallel for collapse(2)
	//for (int i = 0; i < size; ++i) {
	//	for (int j = 0; j < size; ++j) { 
	//		sum += str[j];
	//	}
	//	curr_max = (curr_max < sum) ? sum : curr_max;
	//	sum = 0.0;
	//	str = array + (i + 1) * size;
	//}
	for (int i = 0; i < size; ++i) { //#pragma simd reduction(+:sum)
		for (int j = 0; j < size; ++j) {
			sum += array[i * size + j];
		}
		curr_max = (curr_max < sum) ? sum : curr_max;
		sum = 0.0;
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
	Type sum = 0.0; //Type* col = array;
#pragma omp parallel for collapse(2)
	//for (int i = 0; i < size; ++i) { //#pragma simd reduction(+:sum)
	//	for (int j = 0; j < size; ++j) {
	//		sum += col[j * size];
	//	}
	//	curr_max = (curr_max < sum) ? sum : curr_max;
	//	sum = 0.0;
	//	col = array + i;
	//}
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			sum += array[i + j * size];
		}
		curr_max = (curr_max < sum) ? sum : curr_max;
		sum = 0.0;
	}
	return curr_max;
}

void SquareMatrix::crop(size_t csi, size_t rsi, size_t sz, Type* res_arr) const {
	size_t thsz = this->size;
	Type* tharr = this->array + rsi * thsz + csi;
	for (size_t i = 0; i < sz; ++i) {
		Type* this_str = tharr + i * thsz;
		Type* arr_str = res_arr + i * sz;
		std::copy(this_str, this_str + sz, arr_str);
	}
}

bool SquareMatrix::operator==(const SquareMatrix& m) {
	if (size != m.size) return false;
	double eps = 1e-4;
	//double rel_eps = size * size * machine_eps;
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

void SquareMatrix::decompose_LU(SquareMatrix& L, SquareMatrix& U) {
	const size_t n = size;
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++) {
			if (j < i) L(i, j) = array[i * n + j];
			if (j == i) L(i, j) = 1;
		}
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			if (j >= i) U(i, j) = array[i * n + j];
}

void print_LU(const SquareMatrix& m, ostream& out) {
	const size_t n = m.size;
	out << "Matrix L:\n";
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			if (j < i) out << m(i, j);
			else out << (int)(i == j);
			out << " ";
		}
		out << endl;
	} out << endl;
	out << "Matrix U:\n";
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			if (j >= i) out << m(i, j);
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

void get_LU(SquareMatrix& matrix_pointer) {
	Type*& m = matrix_pointer.get_array();
	const size_t size = matrix_pointer.get_size();
	size_t k_iter_max = size - 1;
	for (size_t k = 0; k < k_iter_max; k++) {
		Type* A_ik_p = m + k;
		Type* U_ki_p = m + k * size;
		Type A_kk = m[k * size + k];
#pragma omp parallel for
		for (int i = k + 1; i < size; i++) {
			Type* A_k_p = A_ik_p + i * size;
			Type* A_irow = m + i * size;
			(*A_k_p) /= A_kk;
#pragma omp simd 
			for (int j = k + 1; j < size; j++) {
				A_irow[j] -= (*A_k_p) * U_ki_p[j];
			}
		}
	}
}

void block_get_LU(Type* matrix_array_p, size_t curr_sz, size_t start_sz) {
	const int block_size = 32;

	int curr_size = (int)curr_sz;
	int start_size = (int)start_sz; 
	int iter_max = start_size * start_size;
	int iter_step = block_size * start_size + block_size;

	for (int iter = 0; iter < iter_max; iter += iter_step) {
		Type* m_arr_p = matrix_array_p + iter;

		bool flag = curr_size <= block_size;
		int lim = (flag) ? curr_size : block_size;

		// L11 & U11
		size_t k_iter_max = lim - 1;
		for (size_t k = 0; k < k_iter_max; k++) {
			Type* A_ik_p = m_arr_p + k;
			Type* U_ki_p = m_arr_p + k * start_size;
			Type A_kk = m_arr_p[k * start_size + k];
#pragma omp parallel for
			for (int i = k + 1; i < lim; i++) {
				Type* A_k_p = A_ik_p + i * start_size;
				Type* A_irow = m_arr_p + i * start_size;
				(*A_k_p) /= A_kk;
#pragma omp simd 
				for (int j = k + 1; j < lim; j++) {
					A_irow[j] -= (*A_k_p) * U_ki_p[j];	
				}	
			}
		}
		if (flag) return;

#pragma omp parallel for // L21
		for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
			int i1 = std::min(i0 + block_size, curr_size);
			for (int i = i0; i < i1; ++i) {
				Type* L_ix = m_arr_p + i * start_size;
				for (int k = 0; k < block_size; k++) {
					Type* L_ik = L_ix + k;
// #pragma omp simd // - не векторизуется
					for (int j = 0; j < k; j++) {
						Type* L_ij = L_ix + j;
						Type* U_jx = m_arr_p + j * start_size;
						*L_ik -= *(L_ij) * U_jx[k];
					}
					Type* U_kx = m_arr_p + k * start_size;
					*L_ik /= U_kx[k];
				}
			}
		}
#pragma omp parallel for // U12 
		for (int j0 = block_size; j0 < curr_size; j0 += block_size) {
			int j1 = std::min(j0 + block_size, curr_size);
			for (int j = j0; j < j1; ++j) {
				Type* m_xj = m_arr_p + j;
				for (int k = 0; k < block_size; ++k) {
					Type* m_kx = m_arr_p + k * start_size;
					Type* m_kj = m_kx + j;
// #pragma omp simd // - не векторизуется
					for (int i = 0; i < k; ++i) {
						Type* m_ki = m_kx + i;
						Type* m_ij = m_xj + i * start_size;
						*m_kj -= (*m_ki) * (*m_ij);
					}
				}
			}
		}

		// L22 * U22 = A22 - L21 * U12
#pragma omp parallel for collapse(2)
		for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
			for (int j0 = block_size; j0 < curr_size; j0 += block_size) {
				int i1 = std::min(i0 + block_size, curr_size);
				int j1 = std::min(j0 + block_size, curr_size);

				for (int i = i0; i < i1; ++i) {
					Type* A22_irow = m_arr_p + i * start_size;
					Type* L_ix = m_arr_p + i * start_size;
					for (int k = 0; k < block_size; ++k) {
						Type L_ik = *(L_ix + k);
						Type* U_kx = m_arr_p + k * start_size;
#pragma omp simd
						for (int j = j0; j < j1; ++j) {
							A22_irow[j] -= L_ik * *(U_kx + j);
						}
						// энтринтики не используем, чтоб кроссплатформенный код был
					}
				}
			}
		}
		curr_size -= block_size;
	}
}
