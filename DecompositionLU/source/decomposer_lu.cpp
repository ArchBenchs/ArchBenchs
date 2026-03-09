#include "decomposer_lu.h"

void DecomposerLU::block_get_LU(Type* matrix_array_p, size_t curr_sz, size_t start_sz) {
	const int block_size = BLOCK_SIZE;
	const int kbs = LESSER_BLOCK_SIZE;

	int curr_size = (int)curr_sz;
	int start_size = (int)start_sz;
	int iter_max = start_size * start_size;
	int iter_step = block_size * start_size + block_size;

	for (int iter = 0; iter < iter_max; iter += iter_step) {
		Type* m_arr_p = matrix_array_p + iter;

		bool flag = curr_size <= block_size;
		int lim = (flag) ? curr_size : block_size;

		// L11 & U11
		for (size_t k = 0; k < lim - 1; k++) {
			Type* A_xk = m_arr_p + k;
			Type* U_kx = m_arr_p + k * start_size;
			Type A_kk = m_arr_p[k * start_size + k];
		#pragma omp parallel for
			for (int i = k + 1; i < lim; i++) {
				Type* A_ik_p = A_xk + i * start_size;
				Type* A_ix = m_arr_p + i * start_size;
				(*A_ik_p) /= A_kk; 
			#pragma omp simd
				for (int j = k + 1; j < lim; j++) {
					A_ix[j] -= (*A_ik_p) * U_kx[j]; 
				}
			}
		}
		if (flag) return;
#pragma omp parallel
		{ // комментарии сюда 
#pragma omp for // L21
			for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int i1 = std::min(i0 + block_size, curr_size);
					int k1 = std::min(k0 + kbs, block_size);
					for (int i = i0; i < i1; ++i) {
						Type* L_ix = m_arr_p + i * start_size;
						for (int k = k0; k < k1; ++k) {
							Type* U_xk = m_arr_p + k;
							Type* L_ik_p = L_ix + k;
							Type sum = 0.0;
#pragma omp simd reduction(+:sum)
							for (int j = 0; j < k; ++j) {
								sum += L_ix[j] * (*(U_xk + j * start_size));
							}
							*L_ik_p = (*L_ik_p - sum) / (*(U_xk + k * start_size));
						}
					}
				}

			}
		#pragma omp for // U12
			for (int j0 = block_size; j0 < curr_size; j0 += block_size) {
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int j1 = std::min(j0 + block_size, curr_size);
					int k1 = std::min(k0 + kbs, block_size);
					for (int k = k0; k < k1; ++k) {
						Type* m_kx = m_arr_p + k * start_size;
						for (int i = 0; i < k; ++i) {
							Type* m_ix = m_arr_p + i * start_size;
							Type m_ki = m_kx[i];
						#pragma omp simd //reduction(+:sum)
							for (int j = j0; j < j1; ++j) {
								m_kx[j] -= m_ki * m_ix[j];
							}
						}
					}
				}
			}
		}
		// L22 * U22 = A22 - L21 * U12
#pragma omp parallel for // collapse(2) конкретно этот блок замерить, че сколько занимает
		for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
			for (int j0 = block_size; j0 < curr_size; j0 += block_size) {
				int i1 = std::min(i0 + block_size, curr_size);
				int j1 = std::min(j0 + block_size, curr_size);
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int k1 = std::min(k0 + kbs, block_size);
					for (int i = i0; i < i1; ++i) {
						Type* A22_irow = m_arr_p + i * start_size;
						Type* L_ix = m_arr_p + i * start_size;
						for (int k = k0; k < k1; ++k) {
							Type L_ik = *(L_ix + k);
							Type* U_kx = m_arr_p + k * start_size;
#pragma omp simd
							for (int j = j0; j < j1; ++j) {
								A22_irow[j] -= L_ik * U_kx[j];
							}
						}
					}
				}
			}
		}
#pragma omp single 
		curr_size -= block_size;
	}
}

// коллапс, когда верхний цикл маленький и хочется его распараллелить, объединив со вторым

void DecomposerLU::get_LU(SquareMatrix& matrix_pointer) {
	Type*& m = matrix_pointer.get_array();
	const size_t size = matrix_pointer.get_size();
	size_t k_iter_max = size - 1;
	for (size_t k = 0; k < k_iter_max; k++) {
		Type* A_xk = m + k;
		Type* U_kx = m + k * size;
		Type A_kk = m[k * size + k];
#pragma omp parallel for
		for (int i = k + 1; i < size; i++) {
			Type* A_ik_p = A_xk + i * size;
			Type* A_ix = m + i * size;
			(*A_ik_p) /= A_kk;
#pragma omp simd 
			for (int j = k + 1; j < size; j++) {
				A_ix[j] -= (*A_ik_p) * U_kx[j];
			}
		}
	}
}

void DecomposerLU::decompose_LU(const SquareMatrix& A, SquareMatrix& L, SquareMatrix& U) {
	const size_t n = A.get_size();
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++) {
			if (j < i) L(i, j) = A(i, j);
			if (j == i) L(i, j) = 1;
		}
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			if (j >= i) U(i, j) = A(i, j);
}

void DecomposerLU::print_LU(const SquareMatrix& m, ostream& out) {
	const size_t n = m.get_size();
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
