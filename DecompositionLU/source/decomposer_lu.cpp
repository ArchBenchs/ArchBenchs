#include "decomposer_lu.h"

void DecomposerLU::block_get_LU(Type* matrix_array_p, size_t curr_sz, size_t start_sz) {
	const int block_size = BLOCK_SIZE;
	const int kbs = LESSER_BLOCK_SIZE;

	int curr_size = (int)curr_sz;
	int start_size = (int)start_sz;
	int iter_max = start_size * start_size;
	int iter_step = block_size * start_size + block_size;

/// For further explanation, let's represent the input matrix A as follows 
/// (let matrix size = n, large block size = b, small block size = m):
/// 
///		/	  |             \              /  A(0, 0)  ...   A(0, b) \		   / A(0, b+1) ... A(0, n) \.			 
///		| A11 |     A12     |, where A11 = |	...    ...	   ...   |   A12 = |	...	   ...   ...   |
/// A = |-----+-------------|			   \  A(b, 0)  ...  A(b, b)  /		   \ A(b, b+1) ... A(b, n) /
///		|     |				|
///		|	  |				|            / A(b+1, 0) ... A(b+1, b) \		 / A(b+1, b+1) ... A(b+1, n) \.			 			 
///		| A21 |		A22		|	   A21 = |	  ...	 ...	...	   |   A22 = |	  ...	   ...   ...	 |	   
///		|     |				|			 \  A(n, 0)  ...  A(n, b)  /		 \ A(n, b+1)   ... A(n, n)   /			 
///		\	  |				/
///							

	for (int iter = 0; iter < iter_max; iter += iter_step) {
		Type* m_arr_p = matrix_array_p + iter;

		bool flag = curr_size <= block_size;
		int lim = (flag) ? curr_size : block_size;

		// L11 & U11
		int klim = lim - 1;
		for (size_t k = 0; k < klim; ++k) {
			// select the next element on the main diagonal (hereinafter - pivot)
			Type* A_xk = m_arr_p + k;
			Type* U_kx = m_arr_p + k * start_size;
			Type A_kk = U_kx[k];
		#pragma omp parallel for
			for (int i = k + 1; i < lim; ++i) {
				Type* A_ik_p = A_xk + i * start_size;
				Type* A_ix = m_arr_p + i * start_size;
				(*A_ik_p) /= A_kk;
				// A(i, k) /= A(k, k) - processing the column below the pivot element (1st step),
				// after this A(i, k) == L(i, k)
			#pragma omp simd
				for (int j = k + 1; j < lim; ++j) {
					A_ix[j] -= (*A_ik_p) * U_kx[j];
					// A(i, j) -= L(i, k) * U(k, j); 
					// (For the k-th row at this moment: A(k, j) == U(k, j))
					// processing the submatrix from row k+1 and column k+1 (2nd step)
				}
			}
		}
/// Visualization (elements changed at each step are marked with ^; 
/// elements that will be changed later are enclosed in parentheses):
/// 
///					  /  1   0  ...  0  \   / u00 u01 ... u0b \   / u00          u01                      ...  u0b                                 \.
/// A11 = L11 * U11 = | l10  1  ...  0  |   |  0  u11 ... u1b |   | (l10 * u00)  (l10 * u01 + u11)        ...  (l10 * u0b + u1b)                   |
///					  | ... ... ... ... | X | ... ... ... ... | = | ...          ...                      ...  ...                                 | ===>
///					  \ lb0 lb1 ...  1  /   \  0   0   0  ubb /   \ (lb0 * u00)  (lb0 * u01 + lb1 * u11)  ...  (lb0 * u0b + lb1 * u1b + ... + ubb) /
/// 
///					 / u00     u01                      ...  u0b                                 \.
/// k = 0, 1st step  | l10^    (l10 * u01 + u11)        ...  (l10 * u0b + u1b)                   |
/// ===============> | ...     ...                      ...  ...                                 | ===>
///					 \ lb0^    (lb0 * u01 + lb1 * u11)  ...  (lb0 * u0b + lb1 * u1b + ... + ubb) /
///					 
///					 / u00  u01           ...  u0b                      \.
/// k = 0, 2nd step  | l10  u11^          ...  u1b^                     |
/// ===============> | ...  ...           ...  ...                      | ===> k++, repeat while k < lim-1
///					 \ lb0  (lb1 * u11)^  ...  (lb1 * u1b + ... + ubb)^ /
/// 
/// After this stage:
/// 
///		  / u00 u01 u02 ... u0b \    /\		   \.
///		  |	l10 u11 u12 ... u1b |   |   \  U11  |
///	A11 = | l20 l21 u22 ... u2b | = |     \	    |
///		  | ... ... ... ... ... |   | L11   \   |
///		  \ lb0 lb1 lb2 ... ubb /    \        \/
/// 
		if (flag) return;
#pragma omp parallel
		{
		#pragma omp for // L21
			for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int i1 = std::min(i0 + block_size, curr_size);
					int k1 = std::min(k0 + kbs, block_size);
					for (int i = i0; i < i1; ++i) {
						// select row of matrix A21
						Type* L_ix = m_arr_p + i * start_size;
						for (int k = k0; k < k1; ++k) {
							// select column in this row
							Type* U_xk = m_arr_p + k;
							Type* L_ik_p = L_ix + k;
							Type sum = 0.0;
						#pragma omp simd reduction(+:sum)
							for (int j = 0; j < k; ++j) {
								sum += L_ix[j] * (*(U_xk + j * start_size));
							}
							*L_ik_p = (*L_ik_p - sum) / (*(U_xk + k * start_size));
							// A(i, k) -= sum( L(i,j) * U(j, k) | j = 0,...,k )
							// A(i, k) /= U(k, k)
						}
					}
				}
/// Loops over i and k are divided into blocks, so one block contains 
/// a submatrix of b rows and m columns (notations above).
/// Visualization (notations are similar, non-block processing is considered for simplicity):
/// 
///					  / l(b+1, 0) ... l(b+1, b) \   / u00 u01 ... u0b \.   
/// À21 = L21 * U11 = |	  ...	  ...	...	    | X |  0  u11 ... u1b | = 
///					  \  l(n, 0)  ...  l(n, b)  /   | ... ... ... ... |   
///													\  0   0  ... ubb /   
/// 
///   / (l(b+1, 0) * u00)  (l(b+1, 0) * u01 + l(b+1, 1) * u11)  ...  (l(b+1, 0) * u0b + ... + l(b+1, b) * ubb) \.
/// = | (l(b+2, 0) * u00)  (l(b+2, 0) * u01 + l(b+2, 1) * u11)  ...  (l(b+2, 0) * u0b + ... + l(b+2, b) * ubb) |
///   |		  ...						  ...					...						...					   | ===>
///   \  (l(n, 0) * u00)     (l(n, 0) * u01 + l(n, 1) * u11)    ...    (l(n, 0) * u0b + ... + l(n, b) * ubb)   /
/// 
///				    /     l(b+1, 0)^     (l(b+1, 0) * u01 + l(b+1, 1) * u11)	...  (l(b+1, 0) * u0b + ... + l(b+1, b) * ubb)	\. 
/// i = b+1, k = 0  | (l(b+2, 0) * u00)  (l(b+2, 0) * u01 + l(b+2, 1) * u11)	...  (l(b+2, 0) * u0b + ... + l(b+2, b) * ubb)	|
/// ==============> |		  ...						  ...					...						...						| ===>
///				    \  (l(n, 0) * u00)     (l(n, 0) * u01 + l(n, 1) * u11)		...    (l(n, 0) * u0b + ... + l(n, b) * ubb)	/
///     ...
///				    /     l(b+1, 0)					l(b+1, 1)					...				  l(b+1, b)^					\. 
/// i = b+1, k = b  | (l(b+2, 0) * u00)  (l(b+2, 0) * u01 + l(b+2, 1) * u11)	...  (l(b+2, 0) * u0b + ... + l(b+2, b) * ubb)	|
/// ==============> |		  ...						  ...					...						...						| ===>
///				    \  (l(n, 0) * u00)     (l(n, 0) * u01 + l(n, 1) * u11)		...    (l(n, 0) * u0b + ... + l(n, b) * ubb)	/
/// 
/// ===> i++, repeat while i < n.
/// 
/// After this stage A21 = L21.
			}
		#pragma omp for // U12
			for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int i1 = std::min(i0 + block_size, curr_size);
					int k1 = std::min(k0 + kbs, block_size);
					for (int k = k0; k < k1; ++k) {
						// Select row of matrix A, for further selection of element 
						// of matrix L11 and the processed element of matrix A12
						Type* U_kx = m_arr_p + k * start_size;
						for (int j = 0; j < k; ++j) {
							// Select the column from which the element of matrix L11 is taken, and the row,
							// from which its multiplier from U12 (processed part of A12) is taken
							Type* U_jx = m_arr_p + j * start_size;
							Type L_kj = U_kx[j];
						#pragma omp simd
							for (int i = i0; i < i1; ++i) {
								// A(k, i) -= U(j, i) * L(k, j)
								U_kx[i] -= L_kj * U_jx[i];
							}
						}
					}
				}
			}
/// Loops over i and k are divided into blocks, so 1 block contains 
/// a submatrix of m rows and b columns (counting by element A(k, i)).
/// Visualization (similar; initial condition: k = 1 (nothing happens at k = 0)):
/// 
///					  /  1   0  ...  0  \   / u(0, b+1) ... u(0, n) \.      
/// À12 = L11 * U12 = | l10  1  ...  0  | X |	 ...	...	  ...   | = 
///					  | ... ... ... ... |   \ u(b, b+1) ... u(b, n) /      
///					  \ lb0 lb1 ...  1  /   
/// 
///	  /  u(0, b+1)												u(0, b+2)											  ...   u(0, n)										   \.
///	  | (u(0, b+1) * l10 + u(1, b+1))						   (u(0, b+2) * l10 + u(1, b+2))						  ...  (u(0, n) * l10 + u(1, n))					   |
/// = |  ...													...													  ...	...											   | ===>
///	  \ (u(0, b+1) * lb0 + u(1, b+1) * lb1 + ... + u(b, b+1))  (u(0, b+2) * lb0 + u(1, b+2) * lb1 + ... + u(b, b+2))  ...  (u(0, n) * lb0 + u(1, n) * lb1 + ... + u(b, n)) /
/// 
///				    /  u(0, b+1)											  u(0, b+2)												...   u(0, n)										 \.
/// j = 0, i = b+1  |  u(1, b+1)^											 (u(0, b+2) * l10 + u(1, b+2))							...  (u(0, n) * l10 + u(1, n))						 |
/// ==============> |  ...													  ...													...	  ...											 | ===>
///				    \ (u(0, b+1) * lb0 + u(1, b+1) * lb1 + ... + u(b, b+1))  (u(0, b+2) * lb0 + u(1, b+2) * lb1 + ... + u(b, b+2))  ...  (u(0, n) * lb0 + u(1, n) * lb1 + ... + u(b, n)) /
/// ...
///				  /  u(0, b+1)												 u(0, b+2)											   ...   u(0, n)										\.
/// j = 0, i = n  |  u(1, b+1) 												 u(1, b+2)											   ...	 u(1, n)^										|
/// ============> | (u(0, b+1) * lb0 + u(1, b+1) * lb1 + u(b, b+1))			(u(0, b+2) * lb0 + u(1, b+2) * lb1 + u(b, b+2))		   ...	(u(0, n) * lb0 + u(1, n) * lb1 + u(b, n))		| ===>
///				  |  ...													 ...												   ...													| 
///				  \ (u(0, b+1) * lb0 + u(1, b+1) * lb1 + ... + u(b, b+1))	(u(0, b+2) * lb0 + u(1, b+2) * lb1 + ... + u(b, b+2))  ...  (u(0, n) * lb0 + u(1, n) * lb1 + ... + u(b, n)) /
/// 
/// ===> k++; ===>
/// 
///				    /  u(0, b+1)											 u(0, b+2)											   ...   u(0, n)										\.
/// j = 0, i = b+1  |  u(1, b+1) 											 u(1, b+2)											   ...	 u(1, n) 										|
/// ==============> | (u(1, b+1) * l21 + u(2, b+1))^						(u(0, b+2) * l20 + u(1, b+2) * l21 + u(2, b+2))		   ...	(u(0, n) * l20 + u(1, n) * l21 + u(2, n))		| ===>
///				    |  ...													 ...												   ...	 ...											| 
///				    \ (u(0, b+1) * lb0 + u(1, b+1) * lb1 + ... + u(b, b+1))	(u(0, b+2) * lb0 + u(1, b+2) * lb1 + ... + u(b, b+2))  ...  (u(0, n) * lb0 + u(1, n) * lb1 + ... + u(b, n)) /
/// ...
///				    /  u(0, b+1)											 u(0, b+2)											   ...   u(0, n)										\.
/// j = 0, i = n    |  u(1, b+1) 											 u(1, b+2)											   ...	 u(1, n) 										|
/// ==============> | (u(1, b+1) * l21 + u(2, b+1)) 						(u(1, b+2) * l21 + u(2, b+2))						   ...	(u(1, n) * l21 + u(2, n))^						| ===>
///				    |  ...													 ...												   ...	 ...											| 
///				    \ (u(0, b+1) * lb0 + u(1, b+1) * lb1 + ... + u(b, b+1))	(u(0, b+2) * lb0 + u(1, b+2) * lb1 + ... + u(b, b+2))  ...  (u(0, n) * lb0 + u(1, n) * lb1 + ... + u(b, n)) /
/// ...
///				    /  u(0, b+1)											 u(0, b+2)											   ...   u(0, n)										\.
/// j = 1, i = n    |  u(1, b+1) 											 u(1, b+2)											   ...	 u(1, n) 										|
/// ==============> |  u(2, b+1) 											 u(2, b+2)											   ...	 u(2, n)^										| ===>
///				    |  ...													 ...												   ...	 ...											| 
///				    \ (u(0, b+1) * lb0 + u(1, b+1) * lb1 + ... + u(b, b+1))	(u(0, b+2) * lb0 + u(1, b+2) * lb1 + ... + u(b, b+2))  ...  (u(0, n) * lb0 + u(1, n) * lb1 + ... + u(b, n)) /
///
/// ===> k++ and so on.
/// After this stage A12 = U12.
		}
		// L22 * U22 = A22 - L21 * U12
	#pragma omp parallel for // measure this block specifically
		for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
			for (int j0 = block_size; j0 < curr_size; j0 += block_size) {
				int i1 = std::min(i0 + block_size, curr_size);
				int j1 = std::min(j0 + block_size, curr_size);
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int k1 = std::min(k0 + kbs, block_size);
					for (int i = i0; i < i1; ++i) {
						// Select row of matrix A, for further selection of element 
						// of matrix L21 and the processed element of matrix A22
						Type* A22_ix = m_arr_p + i * start_size;
						for (int k = k0; k < k1; ++k) {
							// Select the column from which the element of matrix L21 is taken, and the row,
							// from which its multiplier from U12 is taken
							Type L_ik = *(A22_ix + k);
							Type* U_kx = m_arr_p + k * start_size;
						#pragma omp simd
							for (int j = j0; j < j1; ++j) {
								// A(i, j) -= L(i, k) * U(k, j)
								A22_ix[j] -= L_ik * U_kx[j];
							}
						}
					}
				}
			}
		}
/// Visualization is not required for 2 reasons:
///		- It is almost identical to the U12 visualization, except that 
///		  the loop over j here is blocked, not up to k
///		- Trivial blas implementation (as it seems)
	#pragma omp single 
		curr_size -= block_size;
	}
}

// collapse, when the outer loop is small and you want to parallelize it by combining with the second one

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