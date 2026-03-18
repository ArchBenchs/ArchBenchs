#include "decomposer_lu.h"

void DecomposerLU::block_get_LU(Type* matrix_array_p, size_t curr_sz, size_t start_sz) {
	const int block_size = BLOCK_SIZE;
	const int kbs = LESSER_BLOCK_SIZE;

	int curr_size = (int)curr_sz;
	int start_size = (int)start_sz;
	int iter_max = start_size * start_size;
	int iter_step = block_size * start_size + block_size;

/// Для дальнейших объяснений представим поступившую  на вход матрицу А в следующем виде 
/// (пусть размер матрицы = n, размер большого блока = b, размер малого = m):
/// 
///		/	  |             \            /  A(0, 0)  ...   A(0, b) \		 / A(0, b+1) ... A(0, n) \.			 
///		| A11 |     A12     |, где A11 = |	  ...	 ...	...	   |   A12 = |	  ...	 ...   ...	 |
/// A = |-----+-------------|			 \  A(b, 0)  ...  A(b, b)  /		 \ A(b, b+1) ... A(b, n) /
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
			// выбор очередного элемента с главной диагонали (далее - опорный)
			Type* A_xk = m_arr_p + k;
			Type* U_kx = m_arr_p + k * start_size;
			Type A_kk = U_kx[k];
		#pragma omp parallel for
			for (int i = k + 1; i < lim; ++i) {
				Type* A_ik_p = A_xk + i * start_size;
				Type* A_ix = m_arr_p + i * start_size;
				(*A_ik_p) /= A_kk; 
				// A(i, k) /= A(k, k) - обработка столбца под опорным элементом (1st step),
				// после этого A(i, k) == L(i, k)
			#pragma omp simd
				for (int j = k + 1; j < lim; ++j) {
					A_ix[j] -= (*A_ik_p) * U_kx[j]; 
					// A(i, j) -= L(i, k) * U(k, j); 
					// (Для k-той строки на данный момент: A(k, j) == U(k, j) )
					// обработка подматрицы от k+1-ой строки и k+1 столбца (2nd step)
				}
			}
		}
/// Визуализация (измененные за каждый шаг элементы отмечены значком ^ ; 
/// элементы, которые далее будут изменяться, заключены в скобки):
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
/// ===============> | ...  ...           ...  ...                      | ===> k++, повторить до lim-1
///					 \ lb0  (lb1 * u11)^  ...  (lb1 * u1b + ... + ubb)^ /
/// 
/// После этого этапа:
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
							// выбор строки матрицы А21
							Type* L_ix = m_arr_p + i * start_size;
							for (int k = k0; k < k1; ++k) {
								// выбор столбца на этой строке
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
/// Циклы по i и k поделены на блоки, следовательно, один блок содержит 
/// в себе подматрицу из b строк и m столбцов (обозначения выше).
/// Визуализация (обозначения аналогичны, для простоты рассмотрена не блочная обработка):
/// 
///					  / l(b+1, 0) ... l(b+1, b) \   / u00 u01 ... u0b \.   
/// А21 = L21 * U11 = |	  ...	  ...	...	    | X |  0  u11 ... u1b | = 
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
/// ===> i++, повторять, пока i < n.
/// 
/// После этого этапа A21 = L21.
			}
#pragma omp for // U12
			for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int i1 = std::min(i0 + block_size, curr_size);
					int k1 = std::min(k0 + kbs, block_size);
					for (int k = k0; k < k1; ++k) {
						// Выбор строки матрицы A, для дальнейшего выбора элемента 
						// матрицы L11 и обрабатываемого элемента матрицы А12
						Type* U_kx = m_arr_p + k * start_size;
						for (int j = 0; j < k; ++j) {
							// Выбор столбца, по которому берется элемент матрицы L11, и строки,
							// по которой берется его множитель из U12 (обработанной части А12)
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
/// Циклы по i и k поделены на блоки, следовательно 1 блок содержит 
/// подматрицу из m строк и b столбцов (считал по элементу A(k, i) ).
/// Визуализация (аналогично; нач. условие: k = 1 (при k = 0 ничего не происходит) ):
/// 
///					  /  1   0  ...  0  \   / u(0, b+1) ... u(0, n) \.      
/// А12 = L11 * U12 = | l10  1  ...  0  | X |	 ...	...	  ...   | = 
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
/// ===> k++ и так далее.
/// После этого этапа A12 = U12.
			}
		// L22 * U22 = A22 - L21 * U12
#pragma omp parallel for // конкретно этот блок замерить, че сколько занимает
		for (int i0 = block_size; i0 < curr_size; i0 += block_size) {
			for (int j0 = block_size; j0 < curr_size; j0 += block_size) {
				int i1 = std::min(i0 + block_size, curr_size);
				int j1 = std::min(j0 + block_size, curr_size);
				for (int k0 = 0; k0 < block_size; k0 += kbs) {
					int k1 = std::min(k0 + kbs, block_size);
					for (int i = i0; i < i1; ++i) {
						// Выбор строки матрицы A, для дальнейшего выбора элемента 
						// матрицы L21 и обрабатываемого элемента матрицы А22
						Type* A22_ix = m_arr_p + i * start_size;
						for (int k = k0; k < k1; ++k) {
							// Выбор столбца, по которому берется элемент матрицы L21, и строки,
							// по которой берется его множитель из U12
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
/// Визуализация не требуется по 2-м причинам:
///		- Она практически идентична визуализации U12, за исключением того, 
///		  что здесь цикл по j блочный, а не до k
///		- Тривиальная реализация blast (вроде бы)
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
