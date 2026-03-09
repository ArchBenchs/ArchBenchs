#pragma once
#include "square_matrix.h"

class DecomposerLU {
public:
	// матрица, одновременно хран€ща€ L и U, называетс€ "обща€ LU матрица"

	// из общей LU матрицы получает матрицы L и U
	static void decompose_LU(const SquareMatrix& A, SquareMatrix& L, SquareMatrix& U);
	// принимает на вход общую LU матрицу и печатает матрицы L и U
	static void print_LU(const SquareMatrix& m, ostream& out);
	// целева€ тестируема€ функци€, перва€ верси€ (не блочна€, не рекурсивна€)
	static void get_LU(SquareMatrix& matrix_pointer);
	// целева€ тестируема€ функци€, втора€ верси€ (блочна€, не рекурсивна€), на вход поступает не матрица, а ее массив
	static void block_get_LU(Type* m_arr_p, size_t curr_sz, size_t start_sz);
};