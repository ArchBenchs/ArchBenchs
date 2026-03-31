#pragma once
#include "square_matrix.h"

class DecomposerLU {
public:
	// matrix that simultaneously stores L and U is called "united LU matrix"

	// extracts L and U matrices from the united LU matrix
	static void decompose_LU(const SquareMatrix& A, SquareMatrix& L, SquareMatrix& U);
	// takes a united LU matrix as input and prints matrices L and U
	static void print_LU(const SquareMatrix& m, ostream& out);
	// target tested function, first version (non-block, non-recursive)
	static void get_LU(SquareMatrix& matrix_pointer);
	// target tested function, second version (block, non-recursive), input is not a matrix but its array
	static void block_get_LU(Type* m_arr_p, size_t curr_sz, size_t start_sz);
};