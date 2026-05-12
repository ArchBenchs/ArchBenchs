#pragma once

#include <random>
#include <iostream>
#include <omp.h>
#include <cstring>

using namespace std;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64 // -DBLOCK_SIZE={some value}
#endif 
#define LESSER_BLOCK_SIZE BLOCK_SIZE >> 1

#ifdef working_type
typedef working_type Type;
#else 
typedef double Type;
#endif 

class SquareMatrix {
private:
	size_t size; // size of matrix line
	Type* array; // array of matrix elements, storaged by the line
	static constexpr size_t TypeSize = sizeof(Type);
public:
	// absolute error of calculations
	static constexpr Type mashine_eps = numeric_limits<Type>::epsilon();

	//  constructors & destructor 
	
	// if idd == true, inits diagonally dominant matrix, else - fills with 0
	SquareMatrix(size_t s, bool init_diag_dominant = 0);
	// fills matrix with in_arr values
	SquareMatrix(size_t s, Type* in_arr);
	// fills created matrix with randomly generated values in range [min, max] 
	SquareMatrix(size_t s, Type min, Type max);
	~SquareMatrix();

	//  copy semantics 

	SquareMatrix(const SquareMatrix& m);
	SquareMatrix& operator=(const SquareMatrix& m);

	//  move semantics 

	SquareMatrix(SquareMatrix&& m) noexcept;
	SquareMatrix& operator=(SquareMatrix&& m) noexcept;

	//  indexation 

	inline Type& operator()(size_t i, size_t j) { return array[i * size + j]; }
	inline const Type& operator()(size_t i, size_t j) const { return array[i * size + j]; }

	//  arithmetic

	SquareMatrix operator+(const SquareMatrix& m);
	SquareMatrix operator-(const SquareMatrix& m);
	SquareMatrix operator*(const SquareMatrix& m);

	//  comparison 

	bool operator==(const SquareMatrix& m);
	bool operator!=(const SquareMatrix& m);

	//  norm calculation 

	double get_infinite_norm() const;
	double get_frobenius_norm() const;
	double get_one_norm() const;

	//  getters

	inline Type*& get_array() { return array; }
	inline const size_t get_size() const { return size; }

	// I/O Stream 

	friend istream& operator>>(istream& istr, SquareMatrix& m);
	friend ostream& operator<<(ostream& ostr, const SquareMatrix& m) noexcept;
};