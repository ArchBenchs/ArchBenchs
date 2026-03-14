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

	SquareMatrix(size_t s, Type* in_arr = nullptr);
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

	Type get_infinite_norm() const;
	Type get_frobenius_norm() const;
	Type get_one_norm() const;

	//  getters & setters 

	inline Type*& get_array() { return array; }
	inline void set_array(Type*&& arr) { array = arr; }

	inline const size_t get_size() const { return size; }

	// I/O Stream 

	friend istream& operator>>(istream& istr, SquareMatrix& m);
	friend ostream& operator<<(ostream& ostr, const SquareMatrix& m) noexcept;

};