#pragma once

#include <random>
#include <iostream>
#include <omp.h>
using namespace std;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif 
#define LESSER_BLOCK_SIZE BLOCK_SIZE >> 1

#ifdef working_type
typedef working_type Type;
#else 
typedef double Type;
#endif 

class SquareMatrix {
private:
	size_t size; // размер строки матрицы
	Type* array;
	static constexpr size_t TypeSize = sizeof(Type);
public:
	static constexpr Type mashine_eps = numeric_limits<Type>::epsilon();

	SquareMatrix(size_t s, Type* in_arr = nullptr);
	// заполняет матрицу случайными значениями в заданном диапазоне 
	SquareMatrix(size_t s, Type min, Type max);
	~SquareMatrix();

	// конструктор копирования
	SquareMatrix(const SquareMatrix& m);
	// копирующий оператор присваивания
	SquareMatrix& operator=(const SquareMatrix& m);

	// конструктор перемещения
	SquareMatrix(SquareMatrix&& m) noexcept;
	// перемещающий оператор присваивания
	SquareMatrix& operator=(SquareMatrix&& m) noexcept;

	// индексация

	inline Type& operator()(size_t i, size_t j) { return array[i * size + j]; }
	inline const Type& operator()(size_t i, size_t j) const { return array[i * size + j]; }

	// арифметика

	SquareMatrix operator+(const SquareMatrix& m);
	SquareMatrix operator-(const SquareMatrix& m);
	SquareMatrix operator*(const SquareMatrix& m);

	// сравнение 

	bool operator==(const SquareMatrix& m);
	bool operator!=(const SquareMatrix& m);

	// получение нормы матрицы

	Type get_infinite_norm() const;
	Type get_frobenius_norm() const;
	Type get_one_norm() const;

	inline Type*& get_array() { return array; }
	inline void set_array(Type*&& arr) { array = arr; }

	inline const size_t get_size() const { return size; }

	friend istream& operator>>(istream& istr, SquareMatrix& m);
	friend ostream& operator<<(ostream& ostr, const SquareMatrix& m) noexcept;
};