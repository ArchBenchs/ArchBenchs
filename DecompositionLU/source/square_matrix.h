#pragma once

#include <iostream>
using namespace std;

typedef double Type;

class SquareMatrix {
private:
	size_t size;
	Type* array;
	static constexpr size_t TypeSize = sizeof(Type);
public:
	SquareMatrix(size_t s, Type* in_arr = nullptr);
	~SquareMatrix();

	SquareMatrix(const SquareMatrix& m);
	SquareMatrix& operator=(const SquareMatrix& m);

	SquareMatrix(SquareMatrix&& m) noexcept;
	SquareMatrix& operator=(SquareMatrix&& m) noexcept;

	Type& at(size_t i, size_t j);
	const Type& at(size_t i, size_t j) const;

	// пока самая простая реализация для тестов
	SquareMatrix operator*(const SquareMatrix& m);
	bool operator==(const SquareMatrix& m);

	inline Type& operator()(size_t i, size_t j) { return array[i * size + j]; }
	inline const Type& operator()(size_t i, size_t j) const { return array[i * size + j]; }

	inline Type*& get_array() { return array; }
	inline const size_t get_size() const { return size; }
	inline void set_array(Type*&& arr) { array = arr; }

	// из матрицы, в которой хранятся L и U получает, соответственно, L и U
	void decompose_LU(SquareMatrix& L, SquareMatrix& U);
	friend void print_LU(const SquareMatrix& m);

	friend istream& operator>>(istream& istr, SquareMatrix& m);
	friend ostream& operator<<(ostream& ostr, SquareMatrix& m) noexcept;
};

void get_LU(SquareMatrix& matrix_pointer);