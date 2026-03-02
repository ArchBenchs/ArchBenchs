#pragma once

#include <random>
#include <iostream>
#include <omp.h>
using namespace std;

typedef double Type;

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
	bool operator==(const SquareMatrix& m);

	// блочная реализация (первая версия)
	SquareMatrix operator*(const SquareMatrix& m);
	// старая версия для замеров и сравнения
	SquareMatrix old_multi(const SquareMatrix& m);

	// получение нормы матрицы

	Type get_infinite_norm() const;
	double get_frobenius_norm() const;
	Type get_one_norm() const;

	inline Type*& get_array() { return array; }
	inline void set_array(Type*&& arr) { array = arr; }

	inline const size_t get_size() const { return size; }

	// далее матрица, одновременно хранящая L и U, называется "общая LU матрица"
	
	// из общей LU матрицы получает, соответственно, L и U
	void decompose_LU(SquareMatrix& L, SquareMatrix& U);
	// вывод L и U из общей LU матрицы
	friend void print_LU(const SquareMatrix& m, ostream& out);

	friend istream& operator>>(istream& istr, SquareMatrix& m);
	friend ostream& operator<<(ostream& ostr, const SquareMatrix& m) noexcept;
};

// целевая тестируемая функция, первая версия (не блочная, не рекурсивная)
void get_LU(SquareMatrix& matrix_pointer);
// целевая тестируемая функция, вторая версия (блочная, не рекурсивная), на вход поступает не матрица, а ее массив
void block_get_LU(Type* m_arr_p, size_t curr_sz, size_t start_sz);

// сравнивает значения, используя абсолютную погрешность
inline bool compare_eps(const Type& arg1, const Type& arg2, const Type& eps) {
	return fabs(arg1 - arg2) <= eps;
}
// сравнивает значения, используя относительную погрешность
inline bool compare_rel(const Type& arg1, const Type& arg2, const Type& eps) {
	Type abs1 = fabs(arg1), abs2 = fabs(arg2);
	Type max = (abs1 > abs2) ? abs1 : abs2;
	return fabs(arg1 - arg2) <= max * eps;
}