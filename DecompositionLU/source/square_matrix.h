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

	// конструктор копирования
	SquareMatrix(const SquareMatrix& m);
	// копирующий оператор присваивания
	SquareMatrix& operator=(const SquareMatrix& m);

	// конструктор перемещения
	SquareMatrix(SquareMatrix&& m) noexcept;
	// перемещающий оператор присваивания
	SquareMatrix& operator=(SquareMatrix&& m) noexcept;

	// индексация с контролем

	Type& at(size_t i, size_t j);
	const Type& at(size_t i, size_t j) const;

	// индексация

	inline Type& operator()(size_t i, size_t j) { return array[i * size + j]; }
	inline const Type& operator()(size_t i, size_t j) const { return array[i * size + j]; }

	// арифметика

	SquareMatrix operator+(const SquareMatrix& m);
	bool operator==(const SquareMatrix& m);

	// блочная реализация (первая версия)
	SquareMatrix operator*(const SquareMatrix& m);
	// старая версия для замеров и сравнения
	SquareMatrix old_multi(const SquareMatrix& m);

	// пишет в res_arr копию участка массива исходной 
	// матрицы, csi и rsi - начальные индексы столбцов и рядов исходной матрицы, откуда будут браться 
	// данные, sz - размер участка
	void crop(size_t csi, size_t rsi, size_t sz, Type* res_arr) const;

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

// целевая тестируемая функция
void get_LU(SquareMatrix& matrix_pointer);
