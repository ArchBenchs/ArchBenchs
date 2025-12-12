#include <Class_Kernel.h>
#include <stdexcept>

Kernel::Kernel(int entered_size)
{
    if (entered_size <= 0 || entered_size > 46340)
    {
        throw std::invalid_argument("Size must be positive and not more than 46340");
    }
    if (entered_size % 2 == 0)
    {
        throw std::invalid_argument("Kernel size must me odd");
    }

    // Создаем массив
    size = entered_size;
    data = new float [size * size];

    // Инициализируем его нулевыми значениями
    for (int i = 0; i < size * size; ++i)
    {
        data[i] = 0.0f;
    }
}

Kernel::~Kernel()
{
    delete[] data; // Обнуляем массив
    data = nullptr; // Избавляемся от висячего указателя
}

void Kernel::fillKernel(int new_data_size, const float* new_data)
{
    // Проверяем правильность введенного размера
    if (size != new_data_size)
    {
        throw std::invalid_argument("Kernel size and Data size is not equal");
    }
    if (new_data == nullptr)
    {
        throw std::invalid_argument("Entered pointer is null");
    }

    for (int i = 0; i < size * size; ++i)
    {
        data[i] = new_data[i];
    }
}

float Kernel::getElement(int x, int y) const
{
    return data[size * y + x];
}

void Kernel::setElement(int x, int y, float value)
{
    data[size * y + x] = value;
}

int Kernel::getSize() const
{
    return size;
}