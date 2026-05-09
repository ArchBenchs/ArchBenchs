#ifndef KERNEL_H
#define KERNEL_H

#include <vector>

class Kernel {
    private:
        size_t size;
        std::vector<float> data;
    public:
        Kernel(size_t entered_size); // Создание пустого ядра (инициализирован нулями)
        ~Kernel() = default;

        void fillKernel(size_t new_size, const std::vector<float>& new_data); // Заполнение ядра из массива
        float kerGet(int x, int y) const; // Получение элемента ядра
        void kerSet(int x, int y, float value); // Установка элемента ядра
        int getSize() const;
};

#endif