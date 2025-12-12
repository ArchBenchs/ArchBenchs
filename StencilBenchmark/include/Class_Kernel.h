#ifndef KERNEL_H
#define KERNEL_H

class Kernel {
    private:
        int size;
        float* data;
    public:
        Kernel(int entered_size); // Создание пустого ядра (инициализирован нулями)
        ~Kernel();
        void fillKernel(int new_data_size, const float* new_data); // Заполнение ядра из массива
        float getElement(int x, int y) const; // Получение элемента ядра
        void setElement(int x, int y, float value); // Установка элемента ядра
        int getSize() const;
};

#endif