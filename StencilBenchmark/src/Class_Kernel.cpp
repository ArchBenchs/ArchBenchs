#include <Class_Kernel.h>
#include <stdexcept>

Kernel::Kernel(size_t entered_size)
{
    size = entered_size;
    data = std::vector<float>(entered_size*entered_size, 0.0f);
}


void Kernel::fillKernel(size_t new_size, const std::vector<float>& new_data)
{   
    size_t expected = new_size * new_size;
    if (expected != new_data.size()) {
        throw std::invalid_argument("Kernel size don't match with data size");
    }

    data.resize(expected);
    size = new_size;
    data = new_data;
}

float Kernel::kerGet(int x, int y) const
{
    return data[size * y + x];
}

void Kernel::kerSet(int x, int y, float value)
{
    data[size * y + x] = value;
}

int Kernel::getSize() const
{
    return size;
}