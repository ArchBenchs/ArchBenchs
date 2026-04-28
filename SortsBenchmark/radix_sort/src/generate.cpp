#include <iostream>
#include <fstream>
#include <random>

int main() {
    const int64_t N = 10000000;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-16.0, 16.0);
    
    std::ofstream file("random_data.bin", std::ios::binary);
    
    for (int64_t i = 0; i < N; i++) {
        double val = dis(gen);
        file.write(reinterpret_cast<char*>(&val), sizeof(double));
    }
    
    file.close();
    std::cout << "Generated " << N << " numbers to random_data.bin" << std::endl;
    
    return 0;
}
