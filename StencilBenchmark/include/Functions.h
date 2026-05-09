#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Class_Image.h>
#include <Class_Kernel.h>

#include <chrono>
#include <vector>
#include <cstdint>
#include <string>

Image upScale(const Image& image, int witdh_scale, int height_scale);

Image convolutionApply(const Kernel& kernel, Image& image, std::chrono::duration<double>& timer);

Image Sobel_operator(Image& image);


#pragma pack(push, 1)

// Файловый заголовок
struct BMPFileHeader {
	uint16_t bfType;
	uint32_t bfSize;
	uint16_t bfReserved1;
	uint16_t bfReserved2;
	uint32_t bfOffBits;
};

// Информационный заголовок
struct BMPInfoHeader {
	uint32_t biSize;
	int32_t biWidth;
	int32_t biHeight;
	uint16_t biPlanes;
	uint16_t biBitCount;
	uint32_t biCompression;
	uint32_t biSizeImage;
	int32_t biXPelsPerMeter;
	int32_t biYPelsPerMeter;
	uint32_t biClrUsed;
	uint32_t biClrImportant;
};

#pragma pack(pop)

// Функция загрузки изображения из файла
bool loadBMP(const std::string& filename, std::vector<uint8_t>& bgrData, 
	int& width, int& height);

// Функция сохранения изображения в файл
bool saveBMP(const std::string& filename, const std::vector<uint8_t>& bgrData, 
	int width, int height);

// Конвертация во Float
std::vector<float> convertToFloat(const std::vector<uint8_t>& input);

// Конвертация в Char
std::vector<uint8_t> convertToChar(const std::vector<float>& input);

// Функция нормализации
void normalize(std::vector<float>& input);

#endif