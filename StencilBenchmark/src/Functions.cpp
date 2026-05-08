#include <Functions.h>
#include <Class_Kernel.h>
#include <Class_Image.h>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

Image upScale(const Image& input, int width_scale, int height_scale) {
    std::cout << "Starting scalying image..." << std::endl;
    int width = input.imWidth();
    int height = input.imHeight();

    Image result(width * width_scale, height * height_scale, 3);

    for (int channel = 0; channel < 3; ++channel) {
        for (int w = 0; w < width; ++w) {
            for (int h = 0; h < height; ++h) {
                for (int ws = 0; ws < width_scale; ++ws) {
                    for (int hs = 0; hs < height_scale; ++hs) {
                        result.imSet(w + width * ws, h + height * hs, channel, input.imGet(w, h, channel));
                    }
                }
            }
        }
    }

    std::cout << "The image is scaled with: Wx" << width_scale << ", Hx" << height_scale << "!" << std::endl;
    return result;
}


Image convolutionApply(const Kernel& kernel, Image& image, std::chrono::duration<double>& timer)
{
    if (kernel.getSize() > image.imHeight() || kernel.getSize() > image.imWidth()) {
        throw std::invalid_argument("Kernel is larger than image");
    }

    int result_height = image.imHeight() - kernel.getSize() + 1;
    int result_width = image.imWidth() - kernel.getSize() + 1;
    Image result(result_width, result_height, 3);

    int kernel_size = kernel.getSize();

    auto start = std::chrono::steady_clock::now();

    // Проход по пикселям изображения (с учетом границ размера ядра)
    for (int result_ycord = 0; result_ycord < result_height; ++result_ycord)
    {
        for (int result_xcord = 0; result_xcord < result_width; ++result_xcord)
        {
            // Проход по каналам
            for (int channel = 0; channel < 3; ++channel)
            {
                float convolution_result = 0.0f;
                // Операция свертки
                for (int kernel_ycord = 0; kernel_ycord < kernel_size; ++kernel_ycord)
                {
                    for (int kernel_xcord = 0; kernel_xcord < kernel_size; ++kernel_xcord)
                    {
                        float temp_result = image.imGet(result_xcord + kernel_xcord, result_ycord + kernel_ycord, channel) * kernel.kerGet(kernel_xcord, kernel_ycord);
                        convolution_result += temp_result;
                    }
                }
                
            result.imSet(result_xcord, result_ycord, channel, convolution_result);
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    timer = end - start;

    std::cout << "Convolution was successfully done!" << std::endl;
    return result;
}


std::vector<float> convertToFloat(const std::vector<uint8_t>& input) {
    size_t length = input.size();
    std::vector<float> floatData(length);

    for (size_t i = 0; i < length; i++) {
        floatData[i] = static_cast<float>(input[i]);
    }

    std::cout << "CHAR Vector was converted to FLOAT!" << std::endl;
    return floatData;
}


std::vector<uint8_t> convertToChar(const std::vector<float>& input) {
    size_t length = input.size();
    std::vector<uint8_t> charData(length);

    for (size_t i = 0; i < length; i++) {
        charData[i] = 
        static_cast<uint8_t>(std::clamp(std::round(input[i]), 0.0f, 255.0f));
    }

    std::cout << "FLOAT Vector was converted to CHAR!" << std::endl;
    return charData;
}


void normalize(std::vector<float>& input) {
    if (input.empty()) return;

    size_t length = input.size();

    auto [minIt, maxIt] = std::minmax_element(input.begin(), input.end());

    float maxVal = *maxIt;
    float minVal = *minIt;

    float range = maxVal - minVal;

    if (range == 0.0f){
        std::fill(input.begin(), input.end(), std::clamp(minVal, 0.0f, 255.0f));
        return;
    }

    for (size_t i = 0; i < length; i++) {
        input[i] = std::clamp((input[i] - minVal) * 255.0f / range, 0.0f, 255.0f);
    }

    std::cout << "Image was normalized!" << std::endl;
    return;
}


Image Sobel_operator(Image& image)
{
    Kernel horizontal (3);
    Kernel vertical (3);
    
    std::vector<float> h = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    std::vector<float> v = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    
    horizontal.fillKernel(3, h);
    vertical.fillKernel(3, v);

    std::cout << "Successfully filled kernels for SOBEL!" << std::endl;
    std::cout << "Starting convolution..." << std::endl;

    std::chrono::duration<double> timer_1;
    std::chrono::duration<double> timer_2;

    Image result_h = convolutionApply(horizontal, image, timer_1);
    std::cout << "Horizontal kernel is successfully completed!" << std::endl;

    Image result_v = convolutionApply(vertical, image, timer_2);
    std::cout << "Vertical kernel is successfully completed!" << std::endl;

    // Запись в файл логов
    std::ofstream log("./resources/output/timelog.txt", std::ios::app);
    if (log.is_open()) {
        log << std::to_string(image.imWidth() * image.imHeight() * 3) << ", " << std::to_string((timer_1 + timer_2).count() / 2.0) << std::endl;
        std::cout << "Writing to log file.." << std::endl;
    }
    else {
        std::cerr << "Error: failed to open log file" << std::endl;
    }
    log.close();

    int width = result_h.imWidth();
    int height = result_h.imHeight();

    Image result(width, height, 3);

    std::cout << "Starting to combine the result.." << std::endl;
    for (int channel = 0; channel < 3; ++channel)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                result.imSet(x, y, channel, sqrt(pow(result_h.imGet(x, y, channel), 2) + pow(result_v.imGet(x, y, channel), 2)));
            }
        }
    }

    std::cout << "Successfully combine the result!" << std::endl;

    return result;
}


bool loadBMP(const std::string& filename, std::vector<uint8_t>& bgrData, 
    int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Failed to open the file " << filename << std::endl;
        return false;
    }

    BMPFileHeader fileHeader;

    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(BMPFileHeader));

    if (fileHeader.bfType != 0x4D42) {
        std::cerr << "Error: file format is not a BMP" << std::endl;
        return false;
    }

    BMPInfoHeader infoHeader;

    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(BMPInfoHeader));
    
    if (infoHeader.biSize != 40) {
        std::cerr << "Error: reader supports only file headers of 40 bit in size." << " Here is " << infoHeader.biSize << std::endl;
        return false;
    }

    if (infoHeader.biBitCount != 24) {
        std::cerr << "Error: reader supports only 24-bit pixel files" << std::endl;
        return false;
    }

    if (infoHeader.biCompression != 0) {
        std::cerr << "Error: reader doesn't support compressed files" << std::endl;
        return false;
    }

    width = infoHeader.biWidth;
    height = (infoHeader.biHeight > 0) ? infoHeader.biHeight : -infoHeader.biHeight;

    bool bottomUp = (infoHeader.biHeight > 0);
    int bytesPerRow = (width * 3 + 3) & ~3;

    file.seekg(fileHeader.bfOffBits, std::ios::beg);

    bgrData.resize(width * height * 3);
    std::vector<uint8_t> rowBuffer(bytesPerRow);

    for (int y = 0; y < height; y++) {
        file.read(reinterpret_cast<char*>(rowBuffer.data()), bytesPerRow);

        if (!file) {
            std::cerr << "Error: failed to read data" << std::endl;
            return false;
        }

        int destRow;
        if (bottomUp) {
            destRow = height - 1 - y;
        }
        else {
            destRow = y;
        }

        for (int x = 0; x < width; x++) {
            int destIndex = (destRow * width + x) * 3;

            bgrData[destIndex] = rowBuffer[x * 3];
            bgrData[destIndex + 1] = rowBuffer[x * 3 + 1];
            bgrData[destIndex + 2] = rowBuffer[x * 3 + 2];
        }
    }

    file.close();
    std::cout << "Successfully opened the image!" << std::endl;
    return true;
}


bool saveBMP(const std::string& filename, const std::vector<uint8_t>& bgrData, int width, int height) {
    int bytesPerRow = (width * 3 + 3) & ~3;
    int imageSize = bytesPerRow * height;

    BMPFileHeader fileHeader{};
    fileHeader.bfType = 0x4D42;
    fileHeader.bfSize = 54 + imageSize;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = 54;

    BMPInfoHeader infoHeader{};
    infoHeader.biSize = 40;
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = imageSize;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: failed to create or open for saving file " << filename << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    if (!file.good()) {
        std::cerr << "Error: failed to write headers" << std::endl;
        return false;
    }

    std::vector<uint8_t> rowBuffer(bytesPerRow, 0);
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int srcIndex = (y * width + x) * 3;
            rowBuffer[x * 3] = bgrData[srcIndex];
            rowBuffer[x * 3 + 1] = bgrData[srcIndex + 1];
            rowBuffer[x * 3 + 2] = bgrData[srcIndex + 2];
        }
        file.write(reinterpret_cast<const char*>(rowBuffer.data()), bytesPerRow);
        if (!file.good()) {
            std::cerr << "Error: failed to write " << y << " string" << std::endl;
            return false;
        }
    }

    std::cout << "Successfully save the image!" << std::endl;
    return true;
}