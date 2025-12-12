#include <Functions.h>
#include <Class_Kernel.h>
#include <cmath>

cv::Mat convolutionApply(const Kernel& kernel, cv::Mat& image)
{
    if (kernel.getSize() > image.rows || kernel.getSize() > image.cols) {
        throw std::invalid_argument("Kernel is larger than image");
    }

    int result_rows = image.rows - kernel.getSize() + 1;
    int result_cols = image.cols - kernel.getSize() + 1;
    cv::Mat result(result_rows, result_cols, CV_32FC3, cv::Scalar(0, 0, 0));

    // Проход по каналам
    for (int channel = 0; channel < 3; ++channel)
    {
        // Проход по пикселям изображения (с учетом границ размера ядра)
        for (int result_y = 0; result_y < result_rows; ++result_y)
        {
            for (int result_x = 0; result_x < result_cols; ++result_x)
            {
                // Операция свертки
                for (int kernel_y = 0; kernel_y < kernel.getSize(); ++kernel_y)
                {
                    for (int kernel_x = 0; kernel_x < kernel.getSize(); ++kernel_x)
                    {
                        result.ptr<cv::Vec3f>(result_y)[result_x][channel] += 
                        image.ptr<cv::Vec3f>(result_y + kernel_y)[result_x + kernel_x][channel] 
                        * kernel.getElement(kernel_x, kernel_y);

                    }
                }
            }
        }
    }

    return result;
}

cv::Mat Sobel_operator(cv::Mat& image)
{
    Kernel horizontal (3);
    Kernel vertical (3);

    float h[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float v[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    horizontal.fillKernel(3, h);
    vertical.fillKernel(3, v);

    cv::Mat result_h = convolutionApply(horizontal, image);
    cv::Mat result_v = convolutionApply(vertical, image);

    cv::Mat result(result_h.rows, result_h.cols, CV_32FC3, cv::Scalar(0, 0, 0));

    for (int channel = 0; channel < 3; ++channel)
    {
        for (int y = 0; y < result.rows; ++y)
        {
            for (int x = 0; x < result.cols; ++x)
            {
                result.ptr<cv::Vec3f>(y)[x][channel] = 
                sqrt(pow(result_h.ptr<cv::Vec3f>(y)[x][channel], 2) + 
                pow(result_v.ptr<cv::Vec3f>(y)[x][channel], 2));
            }
        }
    }

    return result;
}