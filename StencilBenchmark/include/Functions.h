#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <Class_Kernel.h>

cv::Mat convolutionApply(const Kernel& kernel, cv::Mat& image);

cv::Mat Sobel_operator(cv::Mat& image);

#endif