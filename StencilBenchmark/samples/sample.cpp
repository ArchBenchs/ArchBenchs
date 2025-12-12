#include <opencv2/opencv.hpp>
#include <Functions.h>
#include <chrono>
#include <string>
#include <iostream>
#include <filesystem>

using namespace std;

int main()
{   
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    int count_of_pixels_k = 16800;
    string load_from_filename = "../../../../resources/input/" + to_string(count_of_pixels_k) +"k.jpg";
    string load_to_filename = "../../../../resources/output/" + to_string(count_of_pixels_k) + "k.jpg";

    cv::Mat image = cv::imread(load_from_filename, cv::IMREAD_COLOR);

    if (image.empty())
    {
        cout << "Cannot open the image";
        return -1;
    }

    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC3);

    cv::Mat result = Sobel_operator(float_image);

    result.convertTo(image, CV_8UC3);

    bool check = imwrite(load_to_filename, image);

    if (check)
    {
        cout << "Succesfully upload the image";
    }
    else
    {
        cout << "Cannot upload the image";
    }
}