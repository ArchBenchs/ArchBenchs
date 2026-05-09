#include <Class_Image.h>
#include <iostream>

Image::Image(int input_width, int input_height, int channels) : width(input_width), height(input_height), data(input_width * input_height * channels, 0.0f) {

}


Image::Image(int input_width, int input_height, std::vector<float>&& input_data) : width(input_width), height(input_height), data(std::move(input_data)){
	std::cout << "Successfully created the Image with WIDTH = " << width << " and HEIGHT " << height << " !" << std::endl;
}


int Image::imWidth() const {
	return width;
}


int Image::imHeight() const {
	return height;
}


float Image::imGet(int x, int y, int channel) const {
	return data[(width * y + x) * 3 + channel];
}


void Image::imSet(int x, int y, int channel, float value) {
	data[(width * y + x) * 3 + channel] = value;
}


const std::vector<float>& Image::imData() const {
	return data;
}


std::vector<float>& Image::imData() {
	return data;
}