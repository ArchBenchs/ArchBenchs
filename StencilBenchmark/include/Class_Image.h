#ifndef CLASS_IMAGE_H
#define CLASS_IMAGE_H

#include <vector>

class Image {
	private:
		int width;
		int height;
		std::vector<float> data;
	public:
		Image(int width, int height, int channels); // Пустое изображение с нулевыми пикселями
		Image(int width, int height, std::vector<float>&& inner_data); // Заполнение из существующего вектора
		~Image() = default;

		int imWidth() const;
		int imHeight() const;

		float imGet(int x, int y, int channel) const;
		void imSet(int x, int y, int channel, float val);
		const std::vector<float>& imData() const;
		std::vector<float>& imData();

};

#endif