#include "image.hpp"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

Image::Image(const int width, const int height, const int nbChannels)
	: _pixels(nullptr), _width(width), _height(height), _nbChannels(nbChannels)
{
	_pixels = new unsigned char[_width * _height * _nbChannels];
	memset(_pixels, 0, _width * _height * _nbChannels);
}

Image::Image(const char *const path)
	: _pixels(nullptr), _width(0), _height(0)
{
	_pixels = stbi_load(path, &_width, &_height, &_nbChannels, 0);
}

Image::~Image() { stbi_image_free(_pixels); }

void Image::saveToPng(const char *path) const
{
	stbi_write_png(path, _width, _height, _nbChannels, _pixels, _width * _nbChannels);
}
