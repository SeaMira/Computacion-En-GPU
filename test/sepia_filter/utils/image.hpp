#ifndef __IMAGE_HPP__
#define __IMAGE_HPP__

class Image
{
public:
	Image() = delete;											    // undefined
	Image(const int width, const int height, const int nbChannels); // construct an empty image
	Image(const char *const path);								    // load an image
	~Image();

	int getWidth() const { return _width; }
	int getHeight() const { return _height; }
	int getNbChannels() const { return _nbChannels; }
	int getSize() const { return _width * _height * _nbChannels; }
	unsigned char *getPtr() const { return _pixels; }

	void setPixel(const int x, const int y, const unsigned char r, const unsigned char g, const unsigned char b = 0)
	{
		const int i = _nbChannels * (x + y * _width);
		_pixels[i] = r;
		_pixels[i + 1] = g;
		_pixels[i + 2] = b;
		if (_nbChannels == 4)
			_pixels[i + 3] = r;
	}

	void saveToPng(const char *path) const;

private:
	unsigned char *_pixels; // one pixel is 3 or 4 (_nbChannels) unsigned char: R G B (A)
	int _width;
	int _height;
	int _nbChannels;
};

#endif // __IMAGE_HPP__
