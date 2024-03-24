// DO NOT MODIFY THIS FILE !!!

#include <algorithm>

#include "reference.hpp"
#include "utils/chronoCPU.hpp"
#include "utils/utils.hpp"

float convolutionCPU(const Image &in, Image &out, const ConvolutionKernel &conv)
{
	const int width = in.getWidth();
	const int height = in.getHeight();
	const int nbChannels = in.getNbChannels();

	const unsigned char *pixelsIn = in.getPtr();
	unsigned char *pixelsOut = out.getPtr();

	const std::vector<float> &kernelConv = conv.kernel;
	const int kernelDim = conv.dimension;

	ChronoCPU chr;
	chr.start();

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			float3 sum{0.f};
			for (int j = 0; j < kernelDim; ++j)
			{
				for (int i = 0; i < kernelDim; ++i)
				{
					int dX = clamp(x + i - kernelDim / 2, 0, width - 1);
					int dY = clamp(y + j - kernelDim / 2, 0, height - 1);

					const int idKernel = j * kernelDim + i;
					const int idPixelIn = nbChannels * (dX + dY * width);
					sum.x += (float)pixelsIn[idPixelIn] * kernelConv[idKernel];
					sum.y += (float)pixelsIn[idPixelIn + 1] * kernelConv[idKernel];
					sum.z += (float)pixelsIn[idPixelIn + 2] * kernelConv[idKernel];
				}
			}
			const int idPixel = nbChannels * (x + y * width);
			pixelsOut[idPixel] = (unsigned char)clamp(sum.x, 0.f, 255.f);
			pixelsOut[idPixel + 1] = (unsigned char)clamp(sum.y, 0.f, 255.f);
			pixelsOut[idPixel + 2] = (unsigned char)clamp(sum.z, 0.f, 255.f);
			if (nbChannels == 4)
				pixelsOut[idPixel + 3] = pixelsIn[idPixel + 3];
		}
	}

	chr.stop();

	return chr.elapsedTime();
}