// DO NOT MODIFY THIS FILE !!!
#include <algorithm>

#include "reference.hpp"
#include "utils/chronoCPU.hpp"

float sepiaCPU(const Image &in, Image &out)
{
	const int width 	 = in.getWidth();
	const int height 	 = in.getHeight();
	const int nbChannels = in.getNbChannels();

	unsigned char *pixelsIn = in.getPtr();
	unsigned char *pixelsOut = out.getPtr();

	ChronoCPU chr;
	chr.start();

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			const int idPixel = nbChannels * (i + j * width);

			const unsigned char inRed = pixelsIn[idPixel];
			const unsigned char inGreen = pixelsIn[idPixel + 1];
			const unsigned char inBlue = pixelsIn[idPixel + 2];

			const unsigned char outRed = (unsigned char)std::min<float>(255.f, (inRed * .393f + inGreen * .769f + inBlue * .189f));
			const unsigned char outGreen = (unsigned char)std::min<float>(255.f, (inRed * .349f + inGreen * .686f + inBlue * .168f));
			const unsigned char outBlue = (unsigned char)std::min<float>(255.f, (inRed * .272f + inGreen * .534f + inBlue * .131f));

			pixelsOut[idPixel] = outRed;
			pixelsOut[idPixel + 1] = outGreen;
			pixelsOut[idPixel + 2] = outBlue;
			if (nbChannels == 4)
				pixelsOut[idPixel + 3] = pixelsIn[idPixel + 3];
		}
	}

	chr.stop();

	return chr.elapsedTime();
}