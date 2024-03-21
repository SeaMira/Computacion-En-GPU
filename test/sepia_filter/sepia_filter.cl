void kernel kernelComputeSepia(__global const unsigned char *pixelsIn, __global unsigned char *pixelsOut, const int width, const int height, const int nbChannels) {
 
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int idPixel = nbChannels * (x + y);

    for (int i = 0; i < height*width*nbChannels; i += width*nbChannels) {
        const unsigned char inRed = pixelsIn[idPixel +i];
	const unsigned char inGreen = pixelsIn[idPixel + 1 +i];
	const unsigned char inBlue = pixelsIn[idPixel + 2 +i];

    const unsigned char outRed = (unsigned char)fmin(255.f, (inRed * .393f + inGreen * .769f + inBlue * .189f));
	const unsigned char outGreen = (unsigned char)fmin(255.f, (inRed * .349f + inGreen * .686f + inBlue * .168f));
	const unsigned char outBlue = (unsigned char)fmin(255.f, (inRed * .272f + inGreen * .534f + inBlue * .131f));

    pixelsOut[idPixel +i] = outRed;
	pixelsOut[idPixel + 1 +i] = outGreen;
	pixelsOut[idPixel + 2 +i] = outBlue;

    if (nbChannels == 4)
				pixelsOut[idPixel + 3 +i] = pixelsIn[idPixel + 3 +i];
    } 

}