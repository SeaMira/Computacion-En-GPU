#include "student.hpp"

#include "utils/chronoGPU.hpp"

__global__ void kernelComputeSepia(const unsigned char *pixelsIn, unsigned char *pixelsOut,
								   const int width, const int height, const int nbChannels)
{
	for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
	{
		for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
		{
			const int idPixel = nbChannels * (x + y * width);

			const unsigned char inRed = pixelsIn[idPixel];
			const unsigned char inGreen = pixelsIn[idPixel + 1];
			const unsigned char inBlue = pixelsIn[idPixel + 2];

			const unsigned char outRed = (unsigned char)fminf(255.f, (inRed * .393f + inGreen * .769f + inBlue * .189f));
			const unsigned char outGreen = (unsigned char)fminf(255.f, (inRed * .349f + inGreen * .686f + inBlue * .168f));
			const unsigned char outBlue = (unsigned char)fminf(255.f, (inRed * .272f + inGreen * .534f + inBlue * .131f));

			pixelsOut[idPixel] = outRed;
			pixelsOut[idPixel + 1] = outGreen;
			pixelsOut[idPixel + 2] = outBlue;
			if (nbChannels == 4)
				pixelsOut[idPixel + 3] = pixelsIn[idPixel + 3];
		}
	}
}

float sepiaGPU(const Image &in, Image &out)
{
	// Pixel table on Host
	unsigned char *inPtr = in.getPtr();
	unsigned char *outPtr = out.getPtr();

	// Pixel table on Device
	unsigned char *dev_inPtr = nullptr;
	unsigned char *dev_outPtr = nullptr;

	const int sizeImg = in.getSize();

	const int width = in.getWidth();
	const int height = in.getHeight();
	const int channels = in.getNbChannels();

	// Allocate memory on Device
	HANDLE_ERROR(cudaMalloc((void **)&dev_inPtr, sizeImg));
	HANDLE_ERROR(cudaMalloc((void **)&dev_outPtr, sizeImg));

	// Copy from Host to Device
	HANDLE_ERROR(cudaMemcpy(dev_inPtr, inPtr, sizeImg, cudaMemcpyHostToDevice));

	// Configure kernel
	const dim3 threads(32, 32);
	const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

	ChronoGPU chr;
	chr.start();

	// Launch kernel
	kernelComputeSepia<<<blocks, threads>>>(dev_inPtr, dev_outPtr, width, height, channels);

	chr.stop();

	// Copy from Device to Host
	HANDLE_ERROR(cudaMemcpy(outPtr, dev_outPtr, sizeImg, cudaMemcpyDeviceToHost));

	// Free memory on Device
	HANDLE_ERROR(cudaFree(dev_inPtr));
	HANDLE_ERROR(cudaFree(dev_outPtr));

	return chr.elapsedTime();
}
