
#include "student.hpp"
#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"
#include "utils/utils.hpp"

#define MAX_CONV_SIZE 121

// ============================================ Naive
__global__ void kernelConvolution(unsigned char *pixelsOut, const unsigned char *pixelsIn,
								  const int imgWidth, const int imgHeight, const int nbChannels,
								  const float *kernelConv, const int kernelDim)
{
	/// TODO
}

// ============================================ Constant memory
__global__ void kernelConvolutionConstant(unsigned char *pixelsOut, const unsigned char *pixelsIn,
										  const int imgWidth, const int imgHeight, const int nbChannels,
										  const int kernelDim)
{
	/// TODO
}

float convolutionGPU(const Image &in, Image &out, const ConvolutionKernel &conv, const int useConstantMemory)
{
	// Image
	unsigned char *inPtr = in.getPtr();
	unsigned char *outPtr = out.getPtr();
	unsigned char *dev_inPtr = nullptr;
	unsigned char *dev_outPtr = nullptr;

	const int sizeImg = in.getSize();

	const int width = in.getWidth();
	const int height = in.getHeight();
	const int nbChannels = in.getNbChannels();

	// Convolution kernel
	const float *convKernel = conv.kernel.data();
	float *dev_convKernel = nullptr;
	const int convKernelDim = conv.dimension;
	const int sizeConvKernel = convKernelDim * convKernelDim * sizeof(float);

	// Allocate memory on Device
	/// TODO

	// Copy from Host to Device
	/// TODO
	// if (useConstantMemory)
	// else

	// Configure kernel
	/// TODO

	ChronoGPU chr;
	chr.start();

	// Call kernel
	/// TODO
	// if (useConstantMemory)
	// else

	chr.stop();

	// Copy from Device to Host
	/// TODO

	// Free memory on Device
	/// TODO

	return chr.elapsedTime();
}
