#ifndef __STUDENT_HPP__
#define __STUDENT_HPP__

#include "utils/image.hpp"
#include "convolutionKernels.hpp"

// ============================================ Naive
__global__ void kernelConvolution(unsigned char *pixelsOut, const unsigned char *pixelsIn,
								  const int imgWidth, const int imgHeight, const int nbChannels,
								  const float *kernelConv, const int kernelDim);

// ============================================ Constant memory
__global__ void kernelConvolutionConstant(unsigned char *pixelsOut, const unsigned char *pixelsIn,
										  const int imgWidth, const int imgHeight, const int nbChannels,
										  const int kernelDim);

// ============================================ Launch convolution
float convolutionGPU(const Image &in, Image &out, const ConvolutionKernel &conv, const int useConstantMemory);

#endif // __STUDENT_HPP__