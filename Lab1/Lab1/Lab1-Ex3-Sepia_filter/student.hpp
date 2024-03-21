#ifndef __STUDENT_HPP__
#define __STUDENT_HPP__

#include "utils/image.hpp"
#include "utils/commonCUDA.hpp"

__global__ void kernelComputeSepia(const unsigned char *pixelsIn, unsigned char *pixelsOut,
								   const int width, const int height, const int nbChannels);
                                   
float sepiaGPU(const Image &in, Image &out);

#endif // __STUDENT_HPP__
