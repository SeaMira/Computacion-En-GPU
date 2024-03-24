// DO NOT MODIFY THIS FILE !!!

#ifndef __REFERENCE_HPP__
#define __REFERENCE_HPP__

#include "utils/commonCUDA.hpp"
#include "utils/image.hpp"
#include "convolutionKernels.hpp"

float convolutionCPU(const Image &in, Image &out, const ConvolutionKernel &conv);

#endif // __REFERENCE_HPP__
