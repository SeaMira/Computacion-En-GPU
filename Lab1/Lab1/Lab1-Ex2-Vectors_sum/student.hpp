#ifndef __STUDENT_HPP__
#define __STUDENT_HPP__

#include "utils/commonCUDA.hpp"

__host__ void allocateArraysDevice(const size_t n,			 // vectors size
								   int **dev_a, int **dev_b, // 2 vectors on Device
								   int **dev_res);			 // result on Device

__host__ void copyFromHostToDevice(const size_t n,			   // vectors size
								   const int *a, const int *b, // 2 vectors on Host
								   int *dev_a, int *dev_b);	   // 2 vectors on Device

__host__ void launchKernel(const size_t n,					   // vectors size
						   const int *dev_a, const int *dev_b, // vectors to sum
						   int *dev_res);					   // result

__host__ void copyFromDeviceToHost(const size_t n,		// vectors size
								   int *res,			// result on Host
								   const int *dev_res); // result on Device

__host__ void freeArraysCUDA(int *dev_a, int *dev_b, int *dev_res); // 3 vectors to free

// Kernel
__global__ void sumArraysCUDA(const size_t n,					  // vectors size
							  const int *dev_a, const int *dev_b, // vectors to sum
							  int *dev_res);					  // result

#endif // __STUDENT_HPP__
