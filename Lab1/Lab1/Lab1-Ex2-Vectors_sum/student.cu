#include "student.hpp"

__host__ void allocateArraysDevice(const size_t n, int **dev_a, int **dev_b, int **dev_res)
{
	/// TODO
}

__host__ void copyFromHostToDevice(const size_t n, const int *a, const int *b, int *dev_a, int *dev_b)
{
	/// TODO
}

__host__ void launchKernel(const size_t n, const int *const dev_a, const int *dev_b, int *dev_res)
{
	/// TODO
}

__host__ void copyFromDeviceToHost(const size_t n, int *res, const int *dev_res)
{
	/// TODO
}

__host__ void freeArraysCUDA(int *dev_a, int *dev_b, int *dev_res)
{
	/// TODO
}

__global__ void sumArraysCUDA(const size_t n, const int *dev_a, const int *dev_b, int *dev_res)
{
	/// TODO
}
