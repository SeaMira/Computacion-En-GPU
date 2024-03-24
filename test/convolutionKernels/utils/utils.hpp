#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda.h>

template <class T>
__host__ __device__ static T clamp(const T &val, const T &valMin, const T &valMax)
{
#ifdef __CUDACC__
	return min(valMax, max(valMin, val));
#else
	return std::min<T>(valMax, std::max<T>(valMin, val));
#endif
}

#endif // __UTILS_HPP__
