// DO NOT MODIFY THIS FILE !!!

#include "reference.hpp"

void sumArraysRef(const size_t n, const int *a, const int *b, int *res)
{
	for (size_t i = 0; i < n; ++i)
		res[i] = a[i] + b[i];
}
