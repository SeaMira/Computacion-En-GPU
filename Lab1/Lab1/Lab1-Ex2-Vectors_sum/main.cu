// DO NOT MODIFY THIS FILE !!!

#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <random>

#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"

#include "reference.hpp"
#include "student.hpp"

void printTimeInMs(const float ms)
{
	std::cout << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
}

void printUsage(const char *prg)
{
	std::cerr << "Usage: " << prg << std::endl
			  << " \t -n <N>: <N> is the size of the vectors"
			  << std::endl
			  << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	size_t nbElements = 100000000u;

	// Parse program arguments
	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-n"))
		{
			if (sscanf(argv[++i], "%zu", &nbElements) != 1)
				printUsage(argv[0]);
		}
		else
		{
			printUsage(argv[0]);
		}
	}

	std::cout << "Summing vectors of size " << nbElements << std::endl
			  << std::endl;

	// ================================================================================================================
	// Allocation and initialization
	std::cout << "Allocating " << ((nbElements * 3 * sizeof(int)) >> 20) << " MB on Host" << std::endl;

	int *a = new int[nbElements];
	int *b = new int[nbElements];
	int *resCPU = new int[nbElements];

	std::cout << " -> Done" << std::endl
			  << std::endl;

	if (a == nullptr || b == nullptr || resCPU == nullptr)
	{
		delete[] a;
		delete[] b;
		delete[] resCPU;
		std::cerr << "Error allocating host vectors" << std::endl;
		return EXIT_FAILURE;
	}

	const int min = -100;
	const int max = 100;
	std::cout << "Initializing vectors with random integers in [" << min << "," << max << "]" << std::endl
			  << std::endl;
	std::mt19937 gen(std::time(0));
	std::uniform_int_distribution<int> dis(min, max);
	for (size_t i = 0; i < nbElements; ++i)
	{
		a[i] = dis(gen);
		b[i] = dis(gen);
	}

	// ================================================================================================================

	// ================================================================================================================
	// CPU sequential
	std::cout << "============================================" << std::endl;
	std::cout << "         Sequential version on CPU          " << std::endl;
	std::cout << "============================================" << std::endl;

	std::cout << "Summming vectors" << std::endl;
	ChronoCPU chrCPU;
	chrCPU.start();
	sumArraysRef(nbElements, a, b, resCPU);
	chrCPU.stop();

	const float timeComputeCPU = chrCPU.elapsedTime();
	std::cout << " -> Time: ";
	printTimeInMs(timeComputeCPU);
	std::cout << std::endl;

	// ================================================================================================================

	// ================================================================================================================
	// GPU CUDA
	std::cout << "============================================" << std::endl;
	std::cout << "          Parallel version on GPU           " << std::endl;
	std::cout << "============================================" << std::endl;

	int *dev_a = nullptr;
	int *dev_b = nullptr;
	int *dev_res = nullptr;

	// GPU allocation
	std::cout << "Allocating " << ((nbElements * 3 * sizeof(int)) >> 20) << " MB on Device" << std::endl;
	ChronoGPU chrGPU;
	chrCPU.start();
	allocateArraysDevice(nbElements, &dev_a, &dev_b, &dev_res);
	chrCPU.stop();

	const float timeAllocGPU = chrCPU.elapsedTime();
	std::cout << " -> Time: ";
	printTimeInMs(timeAllocGPU);
	std::cout << std::endl;

	// Copy from host to device
	std::cout << "Copying data from Host to Device" << std::endl;
	chrGPU.start();
	copyFromHostToDevice(nbElements, a, b, dev_a, dev_b);
	chrGPU.stop();

	const float timeHtoDGPU = chrGPU.elapsedTime();
	std::cout << " -> Time: ";
	printTimeInMs(timeHtoDGPU);
	std::cout << std::endl;

	// Free useless memory on CPU
	delete[] a;
	delete[] b;

	// Launch kernel
	std::cout << "Summming vectors" << std::endl;
	chrGPU.start();
	launchKernel(nbElements, dev_a, dev_b, dev_res);
	chrGPU.stop();

	const float timeComputeGPU = chrGPU.elapsedTime();
	std::cout << " -> Time: ";
	printTimeInMs(timeComputeGPU);
	std::cout << std::endl;

	// Copy from device to host
	std::cout << "Copying data from Device to Host" << std::endl;

	int *resGPU = new int[nbElements];

	chrGPU.start();

	copyFromDeviceToHost(nbElements, resGPU, dev_res);

	chrGPU.stop();
	const float timeDtoHGPU = chrGPU.elapsedTime();
	std::cout << " -> Time: ";
	printTimeInMs(timeDtoHGPU);
	std::cout << std::endl;

	// Free GPU memory
	freeArraysCUDA(dev_a, dev_b, dev_res);

	// ================================================================================================================

	std::cout << "============================================" << std::endl;
	std::cout << "              Checking results              " << std::endl;
	std::cout << "============================================" << std::endl;

	for (int i = 0; i < nbElements; ++i)
	{
		if (resCPU[i] != resGPU[i])
		{
			std::cerr << "Error at index " << i << std::endl
					  << " - CPU:  " << resCPU[i] << std::endl
					  << " - GPU: " << resGPU[i] << std::endl
					  << "Retry!" << std::endl
					  << std::endl;
			delete[] resCPU;
			delete[] resGPU;
			exit(EXIT_FAILURE);
		}
	}

	delete[] resCPU;
	delete[] resGPU;

	std::cout << "Congratulations! Job's done!" << std::endl
			  << std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << "            Times recapitulation            " << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "-> CPU	Sequential" << std::endl;
	std::cout << "   - Computation:    ";
	printTimeInMs(timeComputeCPU);
	std::cout << "-> GPU	" << std::endl;
	std::cout << "   - Allocation:     ";
	printTimeInMs(timeAllocGPU);
	std::cout << "   - Host to Device: ";
	printTimeInMs(timeHtoDGPU);
	std::cout << "   - Computation:    ";
	printTimeInMs(timeComputeGPU);
	std::cout << "   - Device to Host: ";
	printTimeInMs(timeDtoHGPU);
	std::cout << "   - Total:          ";
	printTimeInMs(timeAllocGPU + timeHtoDGPU + timeComputeGPU + timeDtoHGPU);
	std::cout << std::endl;

	return EXIT_SUCCESS;
}