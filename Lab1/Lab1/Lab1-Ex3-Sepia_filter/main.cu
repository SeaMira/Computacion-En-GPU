// DO NOT MODIFY THIS FILE !!!

#include <iostream>
#include <string>
#include <iomanip>

#include "utils/image.hpp"

#include "reference.hpp"
#include "student.hpp"

void printUsage()
{
	std::cerr << "Usage: " << std::endl
			  << " \t -f <F>: <F> image file name"
			  << std::endl
			  << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	char fileName[2048];

	// Parse program arguments
	if (argc != 3)
	{
		printUsage();
	}

	if (!strcmp(argv[1], "-f"))
	{
		if (sscanf(argv[2], "%s", &fileName) != 1)
			printUsage();
	}
	else
		printUsage();

	// ================================================================================================================
	// Get input image
	std::cout << "Loading image: " << fileName << std::endl;
	const Image input(fileName);
	if (input.getPtr() == nullptr)
	{
		std::cerr << "Cannot load the image." << std::endl;
		return EXIT_FAILURE;
	}
	
	const int width = input.getWidth();
	const int height = input.getHeight();
	const int nbChannels = input.getNbChannels();

	std::cout << "Image has " << width << " x " << height << " pixels with " << nbChannels << " channels." << std::endl;

	std::string baseSaveName = fileName;
	size_t lastPoint = baseSaveName.find_last_of(".");
	baseSaveName.erase(lastPoint); // erase extension

	// Create 2 output images
	Image outCPU(width, height, nbChannels);
	Image outGPU(width, height, nbChannels);

	// ================================================================================================================

	// ================================================================================================================
	// CPU sequential
	std::cout << "============================================" << std::endl;
	std::cout << "         Sequential version on CPU          " << std::endl;
	std::cout << "============================================" << std::endl;

	const float timeCPU = sepiaCPU(input, outCPU);

	std::string cpuName = baseSaveName;
	cpuName += "_sepia_CPU.png";
	std::cout << "Save CPU result: " << cpuName << std::endl;
	outCPU.saveToPng(cpuName.c_str());

	std::cout << "-> Done : " << timeCPU << " ms" << std::endl
			  << std::endl;

	// ================================================================================================================

	// ================================================================================================================
	// GPU CUDA
	std::cout << "============================================" << std::endl;
	std::cout << "         Parallel version on GPU            " << std::endl;
	std::cout << "============================================" << std::endl;

	const float timeGPU = sepiaGPU(input, outGPU);

	std::string gpuName = baseSaveName;
	gpuName += "_sepia_GPU.png";
	std::cout << "Save GPU result: " << gpuName << std::endl;
	outGPU.saveToPng(gpuName.c_str());

	std::cout << "-> Done : " << timeGPU << " ms" << std::endl
			  << std::endl;

	// ================================================================================================================

	std::cout << "============================================" << std::endl;
	std::cout << "         Checking results		             " << std::endl;
	std::cout << "============================================" << std::endl;

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const int idPixel = nbChannels * (x + y * width);

			unsigned char *pixelCPU = outCPU.getPtr() + idPixel;
			unsigned char *pixelGPU = outGPU.getPtr() + idPixel;

			// Result may be slightly different between CPU and GPU because of the floating-point calculation
			if (abs(pixelCPU[0] - pixelGPU[0]) > 1 || abs(pixelCPU[1] - pixelGPU[1]) > 1 || abs(pixelCPU[2] - pixelGPU[2]) > 1)
			{
				std::cerr << "Error for pixel [" << x << ";" << y << "]: " << std::endl;
				std::cerr << "\t CPU: [" << (int)pixelCPU[0] << ";"
						  << (int)pixelCPU[1] << ";"
						  << (int)pixelCPU[2] << "]" << std::endl;
				std::cerr << "\t GPU: [" << (int)pixelGPU[0] << ";"
						  << (int)pixelGPU[1] << ";"
						  << (int)pixelGPU[2] << "]" << std::endl;
				std::cerr << "Retry!" << std::endl
						  << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	std::cout << "Congratulations! Job's done!" << std::endl
			  << std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << "     Times recapitulation (only filter)     " << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "-> CPU: " << std::fixed << std::setprecision(2) << timeCPU << " ms" << std::endl;
	std::cout << "-> GPU: " << std::fixed << std::setprecision(2) << timeGPU << " ms" << std::endl;

	return EXIT_SUCCESS;
}