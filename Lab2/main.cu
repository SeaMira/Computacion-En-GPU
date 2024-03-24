// DO NOT MODIFY THIS FILE !!!

#include <iostream>
#include <string>
#include <iomanip>

#include "utils/image.hpp"

#include "reference.hpp"
#include "student.hpp"
#include "convolutionKernels.hpp"

void printUsage()
{
	std::cerr << "Usage: " << std::endl
			  << " \t -f <F>: <F> image file name" << std::endl
			  << " \t -k <K>: <K> convolution kernel" << std::endl
			  << " \t\t - 0 : BUMP 3x3" << std::endl
			  << " \t\t - 1 : SHARPEN 5x5" << std::endl
			  << " \t\t - 2 : EDGE DETECTION 7x7" << std::endl
			  << " \t\t - 3 : MOTION BLUR 9x9" << std::endl
			  << " \t\t - 4 : GAUSSIAN BLUR 11x11" << std::endl
			  << " \t -c <C>: use constant memory ? 0/1"
			  << std::endl
			  << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	char fileName[2048];
	int convType = -1;
	int useConstantMemory = 0;

	// Parse program arguments
	if (argc != 7)
	{
		printUsage();
	}

	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-f"))
		{
			if (sscanf(argv[++i], "%s", &fileName) != 1)
				printUsage();
		}
		else if (!strcmp(argv[i], "-k"))
		{
			if (sscanf(argv[++i], "%d", &convType) != 1)
				printUsage();
			if (convType < 0 || convType > 4)
				printUsage();
		}
		else if (!strcmp(argv[i], "-c"))
		{
			if (sscanf(argv[++i], "%d", &useConstantMemory) != 1)
				printUsage();
			if (useConstantMemory < 0 || useConstantMemory > 1)
				printUsage();
		}
		else
			printUsage();
	}

	ConvolutionKernel conv;
	switch (convType)
	{
	case 0:
	default:
		conv = CONVOLUTION_BUMP;
		break;
	case 1:
		conv = CONVOLUTION_SHARPEN;
		break;
	case 2:
		conv = CONVOLUTION_EDGE_DETECTION;
		break;
	case 3:
		conv = CONVOLUTION_MOTION_BLUR;
		break;
	case 4:
		conv = CONVOLUTION_GAUSSIAN_BLUR;
		break;
	}

	// ================================================================================================================
	// Get input image
	std::cout << "Loading image: " << fileName << std::endl;
	const Image input(fileName);
	const int width = input.getWidth();
	const int height = input.getHeight();
	const int nbChannels = input.getNbChannels();

	std::cout << "Image has " << width << " x " << height << " pixels" << std::endl;

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

	const float timeCPU = convolutionCPU(input, outCPU, conv);

	std::string cpuName = baseSaveName;
	cpuName += conv.name;
	cpuName += "_CPU.png";
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

	const float timeGPU = convolutionGPU(input, outGPU, conv, useConstantMemory);

	std::string gpuName = baseSaveName;
	gpuName += conv.name;
	gpuName += "_GPU";
	if (useConstantMemory)
		gpuName += "_constant";
	gpuName += ".png";
	std::cout << "Save GPU result: " << gpuName << std::endl;
	outGPU.saveToPng(gpuName.c_str());

	std::cout << "-> Done : " << timeGPU << " ms" << std::endl
			  << std::endl;

	// // ================================================================================================================

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
	std::cout << "   Times recapitulation (only convolution)  " << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "-> CPU: " << std::fixed << std::setprecision(2) << timeCPU << " ms" << std::endl;
	std::cout << "-> GPU" << (useConstantMemory ? " (constant)" : "") << ": "
			  << std::fixed << std::setprecision(2) << timeGPU << " ms" << std::endl;

	return EXIT_SUCCESS;
}