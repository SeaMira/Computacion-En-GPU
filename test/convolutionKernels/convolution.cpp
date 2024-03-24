#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#include <iostream>
#include "file_loader.hpp"
#include "utils/image.hpp"

#include "convolutionKernels.hpp"


void printUsage() {
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

int main(int argc, char **argv) {
    char fileName[2048];
	int convType = -1;
	int useConstantMemory = 0;

	// Parse program arguments
	if (argc != 7) {
		printUsage();
	}

	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-f")) {
			if (sscanf(argv[++i], "%s", &fileName) != 1)
				printUsage();
		}
		else if (!strcmp(argv[i], "-k")) {
			if (sscanf(argv[++i], "%d", &convType) != 1)
				printUsage();
			if (convType < 0 || convType > 4)
				printUsage();
		}
		else if (!strcmp(argv[i], "-c")) {
			if (sscanf(argv[++i], "%d", &useConstantMemory) != 1)
				printUsage();
			if (useConstantMemory < 0 || useConstantMemory > 1)
				printUsage();
		}
		else
			printUsage();
	}

	ConvolutionKernel conv;
	switch (convType) {
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
	Image outGPU(width, height, nbChannels);

	// ================================================================================================================

    unsigned char *inPtr = input.getPtr();
	unsigned char *outPtr = outGPU.getPtr();

    const int sizeImg = input.getSize();

    const float *convKernel = conv.kernel.data();
	const int convKernelDim = conv.dimension;
	const int sizeConvKernel = convKernelDim * convKernelDim * sizeof(float);
    // ========================================PROGRAM==============================================================
    std::vector<cl::Platform> default_platforms; cl::Platform::get(&default_platforms);

    if (default_platforms.size() == 0){
        std::cout << " Error finding the platform. Check OpenCL installation!\n";
        exit(1);
    }
    
    cl::Platform DEFAULT_PLATFORM = default_platforms[0];
    std::cout << "Using platform: " << DEFAULT_PLATFORM.getInfo<CL_PLATFORM_NAME>() << "\n";

    std::vector<cl::Device> all_devices; DEFAULT_PLATFORM.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    cl::Device DEFAULT_DEVICE = all_devices[0];
    std::cout << "Using device: " << DEFAULT_DEVICE.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context({DEFAULT_DEVICE});

    std::string src_code = load_from_file("convolutions.cl");  

    cl::Program::Sources sources;
    sources.push_back({src_code.c_str(), src_code.length()});

    cl::Program program(context, sources);
    if (program.build({DEFAULT_DEVICE}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(DEFAULT_DEVICE) << "\n";
        exit(1);
    }
    // ================================================================================================================

    cl::Buffer dev_inPtr(context, CL_MEM_READ_ONLY, sizeof(float) * sizeImg);
    cl::Buffer dev_convKernel(context, CL_MEM_READ_ONLY, sizeConvKernel);
    cl::Buffer dev_outPtr(context, CL_MEM_WRITE_ONLY, sizeof(float) * sizeImg);

    //CommandQueue
    cl::CommandQueue queue(context, DEFAULT_DEVICE); //create queue to which we will push commands for the device.

    queue.enqueueWriteBuffer(dev_inPtr, CL_TRUE, 0, sizeof(unsigned char) * sizeImg, inPtr);
    queue.enqueueWriteBuffer(dev_convKernel, CL_TRUE, 0, sizeConvKernel, convKernel);

    cl::Kernel kernel_convolution = cl::Kernel(program, "kernelConvolution");
    kernel_convolution.setArg(0, dev_inPtr); 
    kernel_convolution.setArg(1, dev_outPtr);
    kernel_convolution.setArg(2, width);
    kernel_convolution.setArg(3, height);
    kernel_convolution.setArg(4, nbChannels);
    kernel_convolution.setArg(5, dev_convKernel);
    kernel_convolution.setArg(6, convKernelDim);
    cl::NDRange global(width, height);
    queue.enqueueNDRangeKernel(kernel_convolution, cl::NullRange, global);
    queue.finish();
    std::cout << "aaa\n";

    queue.enqueueReadBuffer(dev_outPtr, CL_TRUE, 0, sizeof(unsigned char)*sizeImg, outPtr);
    queue.flush();

    std::string gpuName = baseSaveName;
    gpuName += conv.name + ".png";
    std::cout << gpuName << "\n";
    outGPU.saveToPng(gpuName.c_str());

}