#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#include <iostream>
#include "file_loader.hpp"
#include "utils/image.hpp"


void printUsage()
{
	std::cerr << "Usage: " << std::endl
			  << " \t -f <F>: <F> image file name"
			  << std::endl
			  << std::endl;
	exit(EXIT_FAILURE);
}


int main(int argc, char **argv) {

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

    std::vector<cl::Platform> default_platforms; cl::Platform::get(&default_platforms);

    if (default_platforms.size() == 0){
        std::cout << " Error finding the platform. Check OpenCL installation!\n";
        exit(1);
    }
    
    Image outGPU(width, height, nbChannels);

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

    std::string src_code = load_from_file("sepia_filter.cl");  

    cl::Program::Sources sources;
    sources.push_back({src_code.c_str(), src_code.length()});

    cl::Program program(context, sources);
    if (program.build({DEFAULT_DEVICE}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(DEFAULT_DEVICE) << "\n";
        exit(1);
    }

    ////////////////////////////////////////////

    unsigned char *inPtr = input.getPtr();
	unsigned char *outPtr = outGPU.getPtr();

	const int sizeImg = input.getSize();

    // std::cout << inPtr << "img," << sizeImg << "\n";
    std::cout << width << " " << height << " " << nbChannels << "\n";

    cl::Buffer dev_inPtr(context, CL_MEM_READ_WRITE, sizeof(float) * sizeImg);
    cl::Buffer dev_outPtr(context, CL_MEM_READ_WRITE, sizeof(float) * sizeImg);

    //CommandQueue
    cl::CommandQueue queue(context, DEFAULT_DEVICE); //create queue to which we will push commands for the device.
    
    queue.enqueueWriteBuffer(dev_inPtr, CL_TRUE, 0, sizeof(unsigned char) * sizeImg, inPtr); //copy data from arrays A and B to buffer_A and buffer_B which represent memory on the device:

    cl::Kernel kernel_sepia_filter = cl::Kernel(program, "kernelComputeSepia");
    kernel_sepia_filter.setArg(0, dev_inPtr); 
    kernel_sepia_filter.setArg(1, dev_outPtr);
    kernel_sepia_filter.setArg(2, width);
    kernel_sepia_filter.setArg(3, height);
    kernel_sepia_filter.setArg(4, nbChannels);
    cl::NDRange global(width, height);
    queue.enqueueNDRangeKernel(kernel_sepia_filter, cl::NullRange, global);
    queue.finish();
    std::cout << "aaa\n";

    queue.enqueueReadBuffer(dev_outPtr, CL_TRUE, 0, sizeof(unsigned char)*sizeImg, outPtr);
    queue.flush();
    clReleaseKernel(kernel_sepia_filter());
    clReleaseProgram(program());
    clReleaseMemObject(dev_inPtr());
    clReleaseMemObject(dev_outPtr());
    clReleaseCommandQueue(queue());
    clReleaseContext(context());

    std::string baseSaveName = fileName;
	size_t lastPoint = baseSaveName.find_last_of(".");
	baseSaveName.erase(lastPoint); // erase extension
    std::string gpuName = baseSaveName;
    gpuName += "_sepia_GPU.png";
    std::cout << gpuName << "\n";
    outGPU.saveToPng(gpuName.c_str());

};