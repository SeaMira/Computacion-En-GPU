#include<iostream>
#include "file_loader.hpp"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>

int NUM_ELEMENTS = 4096*4096, LOCAL_SIZE = 256;
size_t numWorkGroups = NUM_ELEMENTS / LOCAL_SIZE;

int main() {


    float * hA = new float [ NUM_ELEMENTS ];
    float * hB = new float [ NUM_ELEMENTS ];
    float * hC = new float [ numWorkGroups ];

    for( int i = 0; i < NUM_ELEMENTS; ++i ) {
        hA[i] = i;
        hB[i] = i;
    }

    size_t abSize = NUM_ELEMENTS * sizeof(float);
    size_t cSize = numWorkGroups * sizeof(float);


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

    std::string src_code = load_from_file("kernel.cl");
    cl::Program::Sources sources;
    sources.push_back({src_code.c_str(), src_code.length()});

    cl::Program program(context, sources);
    if (program.build({DEFAULT_DEVICE}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(DEFAULT_DEVICE) << "\n";
        exit(1);
    }
    // ================================================================================================================

    cl_int status;

    // cl_mem dA = clCreateBuffer( context, CL_MEM_READ_ONLY, abSize, NULL, &status );
    cl::Buffer dA(context, CL_MEM_READ_ONLY, abSize, NULL);
    // cl_mem dB = clCreateBuffer( context, CL_MEM_READ_ONLY, abSize, NULL, &status );
    cl::Buffer dB(context, CL_MEM_READ_ONLY, abSize, NULL);
    // cl_mem dC = clCreateBuffer( context, CL_MEM_WRITE_ONLY, cSize, NULL, &status );
    cl::Buffer dC(context, CL_MEM_WRITE_ONLY, cSize, NULL);

    cl::CommandQueue queue(context, DEFAULT_DEVICE); //create queue to which we will push commands for the device.

    status = queue.enqueueWriteBuffer(dA, CL_FALSE, 0, abSize, hA, 0, NULL);
    status = queue.enqueueWriteBuffer(dB, CL_FALSE, 0, abSize, hB, 0, NULL);

    std::cout << "sum" << std::endl;
    cl::Kernel kernel = cl::Kernel(program, "ArrayMultReduce");

    status = kernel.setArg(0, dA);
    status = kernel.setArg(1, dB);
    status = kernel.setArg(2, LOCAL_SIZE * sizeof(float), NULL); // local “prods” array is dimensioned the size of each work-group
    status = kernel.setArg(3, dC );

    cl::NDRange global( NUM_ELEMENTS );
    cl::NDRange local( 256 );
    status = queue.enqueueNDRangeKernel(kernel, 0, global, local, NULL, NULL);
    queue.finish();


    queue.enqueueReadBuffer(dC, CL_TRUE, 0, numWorkGroups*sizeof(float), hC);
    queue.flush();

    float sum = 0.;
    std::cout << hA[ NUM_ELEMENTS-1 ] << std::endl;
    std::cout << hB[ NUM_ELEMENTS-1 ] << std::endl;
    for( int i = 0; i < numWorkGroups; i++ )
    {
        // std::cout << hC[ i ] << std::endl;
        sum += hC[ i ];
    }
    std::cout << sum << std::endl;

    // float sum2 = 0;
    // for( int i = 0; i < NUM_ELEMENTS; i++ )
    // {
    //     sum2 += hA[ i ]*hB[ i ];
    // }
    // std::cout << sum2 << std::endl;

}
#endif