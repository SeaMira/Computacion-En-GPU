#include<iostream>
#include<random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <ctime>
#include <fstream>
#include <sstream>
#include <memory>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#define LIST_SIZE 100000u
#define MAX_SOURCE_SIZE (0x100000)


std::string load_from_file(const std::string &path)
{

    auto close_file = [](FILE *f) { fclose(f); };

    auto holder = std::unique_ptr<FILE, decltype(close_file)>(fopen(path.c_str(), "rb"), close_file);
    if (!holder)
        return "";

    FILE *f = holder.get();

    // in C++17 following lines can be folded into std::filesystem::file_size invocation
    if (fseek(f, 0, SEEK_END) < 0)
        return "";

    const long size = ftell(f);
    if (size < 0)
        return "";

    if (fseek(f, 0, SEEK_SET) < 0)
        return "";

    std::string res;
    res.resize(size);

    // C++17 defines .data() which returns a non-const pointer
    fread(const_cast<char *>(res.data()), 1, size, f);

    return res;
}

int main() {
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

    ///////////// kernel setup /////////////
    // FILE *fp;
    // char *src_code;
    // size_t src_size;

    // fp = fopen("vector_add_kernel.cl", "r");
    // if (!fp) {
    //     fprintf(stderr, "Failed to load kernel.\n");
    //     exit(1);
    // }
    // src_code = (char*)malloc(MAX_SOURCE_SIZE);
    // src_size = fread( src_code, 1, MAX_SOURCE_SIZE, fp);
    // fclose( fp );
    std::string src_code = load_from_file("vector_add_kernel.cl");  

    std::cout << src_code << "\n";

    cl::Program::Sources sources;
    sources.push_back({src_code.c_str(), src_code.length()});

    cl::Program program(context, sources);
    if (program.build({DEFAULT_DEVICE}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(DEFAULT_DEVICE) << "\n";
        exit(1);
    }
    ////////////////////////////////////////
    
    // Semilla para la generación de números aleatorios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100); // Distribución uniforme entre 1 y 100

    // Crear los vectores
    int vector1[LIST_SIZE];
    int vector2[LIST_SIZE];

    // Llenar los vectores con valores aleatorios
    for (int i = 0; i < LIST_SIZE; ++i) { // Puedes cambiar el tamaño del vector aquí (10 en este caso)
        vector1[i] = dis(gen); // Agrega un número aleatorio al vector1
        vector2[i] = dis(gen); // Agrega un número aleatorio al vector2
    }

    std::clock_t start_time;
    double GPUtime1;
    start_time = std::clock();

    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * LIST_SIZE);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * LIST_SIZE);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * LIST_SIZE);

    //CommandQueue
    cl::CommandQueue queue(context, DEFAULT_DEVICE); //create queue to which we will push commands for the device.
    
    //Adding to CommandQueue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * LIST_SIZE, vector1); //copy data from arrays A and B to buffer_A and buffer_B which represent memory on the device:
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * LIST_SIZE, vector2);

    cl::Kernel kernel_add = cl::Kernel(program, "vector_add");
    kernel_add.setArg(0, buffer_A); 
    kernel_add.setArg(1, buffer_B); 
    kernel_add.setArg(2, buffer_C);
    queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(LIST_SIZE), cl::NDRange(32));
    queue.finish();


    int C[LIST_SIZE];
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*LIST_SIZE, C);

    GPUtime1 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    std::cout << "GPU version took:\n" << GPUtime1 << "\n";

    queue.flush();
    queue.finish();
    clReleaseKernel(kernel_add());
    clReleaseProgram(program());
    clReleaseMemObject(buffer_A());
    clReleaseMemObject(buffer_B());
    clReleaseMemObject(buffer_C());
    clReleaseCommandQueue(queue());
    clReleaseContext(context());

    std::vector<int> vector3;

    std::clock_t start_time2;
    double GPUtime2;
    start_time2 = std::clock();

    for (int i = 0; i < LIST_SIZE; i++) {
        vector3.push_back(vector1[i] +vector2[i]); 
    }

    GPUtime2 = (std::clock() - start_time2) / (double) CLOCKS_PER_SEC;

    std::cout << "CPU version took:\n" << GPUtime2 << "\n";

    for (int i = 0; i < LIST_SIZE; i++) {
        // std::cout << C[i] << ", " << vector3[i] << "\n";
        if (C[i] != vector3[i]) {
            std::cout << "No son iguales los resultados " << i << "\n";
            return 0;
        }
    }
    std::cout << "Se termino exitosamente!" << "\n";
    
    // free(src_code);
}