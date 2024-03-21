#include <iostream>
#include <vector>
#include <CL/cl.hpp>

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No se encontraron plataformas OpenCL.\n";
        return 1;
    }

    std::cout << "Plataformas OpenCL encontradas:\n";
    for (const auto& platform : platforms) {
        std::cout << "Nombre: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (devices.empty()) {
            std::cerr << "No se encontraron dispositivos para esta plataforma.\n";
        } else {
            std::cout << "Dispositivos:\n";
            for (const auto& device : devices) {
                std::cout << "  - Nombre: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            }
        }
    }

    return 0;
}
