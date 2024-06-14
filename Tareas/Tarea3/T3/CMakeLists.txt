cmake_minimum_required(VERSION 3.16.3)
project(surfaceLevelCurves)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)
# include(CTest)

set(OpenCL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(OpenCL_LIBRARY "C:/Windows/System32/OpenCL.dll")
find_package(OpenGL REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(OpenCL)

add_subdirectory(extern)
add_subdirectory(src)
# add_subdirectory(test)

#quitar para evitar test
# enable_testing()
# find_package(GLEW REQUIRED)



target_link_libraries(${PROJECT_NAME} surfaceLeveldependencies)