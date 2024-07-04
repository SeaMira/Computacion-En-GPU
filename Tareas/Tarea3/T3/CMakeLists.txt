cmake_minimum_required(VERSION 3.16.3)
project(surfaceLevelCurves)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)


find_package(OpenGL REQUIRED)
find_package(GLFW3 REQUIRED)

add_subdirectory(extern)
add_subdirectory(src)




target_link_libraries(${PROJECT_NAME} surfaceLeveldependencies)