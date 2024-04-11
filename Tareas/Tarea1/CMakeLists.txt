# cmake_minimum_required(VERSION 3.20)

# project(MatrixLib)

# set(CMAKE_CXX_STANDARD 23) # Importante para la tarea!!
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Importante para un buen autocompletado!!

# set(MATRIXLIB_H
#   "${PROJECT_SOURCE_DIR}/include/matrix/Matrix.h"
# )

# set(MATRIXLIB_SRC 
#   "${PROJECT_SOURCE_DIR}/src/Matrix.cpp"
# )

# add_library(${PROJECT_NAME} INTERFACE ${MATRIXLIB_H})
# add_library(${PROJECT_NAME} ${MATRIXLIB_SRC})
# target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(${PROJECT_NAME} matrix)

# include(CTest)
# add_subdirectory(extern)
# add_subdirectory(test)
# add_subdirectory(src)


cmake_minimum_required(VERSION 3.20)
project(MatrixLib)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Asume que Matrix.cpp está en src y Matrix.h en include
add_library(matrix_lib src/Matrix.cpp)
target_include_directories(matrix_lib PUBLIC include)

include(CTest)
add_subdirectory(src)
add_subdirectory(extern/googletest) # Asegúrate de tener esto correctamente apuntando a googletest
add_subdirectory(test)

# cmake_minimum_required(VERSION 3.20)

# project(MatrixLib)

# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(MATRIXLIB_H
#   "${PROJECT_SOURCE_DIR}/include/matrix/Matrix.h"
# )

# add_library(${PROJECT_NAME} INTERFACE ${MATRIXLIB_H})

# target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)

# include(CTest)
# add_subdirectory(src)
# add_subdirectory(extern)
# add_subdirectory(test)
