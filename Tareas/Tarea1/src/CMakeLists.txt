set(MATRIX_H_LIST "${PROJECT_SOURCE_DIR}/include/matrix/Matrix.h")

set(MATRIX_CPP_LIST "Matrix.cpp")

add_library(matrix  ${MATRIX_CPP_LIST} ${MATRIX_H_LIST})

target_include_directories(matrix PUBLIC ${PROJECT_SOURCE_DIR}/include)
