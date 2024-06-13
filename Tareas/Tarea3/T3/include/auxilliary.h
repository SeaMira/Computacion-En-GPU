#include <string>
#include <cstdlib> 
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <memory>

void init_values(int x_limit, int y_limit, int z_limit, float *arr, int arr_size);
std::string load_from_file(const std::string &path);