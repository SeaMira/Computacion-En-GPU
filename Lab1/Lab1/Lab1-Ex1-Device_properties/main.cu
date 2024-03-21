#include <iostream>
#include <cstdlib>

#include "deviceProperties.hpp"

int main() 
{
	int cptDevice = countDevices();

	std::cout << "-> " << cptDevice << " CUDA capable device(s)" << std::endl << std::endl;
	
	for ( int i = 0; i < cptDevice; ++i )
	{
		printDeviceProperties( i );
	}
	
	return EXIT_SUCCESS;
}