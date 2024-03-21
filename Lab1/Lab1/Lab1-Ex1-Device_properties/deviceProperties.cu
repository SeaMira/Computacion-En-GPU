#include "deviceProperties.hpp"
#include "commonCUDA.hpp"

int countDevices() 
{
	int cptDevice = 0;
	HANDLE_ERROR( cudaGetDeviceCount( &cptDevice ) );
	return cptDevice;
}

void printDeviceProperties( const int i ) 
{
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
		
	std::cout << "============================================"	<< std::endl;
	std::cout << "               Device " << i					<< std::endl;
	std::cout << "============================================"	<< std::endl;
	std::cout << " - name: "				<< prop.name							<< std::endl;
	std::cout << " - is a Tesla device: "	<< ( prop.tccDriver ? "yes" : "no" )	<< std::endl;
	std::cout << " - pciBusId: "			<< prop.pciBusID						<< std::endl;
	std::cout << " - pciDeviceId: "			<< prop.pciDeviceID						<< std::endl;
	std::cout << " - pciDomainId: "			<< prop.pciDomainID						<< std::endl;
    
	std::cout << "============================================" << std::endl;
	std::cout << " - compute capability max: "		<< prop.major << "." << prop.minor					<< std::endl;
	std::cout << " - is integrated: "				<< ( prop.integrated ? "yes" : "no" )				<< std::endl;
	std::cout << " - max kernel execution time: "	<< ( prop.kernelExecTimeoutEnabled ? "yes" : "no" )	<< std::endl;
	std::cout << " - device can overlap: "			<< ( prop.deviceOverlap ?	"yes" : "no" )			<< std::endl;
	std::cout << " - concurrent kernels allowed: "	<< ( prop.concurrentKernels ? "yes" : "no" )		<< std::endl; 
	std::cout << " - compute mode: "				<< ( prop.computeMode ? "yes" : "no" )				<< std::endl;
	
	std::cout << "============================================" << std::endl;
	std::cout << " - total Mem: "				<< ( prop.totalGlobalMem >> 20 ) << " Mo"		<< std::endl;
	std::cout << " - shared Mem: "				<< ( prop.sharedMemPerBlock >> 10 ) << " Ko"	<< std::endl;
	std::cout << " - total constant memory: "	<< ( prop.totalConstMem >> 10 ) << " Ko"		<< std::endl;
	std::cout << " - memory pitch: "			<< ( prop.memPitch >> 20 ) << " Mo"				<< std::endl;
	std::cout << " - can map host memory: "		<< ( prop.canMapHostMemory ? "yes" : "no" )		<< std::endl;
	std::cout << " - memory bus width: "		<< prop.memoryBusWidth	<< "-bit"				<< std::endl;
	std::cout << " - memory clock rate: "		<< prop.memoryClockRate	* 1e-3f << " MHz"		<< std::endl;
	std::cout << " - unified addressing: "		<< ( prop.unifiedAddressing ? "yes" : "no" )	<< std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << " - registers per blocks: "			<< prop.regsPerBlock				<< std::endl;
	std::cout << " - warpSize: "						<< prop.warpSize					<< std::endl;
	std::cout << " - max threads dim: "					<< prop.maxThreadsDim[0] << ", "
														<< prop.maxThreadsDim[1] << ", "
														<< prop.maxThreadsDim[2]			<< std::endl;
	std::cout << " - max threads per block: "			<< prop.maxThreadsPerBlock			<< std::endl;
	std::cout << " - max threads per multiprocessor: "	<< prop.maxThreadsPerMultiProcessor	<< std::endl;
	std::cout << " - max grid size: "					<< prop.maxGridSize[0] << ", "
														<< prop.maxGridSize[1] << ", "
														<< prop.maxGridSize[2]				<< std::endl;
	std::cout << " - multiProcessor count: "			<< prop.multiProcessorCount			<< std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << " - texture alignment: "	<< prop.textureAlignment		<<std::endl;
	std::cout << " - max Texture 1D: "		<< prop.maxTexture1D			<<std::endl;
	std::cout << " - max texture 2d: "		<< prop.maxTexture2D[0] << ", " 
											<< prop.maxTexture2D[1]			<< std::endl;
	std::cout << " - max texture 3D: "		<< prop.maxTexture3D[0] << ", "
											<< prop.maxTexture3D[1] << ", "
											<< prop.maxTexture3D[2]			<< std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << " - number of asynchronous engines: "	<< prop.asyncEngineCount				<< std::endl;
	std::cout << " - clock frequency: "					<< prop.clockRate * 1e-3f << " MHz"		<< std::endl;
	std::cout << " - ECCEnabled: "						<< ( prop.ECCEnabled ? "yes" : "no"	)	<< std::endl;
	std::cout << " - level 2 cache size: "				<< ( prop.l2CacheSize >> 10 ) << " Ko"	<< std::endl << std::endl;
}
