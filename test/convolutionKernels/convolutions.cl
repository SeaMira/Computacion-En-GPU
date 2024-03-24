void kernel kernelConvolution(__global const unsigned char *pixelsIn, __global unsigned char *pixelsOut, const int width, const int height, const int nbChannels,
								  __global const float *kernelConv, const int kernelDim) {
 
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int idPixel = nbChannels * (x + y);

    for (int i = 0; i < height*width*nbChannels; i += width*nbChannels) {

        float outRed = 0;
        float outGreen = 0;
        float outBlue = 0;
        const int halfKernelDim = kernelDim / 2;
        
        for (int j = 0; j < kernelDim; j++) {
            for (int k = 0; k < kernelDim; k++) {
                int prov_x = x + k - halfKernelDim;
                int prov_y = y + j - halfKernelDim;
                int idPixelFinal = (prov_x + prov_y)*nbChannels; 
                outRed += kernelConv[k+j*kernelDim]*pixelsIn[idPixelFinal + i];
                outGreen += kernelConv[k+j*kernelDim]*pixelsIn[idPixelFinal + 1 + i];
                outBlue += kernelConv[k+j*kernelDim]*pixelsIn[idPixelFinal + 2 + i];
            }
        }

        pixelsOut[idPixel +i] = (unsigned char)outRed;
        pixelsOut[idPixel + 1 +i] = (unsigned char)outGreen;
        pixelsOut[idPixel + 2 +i] = (unsigned char)outBlue;

        if (nbChannels == 4)
                    pixelsOut[idPixel + 3 +i] = pixelsIn[idPixel + 3 +i];
    } 

}