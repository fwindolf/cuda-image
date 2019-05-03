#include "image/cuda/kernel.h"

/**
 * Block / Grid Helpers
 */
dim3 block2D(const int blockSize)
{
    return dim3(blockSize, blockSize);
}

dim3 block2D(const int blockSizeX, const int blockSizeY)
{
    return dim3(blockSizeX, blockSizeY);
}

dim3 block3D(const int blockSize, const size_t channels)
{
    return dim3(blockSize, blockSize, channels);
}

dim3 grid2D(const size_t width, const size_t height, const dim3 block)
{
    return dim3((width  + block.x - 1) / block.x, 
                (height + block.y - 1) / block.y);
}

dim3 grid3D(const size_t width, const size_t height, const size_t channels, const dim3 block)
{
    return dim3((width  + block.x - 1) / block.x, 
                (height + block.y - 1) / block.y, 
                (channels  + block.z - 1) / block.z);
}
