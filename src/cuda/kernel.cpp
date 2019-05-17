#include "cuimage/cuda/kernel.h"

#include "assert.h"

#define MAX_GRID_SIZE 65535

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

dim3 grid2D(const size_t width, const size_t height, dim3& block)
{
    assert(block.x * block.y * block.z <= 1024);

    auto grid = dim3((width  + block.x - 1) / block.x, 
                     (height + block.y - 1) / block.y);

    // Prevent grid overflow by adapting block sizes
    if (grid.x >= MAX_GRID_SIZE)
    {
        block.x *= 2;
        block.y /= 2;
        return grid2D(width, height, block);
    }
    
    if (grid.y >= MAX_GRID_SIZE)
    {
        block.y *= 2;
        block.x /= 2;
        return grid2D(width, height, block);
    }

    return grid;
}

dim3 grid3D(const size_t width, const size_t height, const size_t channels, dim3& block)
{
    assert(block.x * block.y * block.z <= 1024);

    auto grid = dim3((width  + block.x - 1) / block.x, 
                     (height + block.y - 1) / block.y, 
                     (channels  + block.z - 1) / block.z);

    // Prevent grid overflow by adapting block sizes
    if (grid.x >= MAX_GRID_SIZE)
    {
        block.x *= 2;
        block.y /= 2;
        return grid2D(width, height, block);
    }
    
    if (grid.y >= MAX_GRID_SIZE)
    {
        block.y *= 2;
        block.x /= 2;
        return grid2D(width, height, block);
    }

    if (grid.z >= MAX_GRID_SIZE)
    {
        block.z *= 2;
        return grid2D(width, height, block);
    }
    
    return grid;
}
