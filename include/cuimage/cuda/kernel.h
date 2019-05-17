#pragma once 

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

/**
 * Block / Grid Helpers
 */
dim3 block2D(const int blockSize);

dim3 block2D(const int blockSizeX, const int blockSizeY);

dim3 block3D(const int blockSize, const size_t channels);

dim3 grid2D(const size_t width, const size_t height, dim3& block);

dim3 grid3D(const size_t width, const size_t height, const size_t channels, dim3& block);

/**
 * Device indexing helpers
 */

inline __device__ dim3 getPos(const dim3 blockIdx, const dim3 blockDim, const dim3 threadIdx)
{
    dim3 pos;
    pos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pos.y = blockIdx.y * blockDim.y + threadIdx.y;
    pos.z = blockIdx.z * blockDim.z + threadIdx.z;
    return pos;
}

inline __device__ int getIndex(const dim3 position, const dim3 dimensions)
{
    return position.x +
        position.y * dimensions.x +
        position.z * (dimensions.x * dimensions.y);
}

inline __device__ int getIndex(const dim3 position, const size_t width, const size_t height, const size_t channels = 0)
{
    return getIndex(position, dim3(width, height, channels));
}
 
