#include "image/operations/conversion_cu.h"

#include "image/cuda/utils.h"
#include "image/cuda/arithmetic.h"
#include "image/cuda/kernel.h"
#include "image/cuda/devptr.h"

namespace image
{

template <typename T>
__global__ void g_add(T* lhs, const T* rhs, const float factor, const size_t width, const size_t height)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= width || pos.y >= height)
        return;
    
    const int idx = pos.x + pos.y * width;
    lhs[idx] += factor * rhs[idx];
}

template <typename T>
void add(T* lhs, const T* rhs, const float factor, const size_t width, const size_t height)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(width, height, block);

    g_add <<< grid, block >>> (lhs, rhs, factor, width, height);

    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());    
}

template <typename T>
__global__ void g_mul(T* lhs, const T* rhs, const float factor, const size_t width, const size_t height)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= width || pos.y >= height)
        return;
    
    const int idx = pos.x + pos.y * width;
    lhs[idx] *= factor * rhs[idx];
}

template <typename T>
void multiply(T* lhs, const T* rhs, const float factor, const size_t width, const size_t height)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(width, height, block);

    g_mul <<< grid, block >>> (lhs, rhs, factor, width, height);

    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());    
}

template <typename T>
T* colorToGrey(const void* h_data, const size_t width, const size_t height, const size_t chans)
{
    size_t channelsT = channels<T>();
    assert(channelsT == 1); // grey

    // Add every channel with multiplier, then multiply with intesity (if exists)
    T* d_data;
    T* tmp;
    size_t sizeBytes = width * height * sizeof(T);
    cudaSafeCall(cudaMalloc(&d_data, sizeBytes));
    cudaSafeCall(cudaMalloc(&tmp, sizeBytes));
    cudaSafeCall(cudaMemset(d_data, 0, sizeBytes));
    
    size_t dstPitch = sizeof(T);
    size_t srcPitch = sizeof(T) * chans;

    float rgb[3] = { 0.21f, 0.72f, 0.07f };
    for (int c = 0; c < chans; c++)
    {
        // Offset the host data so we take a new part of the interleaved data every iteration
        void* h_data_off = (void*) &((T*)h_data)[c * sizeof(T)];
        cudaSafeCall(cudaMemcpy2D(tmp, dstPitch, h_data_off, srcPitch, srcPitch, width * height, cudaMemcpyHostToDevice));
        if (chans < 3)
            add(d_data, tmp, rgb[c], width, height); // RGB
        else 
            multiply(d_data, tmp, 1.f, width, height); // A
    }

    return d_data;
}

/**
 * Explicit instantiations
 */

template __global__ void g_add(float* lhs, const float* rhs, const float factor, const size_t width, const size_t height);

template __global__ void g_add(uchar* lhs, const uchar* rhs, const float factor, const size_t width, const size_t height);

template __global__ void g_add(int* lhs, const int* rhs, const float factor, const size_t width, const size_t height);


template void add(float* lhs, const float* rhs, const float factor, const size_t width, const size_t height);

template void add(uchar* lhs, const uchar* rhs, const float factor, const size_t width, const size_t height);

template void add(int* lhs, const int* rhs, const float factor, const size_t width, const size_t height);


template __global__ void g_mul(float* lhs, const float* rhs, const float factor, const size_t width, const size_t height);

template __global__ void g_mul(uchar* lhs, const uchar* rhs, const float factor, const size_t width, const size_t height);

template __global__ void g_mul(int* lhs, const int* rhs, const float factor, const size_t width, const size_t height);


template void multiply(float* lhs, const float* rhs, const float factor, const size_t width, const size_t height);

template void multiply(uchar* lhs, const uchar* rhs, const float factor, const size_t width, const size_t height);

template void multiply(int* lhs, const int* rhs, const float factor, const size_t width, const size_t height);


template float* colorToGrey(const void* h_data, const size_t width, const size_t height, const size_t chans);

template uchar* colorToGrey(const void* h_data, const size_t width, const size_t height, const size_t chans);

template int* colorToGrey(const void* h_data, const size_t width, const size_t height, const size_t chans);

/**
 * Implementation
 */


template <typename T>
T* greyToColor(const void* h_data, const size_t width, const size_t height)
{
    size_t channelsT = channels<T>();
    assert(channelsT >= 3); // rgb or rgba

    // Cannot recover colors, so just allocate and copy
    T* d_data;
    size_t sizeBytes = width * height * sizeof(T); // Assume caller knows what he is doing
    cudaSafeCall(cudaMalloc(&d_data, sizeBytes));

    size_t dstPitch = sizeof(T);
    size_t srcPitch = sizeof(T) / channels<T>();

    /*
    void* tmp;

    size_t sizeBytes = width * height * sizeof(T); // Assume caller knows what he is doing
    cudaSafeCall(cudaMalloc(&d_data, sizeBytes));

    // Copy host to device
    cudaSafeCall(cudaMalloc(&tmp, sizeBytes / channelsT));
    cudaSafeCall(cudaMemcpy(tmp, h_data, sizeBytes / channelsT, cudaMemcpyHostToDevice));
    
    // Broadcast to color
    dim3 block = block2D(32);
    dim3 grid = grid2D(width, height, block);
    cudaSafeCall(cudaMemcpy(d_data, h_data, width * height * ))
    g_broadcast <<< grid, block >>> (tmp, d_data, width, height);
    */

    for (int c = 0; c < channels<T>(); c++)
        cudaSafeCall(cudaMemcpy2D(&d_data[c], dstPitch, h_data, srcPitch, srcPitch, width * height, cudaMemcpyHostToDevice));

    return d_data;
}

template <typename T>
T* cu_greyToRgb(const void* h_data, const size_t width, const size_t height)
{
    return greyToColor<T>(h_data, width, height);
}

template <typename T>
T* cu_greyToRgba(const void* h_data, const size_t width, const size_t height)
{
    return greyToColor<T>(h_data, width, height);
}

template <typename T>
T* cu_rgbToGrey(const void* h_data, const size_t width, const size_t height)
{
    return colorToGrey<T>(h_data, width, height, 3);
}

template <typename T>
T* cu_rgbaToGrey(const void* h_data, const size_t width, const size_t height)
{
    return colorToGrey<T>(h_data, width, height, 4);
}


/**
 * Explicit instantiations
 */


template uchar3* cu_greyToRgb(const void* h_data, const size_t width, const size_t height);

template float3* cu_greyToRgb(const void* h_data, const size_t width, const size_t height);

template int3* cu_greyToRgb(const void* h_data, const size_t width, const size_t height);


template uchar4* cu_greyToRgba(const void* h_data, const size_t width, const size_t height);

template float4* cu_greyToRgba(const void* h_data, const size_t width, const size_t height);

template int4* cu_greyToRgba(const void* h_data, const size_t width, const size_t height);


template uchar* cu_rgbToGrey(const void* h_data, const size_t width, const size_t height);

template float* cu_rgbToGrey(const void* h_data, const size_t width, const size_t height);

template int* cu_rgbToGrey(const void* h_data, const size_t width, const size_t height);


template uchar* cu_rgbaToGrey(const void* h_data, const size_t width, const size_t height);

template float* cu_rgbaToGrey(const void* h_data, const size_t width, const size_t height);

template int* cu_rgbaToGrey(const void* h_data, const size_t width, const size_t height);

}