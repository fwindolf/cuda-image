#include "cuimage/operations/resize_cu.h"

#include "cuimage/cuda/utils.h"
#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/interpolation.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/devptr.h"

#define USE_HALF_PIXEL_OFFSET true

namespace cuimage
{

__device__ float d_pixel(const int p, const float f)
{
#if USE_HALF_PIXEL_OFFSET
    return ((p - 0.5) / f) + 0.5;
#else
    return p / f;
#endif
}

__device__ int2 d_surrounding(const int p, const int min, const int max)
{
#if USE_HALF_PIXEL_OFFSET
    const int px0 = clamp<int>(floor(p - 0.5), min, max);
    const int px1 = clamp<int>( ceil(p - 0.5), min, max);
#else
    const int px0 = clamp<int>(floor(p), min, max);
    const int px1 = clamp<int>( ceil(p), min, max);
#endif

    return make_int2(px0, px1);
}

__device__ int2 d_closest(const float p)
{
#if USE_HALF_PIXEL_OFFSET
    const int pp = int(p) + 1;
    const int pm = int(p);
#else
    const int pp = (int)ceil(p);
    const int pm = (int)floor(p);
#endif

    return make_int2(pm, pp);
}

template <typename T>
__device__ T d_interpolate(const DevPtr<T>& image, const int px, const int py, const float fx, const float fy)
{
    const int w_old = image.width;
    const int h_old = image.height;    

    // New pixels in old coordinate system
    const float px_ = d_pixel(px, fx);
    const float py_ = d_pixel(py, fy);

    // Surrounding pixels
    const int2 ppx = d_surrounding(px_, 0, w_old - 1);
    const int2 ppy = d_surrounding(py_, 0, h_old - 1);

    const T a00 = image(ppx.x, ppy.x);
    const T a10 = image(ppx.y, ppy.x);
    const T a01 = image(ppx.x, ppy.y);
    const T a11 = image(ppx.y, ppy.y);

    if (isnan(a00) && isnan(a10) && isnan(a01) && isnan(a11))
        return a00;
    
    // In x direction
    const int2 pxn = d_closest(px_);

    const T ax0 = d_interpolate_linear(a00, a10, pxn.x, pxn.y, px_);
    const T ax1 = d_interpolate_linear(a01, a11, pxn.x, pxn.y, px_);

    // in y direction
    const int2 pyn = d_closest(py_);

    return d_interpolate_linear(ax0, ax1, pyn.x, pyn.y, py_);
}

template <typename T>
__device__ T d_interpolate_masked(const DevPtr<T>& image, const DevPtr<uchar>& mask, const int px, const int py, const float fx, const float fy)
{
    const int w_old = image.width;
    const int h_old = image.height;    

    // New pixels in old coordinate system
    const float px_ = d_pixel(px, fx);
    const float py_ = d_pixel(py, fy);

    // Surrounding pixels
    const int2 ppx = d_surrounding(px_, 0, w_old - 1);
    const int2 ppy = d_surrounding(py_, 0, h_old - 1);

    const T a00 = image(ppx.x, ppy.x);
    const T a10 = image(ppx.y, ppy.x);
    const T a01 = image(ppx.x, ppy.y);
    const T a11 = image(ppx.y, ppy.y);

    const uchar m00 = mask(ppx.x, ppy.x);
    const uchar m10 = mask(ppx.y, ppy.x);
    const uchar m01 = mask(ppx.x, ppy.y);
    const uchar m11 = mask(ppx.y, ppy.y);

    if (isnan(a00) && isnan(a10) && isnan(a01) && isnan(a11))
        return a00;
    
    // In x direction
    const int2 pxn = d_closest(px_);

    const T ax0 = d_interpolate_linear_masked(a00, a10, m00, m10, pxn.x, pxn.y, px_);
    const T ax1 = d_interpolate_linear_masked(a01, a11, m01, m11, pxn.x, pxn.y, px_);

    // in y direction
    const int2 pyn = d_closest(py_);

    return d_interpolate_linear(ax0, ax1, pyn.x, pyn.y, py_);
} 


template <typename T>
__global__ void g_ResizeLinear(DevPtr<T> output, const DevPtr<T> input)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    output(pos.x, pos.y) = d_interpolate(input, pos.x, pos.y, fx, fy);
}

template <typename T>
__global__ void g_ResizeLinear(DevPtr<T> output, const DevPtr<T> input, const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= output.width || pos.y >= output.height)
        return;
    
    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    output(pos.x, pos.y) = d_interpolate_masked(input, mask, pos.x, pos.y, fx, fy);
}

template <typename T>
__global__ void g_applyMask(DevPtr<T> image, const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = image.width;
    const int height = image.height;

    if (pos.x >= width || pos.y >= height)
        return;
    
    if (!mask(pos.x, pos.y))
        image(pos.x, pos.y) = make<T>(0);
}


template <typename T>
void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinear <<< grid, block >>> (output, input);

    cudaCheckLastCall();
    cudaDeviceSynchronize();
}

template <typename T>
void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask)
{
    assert(output.width == mask.width);
    assert(output.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinear <<< grid, block >>> (output, input, mask);

    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());
}

template <typename T>
void cu_ApplyMask(DevPtr<T> image, const DevPtr<uchar>& mask)
{
    assert(image.width == mask.width);
    assert(image.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(image.width, image.height, block);

    g_applyMask <<< grid, block >>> (image, mask);

    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());
}

#define DECLARE_RESIZE_FUNCTION(type, function) \
    template void function(DevPtr<type>, const DevPtr<type>&);

#define DECLARE_MASKED_RESIZE_FUNCTION(type, function) \
    template void function(DevPtr<type>, const DevPtr<type>&, const DevPtr<uchar>&);

#define DECLARE_MASK_FUNCTION(type, function) \
    template void function(DevPtr<type>, const DevPtr<uchar>&);
    
FOR_EACH_TYPE(DECLARE_RESIZE_FUNCTION, cu_ResizeLinear)
FOR_EACH_TYPE(DECLARE_MASKED_RESIZE_FUNCTION, cu_ResizeLinear)
FOR_EACH_TYPE(DECLARE_MASK_FUNCTION, cu_ApplyMask)




} // image