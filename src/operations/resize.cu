#include "cuimage/operations/resize_cu.h"

#include "cuimage/cuda/utils.h"
#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/interpolation.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/devptr.h"

#define USE_HALF_PIXEL_OFFSET true

namespace cuimage
{

template <typename T>
__device__ T d_interpolate(const DevPtr<T>& image, const int px, const int py, const float fx, const float fy)
{
    const int w_old = image.width;
    const int h_old = image.height;    

    // New pixels in old coordinate system
#if USE_HALF_PIXEL_OFFSET
    const float px_ = ((px - 0.5) / fx) + 0.5;
    const float py_ = ((py - 0.5) / fy) + 0.5;
#else
    const float px_ = px / fx;
    const float py_ = py / fy;
#endif

    // Surrounding pixels
#if USE_HALF_PIXEL_OFFSET
    const int px0 = clamp<int>(floor(px_ - 0.5), 0, w_old - 1);
    const int px1 = clamp<int>(ceil(px_ - 0.5),  0, w_old - 1);
    const int py0 = clamp<int>(floor(py_ - 0.5), 0, h_old - 1);
    const int py1 = clamp<int>(ceil(py_ - 0.5), 0, h_old - 1);
#else
    const int px0 = clamp<int>(floor(px_), 0, w_old - 1);
    const int px1 = clamp<int>(ceil(px_),  0, w_old - 1);
    const int py0 = clamp<int>(floor(py_), 0, h_old - 1);
    const int py1 = clamp<int>(ceil(py_), 0, h_old - 1);
#endif

    const T a00 = image(px0, py0);
    const T a10 = image(px1, py0);
    const T a01 = image(px0, py1);
    const T a11 = image(px1, py1);

    if (isnan(a00) && isnan(a10) && isnan(a01) && isnan(a11))
        return a00;
    
    // In x direction
#if USE_HALF_PIXEL_OFFSET
    const int pxp = int(px_) + 1;
    const int pxm = int(px_);
#else
    const int pxp = (int)ceil(px_);
    const int pxm = (int)floor(px_);
#endif

    T ax0 = interpolate_linear(a00, a10, pxm, pxp, px_);
    T ax1 = interpolate_linear(a01, a11, pxm, pxp, px_);

    // in y direction
#if USE_HALF_PIXEL_OFFSET
    const int pyp = int(py_) + 1;
    const int pym = int(py_);
#else
    const int pyp = (int)ceil(py_);
    const int pym = (int)floor(py_);
#endif

    return interpolate_linear(ax0, ax1, pym, pyp, py_);
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

    if (!mask(pos.x, pos.y))
    {
        output(pos.x, pos.y) = make<T>(0);
        return;
    }
    
    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    output(pos.x, pos.y) = d_interpolate(input, pos.x, pos.y, fx, fy);
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