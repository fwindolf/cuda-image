#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/devptr.h"
#include "cuimage/cuda/interpolation.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/utils.h"
#include "cuimage/operations/resize_cu.h"

#define USE_HALF_PIXEL_OFFSET true

namespace cuimage
{

__device__ float d_pixel(const int p, const float f)
{
#if USE_HALF_PIXEL_OFFSET
    return p / f;
#else
    return p / f;
#endif
}

__device__ int clamp(const int value, const int min, const int max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}

__device__ int2 d_surrounding(const float p, const int min, const int max)
{
#if USE_HALF_PIXEL_OFFSET
    const int px0 = clamp(floor(p - 0.5), min, max);
    const int px1 = clamp(ceil(p - 0.5), min, max);
#else
    const int px0 = clamp(floor(p), min, max);
    const int px1 = clamp(ceil(p), min, max);
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
using interpolationFn = nvstd::function<T(const T&, const T&, const uchar,
    const uchar, const int, const int, const float)>;

template <typename T>
__device__ T d_inter(const interpolationFn<T>& f, const T& x0, const T& x1,
    const int& px0, const int& px1, const float& p)
{
    if (px0 == px1)
    {
        assert(x0 == x1);
        return x0;
    }

#if USE_HALF_PIXEL_OFFSET
    return f(x0, x1, 1, 1, px0, px1, p - 0.5);
#else
    return f(x0, x1, 1, 1, px0, px1, p);
#endif
}

template <typename T>
__device__ T d_inter(const interpolationFn<T>& f, const T& x0, const T& x1,
    const uchar& m0, const uchar& m1, const int& px0, const int& px1,
    const float& p)
{
    if (px0 == px1)
    {
        assert(x0 == x1);
        return x0;
    }

#if USE_HALF_PIXEL_OFFSET
    return f(x0, x1, m0, m1, px0, px1, p - 0.5);
#else
    return f(x0, x1, m0, m1, px0, px1, p);
#endif
}

template <typename T>
__device__ T d_interpolate(const DevPtr<T>& image, const int px, const int py,
    const float fx, const float fy, const interpolationFn<T>& f)
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

    // In x direction
    T ax0 = d_inter(f, a00, a10, ppx.x, ppx.y, px_);
    T ax1 = d_inter(f, a01, a11, ppx.x, ppx.y, px_);

    // In y direction
    return d_inter(f, ax0, ax1, ppy.x, ppy.y, py_);
}

template <typename T>
__device__ T d_interpolate(const DevPtr<T>& image, const DevPtr<uchar>& mask,
    const int px, const int py, const float fx, const float fy,
    const interpolationFn<T>& f)
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

    // In x direction
    T ax0 = d_inter(f, a00, a10, m00, m10, ppx.x, ppx.y, px_);
    T ax1 = d_inter(f, a01, a11, m01, m11, ppx.x, ppx.y, px_);

    // In y direction
    return d_inter(f, ax0, ax1, m00 & m10, m01 & m11, ppy.x, ppy.y, py_);
}

template <typename T>
__global__ void g_ResizeNearest(DevPtr<T> output, const DevPtr<T> input)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_nearest<T>;
    output(pos.x, pos.y) = d_interpolate(input, pos.x, pos.y, fx, fy, f);
}

template <typename T>
__global__ void g_ResizeNearest(
    DevPtr<T> output, const DevPtr<T> input, const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_nearest_masked<T>;
    output(pos.x, pos.y) = d_interpolate(input, mask, pos.x, pos.y, fx, fy, f);
}

template <typename T>
__global__ void g_ResizeLinear(DevPtr<T> output, const DevPtr<T> input)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_linear<T>;
    output(pos.x, pos.y) = d_interpolate(input, pos.x, pos.y, fx, fy, f);
}

template <typename T>
__global__ void g_ResizeLinear(
    DevPtr<T> output, const DevPtr<T> input, const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_linear_masked<T>;
    output(pos.x, pos.y) = d_interpolate(input, mask, pos.x, pos.y, fx, fy, f);
}

template <typename T>
__global__ void g_ResizeLinearValid(DevPtr<T> output, const DevPtr<T> input)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_linear_valid<T>;
    output(pos.x, pos.y) = d_interpolate(input, pos.x, pos.y, fx, fy, f);
}

template <typename T>
__global__ void g_ResizeLinearValid(
    DevPtr<T> output, const DevPtr<T> input, const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_linear_valid_masked<T>;
    output(pos.x, pos.y) = d_interpolate(input, mask, pos.x, pos.y, fx, fy, f);
}

template <typename T>
__global__ void g_ResizeLinearNonZero(DevPtr<T> output, const DevPtr<T> input)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_linear_nonzero<T>;
    output(pos.x, pos.y) = d_interpolate(input, pos.x, pos.y, fx, fy, f);
}

template <typename T>
__global__ void g_ResizeLinearNonZero(
    DevPtr<T> output, const DevPtr<T> input, const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const float fx = (float)output.width / input.width;
    const float fy = (float)output.height / input.height;

    interpolationFn<T> f = d_interpolate_linear_nonzero_masked<T>;
    output(pos.x, pos.y) = d_interpolate(input, mask, pos.x, pos.y, fx, fy, f);
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
void cu_ResizeNearest(DevPtr<T> output, const DevPtr<T>& input)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeNearest<<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ResizeNearest(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeNearest<<<grid, block>>>(output, input, mask);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinear<<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ResizeLinear(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask)
{
    assert(input.width == mask.width);
    assert(input.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinear<<<grid, block>>>(output, input, mask);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ResizeLinearValid(DevPtr<T> output, const DevPtr<T>& input)
{
    assert(input.width == mask.width);
    assert(input.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinearValid<<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ResizeLinearValid(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask)
{
    assert(input.width == mask.width);
    assert(input.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinearValid<<<grid, block>>>(output, input, mask);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ResizeLinearNonZero(DevPtr<T> output, const DevPtr<T>& input)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinearNonZero<<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ResizeLinearNonZero(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask)
{
    assert(input.width == mask.width);
    assert(input.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ResizeLinearNonZero<<<grid, block>>>(output, input, mask);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T>
void cu_ApplyMask(DevPtr<T> image, const DevPtr<uchar>& mask)
{
    assert(image.width == mask.width);
    assert(image.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(image.width, image.height, block);

    g_applyMask<<<grid, block>>>(image, mask);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

#define DECLARE_RESIZE_FUNCTION(type, function)                               \
    template void function(DevPtr<type>, const DevPtr<type>&);

FOR_EACH_TYPE(DECLARE_RESIZE_FUNCTION, cu_ResizeNearest)
FOR_EACH_TYPE(DECLARE_RESIZE_FUNCTION, cu_ResizeLinear)
FOR_EACH_TYPE(DECLARE_RESIZE_FUNCTION, cu_ResizeLinearValid)
FOR_EACH_TYPE(DECLARE_RESIZE_FUNCTION, cu_ResizeLinearNonZero)

#define DECLARE_MASKED_RESIZE_FUNCTION(type, function)                        \
    template void function(                                                   \
        DevPtr<type>, const DevPtr<type>&, const DevPtr<uchar>&);

FOR_EACH_TYPE(DECLARE_MASKED_RESIZE_FUNCTION, cu_ResizeNearest)
FOR_EACH_TYPE(DECLARE_MASKED_RESIZE_FUNCTION, cu_ResizeLinear)
FOR_EACH_TYPE(DECLARE_MASKED_RESIZE_FUNCTION, cu_ResizeLinearValid)
FOR_EACH_TYPE(DECLARE_MASKED_RESIZE_FUNCTION, cu_ResizeLinearNonZero)

#define DECLARE_MASK_FUNCTION(type, function)                                 \
    template void function(DevPtr<type>, const DevPtr<uchar>&);

FOR_EACH_TYPE(DECLARE_MASK_FUNCTION, cu_ApplyMask)

} // image