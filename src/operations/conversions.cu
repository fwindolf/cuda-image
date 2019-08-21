#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/conversion.h"
#include "cuimage/cuda/devptr.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/utils.h"
#include "cuimage/operations/conversions_cu.h"

namespace cuimage
{

template <typename T, typename TO,
    typename std::enable_if<has_4_channels<TO>::value, TO>::type* = nullptr>
__global__ void g_GrayToColor(DevPtr<TO> output, const DevPtr<T> input)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    output(pos.x, pos.y) = gray2rgba<TO>(as<float>(input(pos.x, pos.y)));
}

template <typename T, typename TO,
    typename std::enable_if<has_3_channels<TO>::value, TO>::type* = nullptr>
__global__ void g_GrayToColor(DevPtr<TO> output, const DevPtr<T> input)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    output(pos.x, pos.y) = gray2rgb<TO>(as<float>(input(pos.x, pos.y)));
}

template <typename T, typename TO,
    typename std::enable_if<has_4_channels<T>::value, T>::type* = nullptr>
__global__ void g_ColorToGray(DevPtr<TO> output, const DevPtr<T> input)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    float val = rgba2gray(input(pos.x, pos.y));
    output(pos.x, pos.y) = as<TO>(val);
}

template <typename T, typename TO,
    typename std::enable_if<has_3_channels<T>::value, T>::type* = nullptr>
__global__ void g_ColorToGray(DevPtr<TO> output, const DevPtr<T> input)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    float val = rgb2gray(input(pos.x, pos.y));
    output(pos.x, pos.y) = as<TO>(val);
}

template <typename T, typename TO,
    typename std::enable_if<has_3_channels<T>::value
            && has_4_channels<TO>::value,
        T>::type* = nullptr>
__global__ void g_ColorToColor(DevPtr<TO> output, const DevPtr<T> input)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    output(pos.x, pos.y) = rgb2rgba<TO>(input(pos.x, pos.y));
}

template <typename T, typename TO,
    typename std::enable_if<has_4_channels<T>::value
            && has_3_channels<TO>::value,
        T>::type* = nullptr>
__global__ void g_ColorToColor(DevPtr<TO> output, const DevPtr<T> input)
{
    dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    output(pos.x, pos.y) = rgba2rgb<TO>(input(pos.x, pos.y));
}

template <typename T, typename TO>
__global__ void g_Convert(DevPtr<TO> output, const DevPtr<T> input)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    output(pos.x, pos.y) = as<TO>(input(pos.x, pos.y));
}

template <typename T, typename TO>
__global__ void g_GetComponent(
    DevPtr<TO> output, const DevPtr<T> input, const ushort component)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= output.width || pos.y >= output.height)
        return;

    const T val = input(pos.x, pos.y);
    output(pos.x, pos.y) = d_get<TO>(val, component);
}

template <typename T, typename TO>
void cu_GrayToColor(DevPtr<TO> output, const DevPtr<T>& input)
{
    assert(output.width == input.width);
    assert(output.height == input.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_GrayToColor<T, TO><<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifndef DNDEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T, typename TO>
void cu_ColorToGray(DevPtr<TO> output, const DevPtr<T>& input)
{
    assert(output.width == input.width);
    assert(output.height == input.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ColorToGray<T, TO><<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifndef DNDEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T, typename TO>
void cu_ColorToColor(DevPtr<TO> output, const DevPtr<T>& input)
{
    assert(output.width == input.width);
    assert(output.height == input.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_ColorToColor<T, TO><<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifndef DNDEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T, typename TO>
void cu_Convert(DevPtr<TO> output, const DevPtr<T>& input)
{
    assert(output.width == input.width);
    assert(output.height == input.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_Convert<T, TO><<<grid, block>>>(output, input);

    cudaCheckLastCall();
#ifndef DNDEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T, typename TO>
void cu_GetComponent(
    DevPtr<TO> output, const DevPtr<T>& input, const ushort component)
{
    assert(output.width == input.width);
    assert(output.height == input.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_GetComponent<T, TO><<<grid, block>>>(output, input, component);

    cudaCheckLastCall();
#ifndef DNDEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

/**
 * Explicit instantiations
 */
#define INST_CONVERSION_FUNCTION(type, typeO, function)                       \
    template void function(DevPtr<typeO>, const DevPtr<type>&);

// All to all conversions with same channel
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float, cu_Convert)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float1, cu_Convert)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar, cu_Convert)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar1, cu_Convert)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int, cu_Convert)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int1, cu_Convert)

FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float, cu_Convert)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float1, cu_Convert)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar, cu_Convert)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar1, cu_Convert)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int, cu_Convert)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int1, cu_Convert)

FOR_EACH_2CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float2, cu_Convert)
FOR_EACH_2CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar2, cu_Convert)
FOR_EACH_2CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int2, cu_Convert)

FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float3, cu_Convert)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar3, cu_Convert)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int3, cu_Convert)

FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float4, cu_Convert)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar4, cu_Convert)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int4, cu_Convert)

// Conversions from gray to color
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float3, cu_GrayToColor)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar3, cu_GrayToColor)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int3, cu_GrayToColor)

FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float3, cu_GrayToColor)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar3, cu_GrayToColor)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int3, cu_GrayToColor)

FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float4, cu_GrayToColor)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar4, cu_GrayToColor)
FOR_EACH_0CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int4, cu_GrayToColor)

FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float4, cu_GrayToColor)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar4, cu_GrayToColor)
FOR_EACH_1CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int4, cu_GrayToColor)

// Conversions from color to gray
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float, cu_ColorToGray)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float1, cu_ColorToGray)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar, cu_ColorToGray)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar1, cu_ColorToGray)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int, cu_ColorToGray)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int1, cu_ColorToGray)

FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float, cu_ColorToGray)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float1, cu_ColorToGray)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar, cu_ColorToGray)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar1, cu_ColorToGray)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int, cu_ColorToGray)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int1, cu_ColorToGray)

// Conversions from color to color
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float4, cu_ColorToColor)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar4, cu_ColorToColor)
FOR_EACH_3CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int4, cu_ColorToColor)

FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, float3, cu_ColorToColor)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, uchar3, cu_ColorToColor)
FOR_EACH_4CHANNEL_TYPE(INST_CONVERSION_FUNCTION, int3, cu_ColorToColor)

// Get single component
#undef INST_CONVERSION_FUNCTION
#define INST_CONVERSION_FUNCTION(type, typeO, function)                       \
    template void function(DevPtr<typeO>, const DevPtr<type>&, const ushort);

FOR_EACH_FLOAT_TYPE(INST_CONVERSION_FUNCTION, float, cu_GetComponent);
FOR_EACH_UCHAR_TYPE(INST_CONVERSION_FUNCTION, uchar, cu_GetComponent);
FOR_EACH_INT_TYPE(INST_CONVERSION_FUNCTION, int, cu_GetComponent);

} // image