#pragma once

#include "type.h"

#include <cassert>
#include <vector_functions.hpp>

namespace cuimage
{

/**
 * Convert from T to TO
 * Only conversions with the same amounts of channels are defined
 */
template <typename TO, typename T> __host__ __device__ TO as(const T v);

/**
 * From float
 */

template <> inline __host__ __device__ float as(const float v) { return v; }

template <> inline __host__ __device__ uchar as(const float v)
{
    return static_cast<uchar>(v);
}

template <> inline __host__ __device__ int as(const float v)
{
    return static_cast<int>(v);
}

template <> inline __host__ __device__ float1 as(const float v)
{
    return make_float1(v);
}

template <> inline __host__ __device__ uchar1 as(const float v)
{
    return make_uchar1(static_cast<uchar>(v));
}

template <> inline __host__ __device__ int1 as(const float v)
{
    return make_int1(static_cast<int>(v));
}

/**
 * From float 1
 */

template <> inline __host__ __device__ float as(const float1 v) { return v.x; }

template <> inline __host__ __device__ uchar as(const float1 v)
{
    return static_cast<uchar>(v.x);
}

template <> inline __host__ __device__ int as(const float1 v)
{
    return static_cast<int>(v.x);
}

template <> inline __host__ __device__ float1 as(const float1 v)
{
    return make_float1(v.x);
}

template <> inline __host__ __device__ uchar1 as(const float1 v)
{
    return make_uchar1(static_cast<uchar>(v.x));
}

template <> inline __host__ __device__ int1 as(const float1 v)
{
    return make_int1(static_cast<int>(v.x));
}

/**
 * From float2
 */

template <> inline __host__ __device__ float2 as(const float2 v) { return v; }

template <> inline __host__ __device__ uchar2 as(const float2 v)
{
    return make_uchar2(static_cast<uchar>(v.x), static_cast<uchar>(v.y));
}

template <> inline __host__ __device__ int2 as(const float2 v)
{
    return make_int2(static_cast<int>(v.x), static_cast<int>(v.y));
}

/**
 * From float 3
 */

template <> inline __host__ __device__ float3 as(const float3 v) { return v; }

template <> inline __host__ __device__ uchar3 as(const float3 v)
{
    return make_uchar3(static_cast<uchar>(v.x), static_cast<uchar>(v.y),
        static_cast<uchar>(v.z));
}

template <> inline __host__ __device__ int3 as(const float3 v)
{
    return make_int3(
        static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z));
}

/**
 * From float4
 */

template <> inline __host__ __device__ float4 as(const float4 v) { return v; }

template <> inline __host__ __device__ uchar4 as(const float4 v)
{
    return make_uchar4(static_cast<uchar>(v.x), static_cast<uchar>(v.y),
        static_cast<uchar>(v.z), static_cast<uchar>(v.w));
}

template <> inline __host__ __device__ int4 as(const float4 v)
{
    return make_int4(static_cast<int>(v.x), static_cast<int>(v.y),
        static_cast<int>(v.z), static_cast<int>(v.w));
}

/**
 * From uchar
 */

template <> inline __host__ __device__ float as(const uchar v)
{
    return static_cast<float>(v);
}

template <> inline __host__ __device__ uchar as(const uchar v) { return v; }

template <> inline __host__ __device__ int as(const uchar v)
{
    return static_cast<int>(v);
}

template <> inline __host__ __device__ float1 as(const uchar v)
{
    return make_float1(static_cast<float>(v));
}

template <> inline __host__ __device__ uchar1 as(const uchar v)
{
    return make_uchar1(v);
}

template <> inline __host__ __device__ int1 as(const uchar v)
{
    return make_int1(static_cast<int>(v));
}

/**
 * From uchar 1
 */

template <> inline __host__ __device__ float as(const uchar1 v)
{
    return static_cast<float>(v.x);
}

template <> inline __host__ __device__ uchar as(const uchar1 v) { return v.x; }

template <> inline __host__ __device__ int as(const uchar1 v)
{
    return static_cast<int>(v.x);
}

template <> inline __host__ __device__ float1 as(const uchar1 v)
{
    return make_float1(static_cast<float>(v.x));
}

template <> inline __host__ __device__ uchar1 as(const uchar1 v)
{
    return make_uchar1(v.x);
}

template <> inline __host__ __device__ int1 as(const uchar1 v)
{
    return make_int1(static_cast<int>(v.x));
}

/**
 * From float2
 */

template <> inline __host__ __device__ float2 as(const uchar2 v)
{
    return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}

template <> inline __host__ __device__ uchar2 as(const uchar2 v)
{
    return make_uchar2(v.x, v.y);
}

template <> inline __host__ __device__ int2 as(const uchar2 v)
{
    return make_int2(static_cast<int>(v.x), static_cast<int>(v.y));
}

/**
 * From float 3
 */

template <> inline __host__ __device__ float3 as(const uchar3 v)
{
    return make_float3(static_cast<float>(v.x), static_cast<float>(v.y),
        static_cast<float>(v.z));
}

template <> inline __host__ __device__ uchar3 as(const uchar3 v)
{
    return make_uchar3(v.x, v.y, v.z);
}

template <> inline __host__ __device__ int3 as(const uchar3 v)
{
    return make_int3(
        static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z));
}

/**
 * From float4
 */

template <> inline __host__ __device__ float4 as(const uchar4 v)
{
    return make_float4(static_cast<float>(v.x), static_cast<float>(v.y),
        static_cast<float>(v.z), static_cast<float>(v.w));
}

template <> inline __host__ __device__ uchar4 as(const uchar4 v)
{
    return make_uchar4(v.x, v.y, v.z, v.w);
}

template <> inline __host__ __device__ int4 as(const uchar4 v)
{
    return make_int4(static_cast<int>(v.x), static_cast<int>(v.y),
        static_cast<int>(v.z), static_cast<int>(v.w));
}

/**
 * From int
 */
template <> inline __host__ __device__ float as(const int v)
{
    return static_cast<float>(v);
}

template <> inline __host__ __device__ uchar as(const int v)
{
    return static_cast<uchar>(v);
}

template <> inline __host__ __device__ int as(const int v) { return v; }

template <> inline __host__ __device__ float1 as(const int v)
{
    return make_float1(static_cast<float>(v));
}

template <> inline __host__ __device__ uchar1 as(const int v)
{
    return make_uchar1(static_cast<uchar>(v));
}

template <> inline __host__ __device__ int1 as(const int v)
{
    return make_int1(v);
}

/**
 * From int1
 */

template <> inline __host__ __device__ float as(const int1 v)
{
    return static_cast<float>(v.x);
}

template <> inline __host__ __device__ uchar as(const int1 v)
{
    return static_cast<uchar>(v.x);
}

template <> inline __host__ __device__ int as(const int1 v) { return v.x; }

template <> inline __host__ __device__ float1 as(const int1 v)
{
    return make_float1(static_cast<float>(v.x));
}

template <> inline __host__ __device__ uchar1 as(const int1 v)
{
    return make_uchar1(static_cast<uchar>(v.x));
}

template <> inline __host__ __device__ int1 as(const int1 v) { return v; }

/**
 * From int2
 */

template <> inline __host__ __device__ float2 as(const int2 v)
{
    return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}

template <> inline __host__ __device__ uchar2 as(const int2 v)
{
    return make_uchar2(static_cast<uchar>(v.x), static_cast<uchar>(v.y));
}

template <> inline __host__ __device__ int2 as(const int2 v) { return v; }

/**
 * From int3
 */

template <> inline __host__ __device__ float3 as(const int3 v)
{
    return make_float3(static_cast<float>(v.x), static_cast<float>(v.y),
        static_cast<float>(v.z));
}

template <> inline __host__ __device__ uchar3 as(const int3 v)
{
    return make_uchar3(static_cast<uchar>(v.x), static_cast<uchar>(v.y),
        static_cast<uchar>(v.z));
}

template <> inline __host__ __device__ int3 as(const int3 v) { return v; }

/**
 * From int4
 */

template <> inline __host__ __device__ float4 as(const int4 v)
{
    return make_float4(static_cast<float>(v.x), static_cast<float>(v.y),
        static_cast<float>(v.z), static_cast<float>(v.w));
}

template <> inline __host__ __device__ uchar4 as(const int4 v)
{
    return make_uchar4(static_cast<uchar>(v.x), static_cast<uchar>(v.y),
        static_cast<uchar>(v.z), static_cast<uchar>(v.w));
}

template <> inline __host__ __device__ int4 as(const int4 v) { return v; }

/**
 * Convert 4channel RGBA to 1channel greyscale
 */
template <typename T> __host__ __device__ float rgba2gray(const T v);

template <> inline __host__ __device__ float rgba2gray(const float4 v)
{
    return (0.21f * v.x + 0.72f * v.y + 0.07f * v.z) * v.w;
}

template <> inline __host__ __device__ float rgba2gray(const uchar4 v)
{
    return (0.21f * static_cast<float>(v.x) + 0.72f * static_cast<float>(v.y)
               + 0.07f * static_cast<float>(v.z))
        * static_cast<float>(v.w / 255.f);
}

template <> inline __host__ __device__ float rgba2gray(const int4 v)
{
    return (0.21f * static_cast<float>(v.x) + 0.72f * static_cast<float>(v.y)
               + 0.07f * static_cast<float>(v.z))
        * static_cast<float>(v.w / 255.f);
}

/**
 * Convert 3channel RGB to 1channel greyscale
 */
template <typename T> __host__ __device__ float rgb2gray(const T v);

template <> inline __host__ __device__ float rgb2gray(const float3 v)
{
    return 0.21f * v.x + 0.72f * v.y + 0.07f * v.z;
}

template <> inline __host__ __device__ float rgb2gray(const uchar3 v)
{
    return 0.21f * static_cast<float>(v.x) + 0.72f * static_cast<float>(v.y)
        + 0.07f * static_cast<float>(v.z);
}

template <> inline __host__ __device__ float rgb2gray(const int3 v)
{
    return 0.21f * static_cast<float>(v.x) + 0.72f * static_cast<float>(v.y)
        + 0.07f * static_cast<float>(v.z);
}

/**
 * Convert 1channel greyscale to 3channel RGB
 */
template <typename T> __host__ __device__ T gray2rgb(const float v);

template <> inline __host__ __device__ float3 gray2rgb(const float v)
{
    return make_float3(v, v, v);
}

template <> inline __host__ __device__ uchar3 gray2rgb(const float v)
{
    return make_uchar3(
        static_cast<uchar>(v), static_cast<uchar>(v), static_cast<uchar>(v));
}

template <> inline __host__ __device__ int3 gray2rgb(const float v)
{
    return make_int3(
        static_cast<int>(v), static_cast<int>(v), static_cast<int>(v));
}

/**
 * Convert 1channel greyscale to 4channel RGBA
 */
template <typename T> __host__ __device__ T gray2rgba(const float v);

template <> inline __device__ float4 gray2rgba(const float v)
{
    return make_float4(v, v, v, 1.f);
}

template <> inline __device__ uchar4 gray2rgba(const float v)
{
    return make_uchar4(static_cast<uchar>(v), static_cast<uchar>(v),
        static_cast<uchar>(v), 255);
}

template <> inline __device__ int4 gray2rgba(const float v)
{
    return make_int4(
        static_cast<int>(v), static_cast<int>(v), static_cast<int>(v), 255);
}

/**
 * Convert 3channel RGB to 4channel RGBA
 */
template <typename T> __host__ __device__ T rgb2rgba(const float3 v);

template <> inline __device__ float4 rgb2rgba(const float3 v)
{
    return make_float4(v.x, v.y, v.z, 1.f);
}

template <> inline __device__ uchar4 rgb2rgba(const float3 v)
{
    return make_uchar4(static_cast<uchar>(v.x), static_cast<uchar>(v.y),
        static_cast<uchar>(v.z), 255);
}

template <> inline __device__ int4 rgb2rgba(const float3 v)
{
    return make_int4(static_cast<int>(v.x), static_cast<int>(v.y),
        static_cast<int>(v.z), 255);
}

template <typename T> __host__ __device__ T rgb2rgba(const uchar3 v);

template <> inline __device__ float4 rgb2rgba(const uchar3 v)
{
    return make_float4(static_cast<float>(v.x), static_cast<float>(v.y),
        static_cast<float>(v.z), 1.f);
}

template <> inline __device__ uchar4 rgb2rgba(const uchar3 v)
{
    return make_uchar4(v.x, v.y, v.z, 255);
}

template <> inline __device__ int4 rgb2rgba(const uchar3 v)
{
    return make_int4(static_cast<int>(v.x), static_cast<int>(v.y),
        static_cast<int>(v.z), 255);
}

template <typename T> __host__ __device__ T rgb2rgba(const int3 v);

template <> inline __device__ float4 rgb2rgba(const int3 v)
{
    return make_float4(static_cast<float>(v.x), static_cast<float>(v.y),
        static_cast<float>(v.z), 1.f);
}

template <> inline __device__ uchar4 rgb2rgba(const int3 v)
{
    return make_uchar4(static_cast<uchar>(v.x), static_cast<uchar>(v.y),
        static_cast<uchar>(v.z), 255);
}

template <> inline __device__ int4 rgb2rgba(const int3 v)
{
    return make_int4(v.x, v.y, v.z, 255);
}

/**
 * Convert 4channel RGBA to 3channel RGB
 * Assume float has max 1.f in last channel
 */

template <typename T> __host__ __device__ T rgba2rgb(const float4 v);

template <> inline __device__ float3 rgba2rgb(const float4 v)
{
    return make_float3(v.x * v.w, v.y * v.w, v.z * v.w);
}

template <> inline __device__ uchar3 rgba2rgb(const float4 v)
{
    return make_uchar3(static_cast<uchar>(v.x * v.w),
        static_cast<uchar>(v.y * v.w), static_cast<uchar>(v.z * v.w));
}

template <> inline __device__ int3 rgba2rgb(const float4 v)
{
    return make_int3(static_cast<int>(v.x * v.w), static_cast<int>(v.y * v.w),
        static_cast<int>(v.z * v.w));
}

/**
 * RGBA to RGB
 * the last component will be 0..255 for uchar
 */
template <typename T> __host__ __device__ T rgba2rgb(const uchar4 v);

template <> inline __device__ float3 rgba2rgb(const uchar4 v)
{
    return make_float3(static_cast<float>(v.x * (v.w / 255.f)),
        static_cast<float>(v.y * (v.w / 255.f)),
        static_cast<float>(v.z * (v.w / 255.f)));
}

template <> inline __device__ uchar3 rgba2rgb(const uchar4 v)
{
    return make_uchar3(
        v.x * (v.w / 255), v.y * (v.w / 255), v.z * (v.w / 255));
}

template <> inline __device__ int3 rgba2rgb(const uchar4 v)
{
    return make_int3(static_cast<int>(v.x * (v.w / 255)),
        static_cast<int>(v.y * (v.w / 255)),
        static_cast<int>(v.z * (v.w / 255)));
}

template <typename T> __host__ __device__ T rgba2rgb(const int4 v);

template <> inline __device__ float3 rgba2rgb(const int4 v)
{
    return make_float3(static_cast<float>(v.x * (v.w / 255.f)),
        static_cast<float>(v.y * (v.w / 255.f)),
        static_cast<float>(v.z * (v.w / 255.f)));
}

template <> inline __device__ uchar3 rgba2rgb(const int4 v)
{
    return make_uchar3(static_cast<uchar>(v.x * (v.w / 255.f)),
        static_cast<uchar>(v.y * (v.w / 255.f)),
        static_cast<uchar>(v.z * (v.w / 255.f)));
}

template <> inline __device__ int3 rgba2rgb(const int4 v)
{
    return make_int3(
        v.x * (v.w / 255.f), v.y * (v.w / 255.f), v.z * (v.w / 255.f));
}

/**
 * Get a channel of a multi-channel variable
 */

template <typename T,
    typename std::enable_if<is_float_type<T>::value, T>::type* = nullptr>
__device__ float d_get(const T& val, const ushort component);

template <typename T>
__device__ float d_get(const float4& val, const ushort component)
{
    assert(component < 4);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    case 2:
        return val.z;
    case 3:
        return val.w;
    default:
        return 0.f;
    }
}

template <typename T>
__device__ float d_get(const float3& val, const ushort component)
{
    assert(component < 3);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    case 2:
        return val.z;
    default:
        return 0.f;
    }
}

template <typename T>
__device__ float d_get(const float2& val, const ushort component)
{
    assert(component < 2);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    default:
        return 0.f;
    }
}

template <typename T>
__device__ float d_get(const float1& val, const ushort component)
{
    assert(component < 1);
    return val.x;
}

template <typename T>
__device__ float d_get(const float& val, const ushort component)
{
    assert(component == 0);
    return val;
}

template <typename T,
    typename std::enable_if<is_uchar_type<T>::value, T>::type* = nullptr>
__device__ uchar d_get(const T& val, const ushort component);

template <typename T>
__device__ uchar d_get(const uchar4& val, const ushort component)
{
    assert(component < 4);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    case 2:
        return val.z;
    case 3:
        return val.w;
    default:
        return 0;
    }
}

template <typename T>
__device__ uchar d_get(const uchar3& val, const ushort component)

{
    assert(component < 3);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    case 2:
        return val.z;
    default:
        return 0;
    }
}

template <typename T>
__device__ uchar d_get(const uchar2& val, const ushort component)

{
    assert(component < 2);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    default:
        return 0;
    }
}

template <typename T>
__device__ uchar d_get(const uchar1& val, const ushort component)
{
    assert(component < 1);
    switch (component)
    {
    case 0:
        return val.x;
    default:
        return 0.f;
    }
}

template <typename T>
__device__ uchar d_get(const uchar& val, const ushort component)
{
    assert(component == 0);
    return val;
}

template <typename T,
    typename std::enable_if<is_int_type<T>::value, T>::type* = nullptr>
__device__ int d_get(const T& val, const ushort component);

template <typename T>
__device__ int d_get(const int4& val, const ushort component)
{
    assert(component < 4);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    case 2:
        return val.z;
    case 3:
        return val.w;
    default:
        return 0;
    }
}

template <typename T>
__device__ int d_get(const int3& val, const ushort component)
{
    assert(component < 3);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    case 2:
        return val.z;
    default:
        return 0;
    }
}

template <typename T>
__device__ int d_get(const int2& val, const ushort component)
{
    assert(component < 2);
    switch (component)
    {
    case 0:
        return val.x;
    case 1:
        return val.y;
    default:
        return 0;
    }
}

template <typename T>
__device__ int d_get(const int1& val, const ushort component)
{
    assert(component < 1);
    switch (component)
    {
    case 0:
        return val.x;
    default:
        return 0;
    }
}

template <typename T>
__device__ int d_get(const int& val, const ushort component)
{
    assert(component == 0);
    return val;
}

} // image