#pragma once

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Needs to be static in order to be invisible for other translation units.
 * If it is non-static, we get a redefinition error during linking
 */
static void cudaCall(cudaError_t call_status, const char* file, const int line)
{
    if(call_status != cudaSuccess)
    {
        std::cerr << file << "(" << line << ") : Cuda call returned with error " 
                  << cudaGetErrorName(call_status) << "(=" << cudaGetErrorString(call_status) << ")" 
                  << std::endl;
        throw std::runtime_error("Cuda Call failed: " + std::string(cudaGetErrorString(call_status)));
    }
}

#define cudaSafeCall(call_status) cudaCall(call_status, __FILE__ ,__LINE__)
#define cudaCheckLastCall()       cudaCall(cudaGetLastError(), __FILE__, __LINE__);


/**
 * Templated printing of custom types
 */
template <typename T>
__host__ __device__ void print(const T v, const char sep=',', const char del='\n');

template <>
inline __host__ __device__ void print(const float v, const char sep, const char del)
{
    printf("%.3f%c", v, del);
}

template <>
inline __host__ __device__ void print(const int v, const char sep, const char del)
{
    printf("%.d%c", v, del);
}

template <>
inline __host__ __device__ void print(const unsigned char v, const char sep, const char del)
{
    printf("%u%c", v, del);
}

template <>
inline __host__ __device__ void print(const float3 v, const char sep, const char del)
{
    printf("(%.3f%c%.3f%c%.3f)%c", v.x, sep, v.y, sep, v.z, del);
}

template <>
inline __host__ __device__ void print(const int3 v, const char sep, const char del)
{
    printf("(%d%c%d%c%d)%c", v.x, sep, v.y, sep, v.z, del);
}

template <>
inline __host__ __device__ void print(const uchar3 v, const char sep, const char del)
{
    printf("(%u%c%u%c%u)%c", v.x, sep, v.y, sep, v.z, del);
}

template <>
inline __host__ __device__ void print(const float4 v, const char sep, const char del)
{
    printf("(%.3f%c%.3f%c%.3f%c%.3f)%c", v.x, sep, v.y, sep, v.z, sep, v.w, del);
}

template <>
inline __host__ __device__ void print(const int4 v, const char sep, const char del)
{
    printf("(%d%c%d%c%d%c%d)%c", v.x, sep, v.y, sep, v.z, sep, v.w, del);
}

template <>
inline __host__ __device__ void print(const uchar4 v, const char sep, const char del)
{
    printf("(%u%c%u%c%u%c%u)%c", v.x, sep, v.y, sep, v.z, sep, v.w, del);
}

template <>
inline __host__ __device__ void print(const dim3 v, const char sep, const char del)
{
    printf("(%d%c%d%c%d)%c", v.x, sep, v.y, sep, v.z, del);
}