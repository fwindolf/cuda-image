/**
 * @file   devptr.h
 * @brief  Device pointer wrapper for easy access in kernels
 * @author Florian Windolf
 */
#pragma once

#include <assert.h>

#include "cuimage/cuda/type.h"
#include "cuimage/cuda/utils.h"

namespace cuimage
{

/**
 * @class DevPtr
 * @brief Device pointer wrapper of cuda type
 * 
 * Inspired by the OpenCV device pointer class
 */
template <typename T>
class DevPtr
{
public:
    /**
     * Create a new DevPtr from exising data
     */
    __host__ __device__ DevPtr(T* data, const size_t width, const size_t height);

    /**
     * Create a new DevPtr that allocates memory
     */
    DevPtr(const size_t width, const size_t height);

    /**
     * Copy by creating new header for data
     */
    __host__ __device__ DevPtr(const DevPtr<T>& other);

    __host__ __device__ ~DevPtr();

    /**
     * Copy the DevPtr to device memory
     */
    DevPtr* upload() const;

    /**
     * Free the data manually
     * Only call this method if the DevPtr was created from size (and allocates memory)
     */
    void free();

    /**
     * Access the data as lvalue via continuous index
     */
    __device__ T& operator()(const size_t idx);

    /**
     * Access the data at continuous index
     */
    __device__ const T operator()(const size_t idx) const;

    /**
     * Access the data as lvalue at column x and row y
     */
    __device__ T& operator()(const size_t x, const size_t y);

    /**
     * Access the data at column x and row y
     */
    __device__ const T operator()(const size_t x, const size_t y) const;

    /**
     * Assign a new DevPtr from another (without copy)
     */
    DevPtr<T>& operator=(const DevPtr<T>& other);

    const int width, height;
    T* data; // row major data on device
private:    
    DevPtr();
};

/**
 * Implementation
 */

template <typename T>
DevPtr<T>::DevPtr(T* ddata, const size_t width, const size_t height)
: width(width),
  height(height),
  data(ddata)
{
}

template <typename T>
DevPtr<T>::DevPtr(const size_t width, const size_t height)
: width(width),
  height(height),
  data(nullptr)
{
    cudaSafeCall(cudaMalloc(&data, width * height * sizeof(T)));
    cudaSafeCall(cudaMemset(data, 0, width * height * sizeof(T)));
}


template <typename T>
 __host__ __device__ DevPtr<T>::DevPtr(const DevPtr<T>& other)
 : width(other.width),
   height(other.height),
   data(other.data)
{
}

template <typename T>
 __host__ __device__ DevPtr<T>::~DevPtr()
{
}

template <typename T>
void DevPtr<T>::free()
{
    cudaSafeCall(cudaFree(data));
    data = nullptr;
}


template <typename T>
__device__ T& DevPtr<T>::operator()(const size_t idx)
{
    assert(idx < width * height);
    return data[idx];
}

template <typename T>
__device__ const T DevPtr<T>::operator()(const size_t idx) const
{
    assert(idx < width * height);
    return data[idx];
}

template <typename T>
__device__ T& DevPtr<T>::operator()(const size_t x, const size_t y)
{
    assert(x < width && y < height);
    return data[y * width + x];
}

template <typename T>
__device__ const T DevPtr<T>::operator()(const size_t x, const size_t y) const
{
    assert(x < width && y < height);
    return data[y * width + x];
}

template <typename T>
DevPtr<T>& DevPtr<T>::operator=(const DevPtr<T>& other)
{
    if (data == nullptr)
        cudaSafeCall(cudaMalloc(&data,  width * height * sizeof(T)));        
    
    if (this == &other)
        throw std::runtime_error("Self-assignment not possible!");

    if (other.width != width && other.height != height)
        throw std::runtime_error("Assignment with different (data) sizes not possible!");

    cudaSafeCall(cudaMemcpy(data, other.data, width * height * sizeof(T), cudaMemcpyDeviceToDevice));
    return *this;
}

/**
 * Explicit instantiation
 */
#define INST_DEVPTR(type, name) \
    template class name<type>;

FOR_EACH_TYPE(INST_DEVPTR, DevPtr)


} // image