/**
 * @file   devptr.h
 * @brief  Device pointer wrapper for easy access in kernels
 * @author Florian Windolf
 */
#pragma once

#include <assert.h>

#include "image/cuda/type.h"
#include "image/cuda/utils.h"

namespace image
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
    DevPtr(T* data, const size_t width, const size_t height);

    DevPtr(const size_t width, const size_t height);

    /**
     * Copy by creating new header for data
     */
    DevPtr(const DevPtr<T>& other);

    ~DevPtr();

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

    const size_t width, height;
    T* data; // row major data on device
private:    
    DevPtr();

    const size_t pitch_;    
};

/**
 * Implementation
 */

template <typename T>
DevPtr<T>::DevPtr(T* ddata, const size_t width, const size_t height)
: width(width),
  height(height),
  data(ddata),
  pitch_(0)
{
}

template <typename T>
DevPtr<T>::DevPtr(const size_t width, const size_t height)
: width(width),
  height(height),
  data(nullptr),
  pitch_(0)
{
    cudaMalloc(&data, width * height * sizeof(T));
}


template <typename T>
DevPtr<T>::DevPtr(const DevPtr<T>& other)
 : width(other.width),
   height(other.height),
   data(other.data),
   pitch_(0)
{
}

template <typename T>
DevPtr<T>::~DevPtr()
{
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
    if (this != &other)
    {
        if (other.width == width && other.height == height)
            cudaSafeCall(cudaMemcpy(data, other.data, width * height * sizeof(T), cudaMemcpyDeviceToDevice));
        else
            throw std::runtime_error("Assigning data from different size not implemented yet!");
    }
    return *this;
}

/**
 * Explicit instantiation
 */
#define INST_DEVPTR(type, name) \
    template class name<type>;

FOR_EACH_TYPE(INST_DEVPTR, DevPtr)


} // image