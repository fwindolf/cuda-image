/**
 * @file   resize_cu.h
 * @brief  Resizing of images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/type.h"
#include "cuimage/cuda/devptr.h"

namespace cuimage
{
    enum ResizeMode
    {
        NEAREST, 
        LINEAR, 
        LINEAR_VALID, 
        LINEAR_NONZERO        
    };

    template <typename T>
    void cu_ResizeNearest(DevPtr<T> output, const DevPtr<T>& input);

    template <typename T>
    void cu_ResizeNearest(DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

    template <typename T>
    void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input);

    template <typename T>
    void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

    template <typename T>
    void cu_ResizeLinearValid(DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

    template <typename T>
    void cu_ResizeLinearNonZero(DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

    template <typename T>
    void cu_ApplyMask(DevPtr<T> image, const DevPtr<uchar>& mask);

} // image
