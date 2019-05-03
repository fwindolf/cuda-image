/**
 * @file   resize_cu.h
 * @brief  Resizing of images
 * @author Florian Windolf
 */
#pragma once

#include "image/cuda/type.h"
#include "image/cuda/devptr.h"

namespace image
{
    template <typename T>
    void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

    template <typename T>
    void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input);

    template <typename T>
    void cu_ApplyMask(DevPtr<T> image, const DevPtr<uchar>& mask);

} // image
