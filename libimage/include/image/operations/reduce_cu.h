/**
 * @file   reduce_h.h
 * @brief  Kernel calls for reducing images
 * @author Florian Windolf
 */
#pragma once

#include "image/cuda/devptr.h"

namespace image
{

/**
 * Find the minimum value in image
 */
template <typename T>
T cu_Min(DevPtr<T> image);

/**
 * Find the maximum value in image
 */
template <typename T>
T cu_Max(DevPtr<T> image);

/**
 * Get the per sum of all values in image
 */
template <typename T>
float cu_Sum(DevPtr<T> image);

/**
 * Get the L1 norm of all values in image
 */
template <typename T>
float cu_Norm1(DevPtr<T> image);


} // image