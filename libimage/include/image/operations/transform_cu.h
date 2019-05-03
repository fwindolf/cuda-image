/**
 * @file   transform_cu.h
 * @brief  Kernel calls for transforming images
 * @author Florian Windolf
 */
#pragma once

#include "image/cuda/devptr.h"

namespace image
{

/**
 * Set every pixel in image to value
 */
template <typename T>
void cu_SetTo(DevPtr<T> image, const T& value);

/**
 * Threshold the image so that every pixel above threshold is value
 */
template <typename T>
void cu_Threshold(DevPtr<T> image, const T& threshold, const T& value);

/**
 * Replace every pixel that holds value with another
 */
template <typename T>
void cu_Replace(DevPtr<T> image, const T& value, const T& with);

/**
 * Replace every NaN value with value
 */
template <typename T>
void cu_ReplaceNan(DevPtr<T> image, const T& value);

} // image
