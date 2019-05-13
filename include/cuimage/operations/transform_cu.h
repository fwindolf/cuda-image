/**
 * @file   transform_cu.h
 * @brief  Kernel calls for transforming images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/devptr.h"

namespace cuimage
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
 * Threshold the image so that every pixel above or equal to threshold is high, and below threshold is low 
 */
template <typename T>
void cu_Threshold(DevPtr<T> image, const T& threshold, const T& low, const T& high);

/**
 * Threshold the image so that every pixel under or equal to threshold is value
 */
template <typename T>
void cu_ThresholdInv(DevPtr<T> image, const T& threshold, const T& value);

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

/**
 * Get the approximate median of all values in image
 */
template <typename T>
T cu_Median(DevPtr<T> image);


} // image
