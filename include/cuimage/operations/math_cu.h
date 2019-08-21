/**
 * @file   operations_cu.h
 * @brief  Operations on images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/devptr.h"

namespace cuimage
{
/**
 * Add other image to image pixelwise
 */
template <typename T> void cu_AddTo(DevPtr<T> image, const DevPtr<T>& other);

/**
 * Multiply image by other image pixelwise
 */
template <typename T>
void cu_MultiplyBy(DevPtr<T> image, const DevPtr<T>& other);

/**
 * Subtract other image from image pixelwise
 */
template <typename T>
void cu_SubtractFrom(DevPtr<T> image, const DevPtr<T>& other);

/**
 * Divide image by other image pixelwise
 */
template <typename T>
void cu_DivideBy(DevPtr<T> image, const DevPtr<T>& other);

/**
 * Add a value to every pixel in image
 */
template <typename T> void cu_AddTo(DevPtr<T> image, const T& value);

/**
 * Multiply each pixel in image by a value
 */
template <typename T> void cu_MultiplyBy(DevPtr<T> image, const T& value);

/**
 * Subtract a value from every pixel in image
 */
template <typename T> void cu_SubtractFrom(DevPtr<T> image, const T& value);

/**
 * Divide every pixel in image by a value
 */
template <typename T> void cu_DivideBy(DevPtr<T> image, const T& value);

} // image
