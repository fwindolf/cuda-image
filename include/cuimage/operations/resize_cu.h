/**
 * @file   resize_cu.h
 * @brief  Resizing of images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/devptr.h"
#include "cuimage/cuda/type.h"

namespace cuimage
{
/**
 * Resize input to output dimension using nearest neighbor interpolation
 */
template <typename T>
void cu_ResizeNearest(DevPtr<T> output, const DevPtr<T>& input);

/**
 * Resize input to output dimension using nearest neighbor interpolation
 * for all pixels in mask
 */
template <typename T>
void cu_ResizeNearest(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

/**
 * Resize input to output dimension using linear interpolation
 */
template <typename T>
void cu_ResizeLinear(DevPtr<T> output, const DevPtr<T>& input);

/**
 * Resize input to output dimension using linear interpolation
 * for all valid pixels.
 * Valid pixels do not contain NaN or Infinity.
 */
template <typename T>
void cu_ResizeLinearValid(DevPtr<T> output, const DevPtr<T>& input);

/**
 * Resize input to output dimension using linear interpolation
 * for all pixels in mask
 */
template <typename T>
void cu_ResizeLinear(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

/**
 * Resize input to output dimension using nearest neighbor interpolation
 * for all valid pixels that are also in mask.
 * Valid pixels do not contain NaN or Infinity.
 */
template <typename T>
void cu_ResizeLinearValid(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

/**
 * Resize input to output dimension using nearest neighbor interpolation
 * for all valid pixels that are also in mask.
 * Valid pixels do not contain NaN or Infinity or Zero.
 */
template <typename T>
void cu_ResizeLinearNonZero(
    DevPtr<T> output, const DevPtr<T>& input, const DevPtr<uchar>& mask);

/**
 * Apply the mask to an image by setting all pixels in image to zero
 * that are also zero in mask.
 */
template <typename T>
void cu_ApplyMask(DevPtr<T> image, const DevPtr<uchar>& mask);

} // image
