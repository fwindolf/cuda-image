/**
 * @file   resize_cu.h
 * @brief  Resizing of images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/devptr.h"
#include "cuimage/cuda/type.h"

/**
 * Resizing will be executed the following:
 *
 * The original image will be accessed in the middle of a pixel (+0.5|+0.5)
 * compared to pixel coordinates. The resized image will be calculated by
 * re-sampling at the coordinate system that is scaled to fit the original one.
 * So a resampled pixel will be accessed at pixel (+0.5/sx|+0.5/sy) in the
 * original coordinate system. The mode of interpolation decides then how the 4
 * nearest pixels to the resampled pixels (measured to middles of pixels) are
 * combined.
 *
 * NEAREST: Use the value of the original pixel that has the closest distance
 * to the resampled LINEAR: Interpolate linearily between the original pixels
 * LINEAR_MASKED: Interpolate linearily, but only consider original pixels that
 * are inside the mask LINEAR_VALID: Interpolate linearily, but only consider
 * original pixels that are not 0 or nan LINEAR_VALID_MASKED: A combination of
 * _MASKED and _VALID, so only pixels in mask that are not 0 or nan
 *
 * Upsampling:   Factor fx = w_new / w_old, fy = h_new / h_old is > 1
 *               Skips pixels if fx, fy > 2
 * Downsampling: Factor fx = w_new / w_old, fy = h_new / h_old is < 1
 *               Repeats pixels if fx, fy < 0.5
 */

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
 * for all valid pixels that are also not zero.
 */
template <typename T>
void cu_ResizeLinearNonZero(DevPtr<T> output, const DevPtr<T>& input);

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
