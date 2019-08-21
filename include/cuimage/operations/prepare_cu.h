/**
 * @file   prepare_cu.h
 * @brief  Kernel calls for preparing images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/devptr.h"

namespace cuimage
{

/**
 * Fill the result with the squared norm of each pixel
 */
template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_SquareNorm(DevPtr<Q> result, const DevPtr<T> image);

/**
 * Fill the result with squares of every pixel value
 */
template <typename T> void cu_Square(DevPtr<T> result, const DevPtr<T> image);

/**
 * Fill the result with the sum of all pixel channels
 */
template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_PixelSum(DevPtr<Q> result, const DevPtr<T> image);

/**
 * Fill the result with 1s for every valid pixels
 */
template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_MarkValid(DevPtr<Q> result, const DevPtr<T> image);

/**
 * Fill the result with 1s for every pixel that is valid and not zero
 */
template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_MarkNonzero(DevPtr<Q> result, const DevPtr<T> image);

/**
 * Fill the result with 1s for every pixel that is valid and not zero
 */
template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_MarkAbove(DevPtr<Q> result, const DevPtr<T> image, const T& value);
}
