/**
 * @file   conversion.h
 * @brief  Conversion between base and color types of images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/conversion.h"
#include "cuimage/cuda/devptr.h"

namespace cuimage
{

template <typename T, typename TO>
void cu_ColorToGray(DevPtr<TO> output, const DevPtr<T>& input);

template <typename T, typename TO>
void cu_GrayToColor(DevPtr<TO> output, const DevPtr<T>& input);

template <typename T, typename TO>
void cu_ColorToColor(DevPtr<TO> output, const DevPtr<T>& input);

template <typename T, typename TO>
void cu_Convert(DevPtr<TO> output, const DevPtr<T>& input);

template <typename T, typename TO>
void cu_GetComponent(
    DevPtr<TO> output, const DevPtr<T>& input, const ushort component);

} // image