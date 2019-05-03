/**
 * @file   conversion.h
 * @brief  Conversion between color and greyscale types
 * @author Florian Windolf
 */
#pragma once

#include "image/cuda/type.h"

namespace image
{

template <typename T>
T* cu_greyToRgb(const void* h_data, const size_t width, const size_t height);

template <typename T>
T* cu_greyToRgba(const void* h_data, const size_t width, const size_t height);

template <typename T>
T* cu_rgbToGrey(const void* h_data, const size_t width, const size_t height);

template <typename T>
T* cu_rgbToRgba(const void* h_data, const size_t width, const size_t height);

template <typename T>
T* cu_rgbaToGrey(const void* h_data, const size_t width, const size_t height);

template <typename T>
T* cu_rgbaToRgb(const void* h_data, const size_t width, const size_t height);

} 