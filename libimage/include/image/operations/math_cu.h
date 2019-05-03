/**
 * @file   operations_cu.h
 * @brief  Operations on images
 * @author Florian Windolf
 */
#pragma once

#include "image/cuda/devptr.h"

namespace image
{

template <typename T>
void cu_AddTo(DevPtr<T> image, const DevPtr<T>& other);

template <typename T>
void cu_MultiplyBy(DevPtr<T> image, const DevPtr<T>& other);

template <typename T>
void cu_SubtractFrom(DevPtr<T> image, const DevPtr<T>& other);

template <typename T>
void cu_DivideBy(DevPtr<T> image, const DevPtr<T>& other);


template <typename T>
void cu_AddTo(DevPtr<T> image, const T& value);

template <typename T>
void cu_MultiplyBy(DevPtr<T> image, const T& value);

template <typename T>
void cu_SubtractFrom(DevPtr<T> image, const T& value);

template <typename T>
void cu_DivideBy(DevPtr<T> image, const T& value);

} // image
