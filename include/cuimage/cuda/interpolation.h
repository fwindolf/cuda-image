#pragma once
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#include <limits>

#include "type.h"

namespace cuimage
{


/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to p- and p+
 * Takes only into account valid pixels
 */
template <typename T>
__device__ T interpolate_linear(const T v0, const T v1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    if (v0 == v1)
        return v0;

    // Both pixels valid -> interpolate between
    if (!isnan(v0) && !isnan(v1))
        return v0 * (p - p0) / (p1 - p0) + v1 * (p1 - p)/(p1 - p0);
    else if (!isnan(v0))
        return v0;
    else if (!isnan(v1))
        return v1;
    else 
        return v0; // some type of NaN   
}

template <typename T>
__device__ T interpolate_nearest(const T v0, const T v1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    if (v0 == v1)
        return v0;

    // Both pixel valid -> take closest
    if (isnan(v0))
        return v1; // might be NaN
    else if (isnan(v1))
        return v0; // might be NaN
    else
        return (p1 - p > p - p0 ? v1 : v0);
}

}  // image