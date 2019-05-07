#pragma once
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#include <limits>

#include "arithmetic.h"
#include "type.h"
#include "utils.h"

namespace cuimage
{


/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to p- and p+
 * Takes only into account valid pixels
 */
template <typename T>
__host__ __device__ T interpolate_linear(const T v0, const T v1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    if (v0 == v1)
        return v0;

    // Both pixels valid -> interpolate between
    if (!isnan(v0) && !isnan(v1))
    {
        float d = p1 - p0;
        float dm = (p1 - p) / d; // Influence of v0: bigger for p -> p0
        float dp = (p - p0) / d; // Influence of v1: bigger for p -> p1
        return v0 * dm + v1 * dp;
    }        
    else if (!isnan(v0))
    {
        return v0;
    }        
    else if (!isnan(v1))
    {
        return v1;
    }        
    else 
    {
        return v0; // some type of NaN   
    }
}

template <typename T>
__host__ __device__ T interpolate_nearest(const T v0, const T v1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    if (v0 == v1)
        return v0;

    // Both pixel valid -> take closest
    if (isnan(v0))
        return v1; // might be NaN
    else if (isnan(v1))
        return v0; // might be NaN
    else if (p1 - p > p - p0)
        return v0; // Influence of v0 bigger
    else
        return v1;
}

}  // image