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
 * Return a weighted average between v0, v1
 * (p1 - p) is bigger for p -> p0: more influence of v0
 * (p - p0) is bigger for p -> p1: more influence of v1
 */
template <typename T>
__host__ __device__ T d_between(const T v0, const T v1, const int p0, const int p1, const float p)
{
    if (v0 == v1)
        return v0;

    float d = p1 - p0;
    return v0 * ((p1 - p) / d) + v1 * ((p - p0) / d);
}

/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to p- and p+
 * Takes only into account valid pixels
 */
template <typename T>
__host__ __device__ T d_interpolate_linear(const T v0, const T v1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    // Both pixels valid -> interpolate between
    if (isvalid(v0) && isvalid(v1))
        return d_between(v0, v1, p0, p1, p);
    else if (isvalid(v0))
        return v0;
    else
        return v1; // v1 is valid -> ok, v1 is invalid -> no valid result anyways
}

/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to p- and p+
 * Takes only into account valid pixels and pixels that are not masked out (mask != 0)
 */
template <typename T>
__host__ __device__ T d_interpolate_linear_masked(const T v0, const T v1, const uchar m0, const uchar m1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    // Both pixels valid -> interpolate between
    if (isvalid(v0) && m0 && isvalid(v1) && m1)
        return d_between(v0, v1, p0, p1, p); 
    else if (isvalid(v0) && m0)
        return v0;
    else if (isvalid(v1) && m1)
        return v1;        
    else if (!m0)
        return make<T>(0.f); // the pixel is masked out
    else
        return v0; // some type of NaN   
}

template <typename T>
__host__ __device__ T d_interpolate_nearest(const T v0, const T v1, const int p0, const int p1, const float p)
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