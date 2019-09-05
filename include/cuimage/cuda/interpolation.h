#pragma once
#include "arithmetic.h"
#include "type.h"
#include "utils.h"

#include <cassert>
#include <cuda_runtime.h>
#include <math.h>
#include <nvfunctional>

namespace cuimage
{

/**
 * Return a weighted average between v0, v1
 * (p1 - p) is bigger for p -> p0: more influence of v0
 * (p - p0) is bigger for p -> p1: more influence of v1
 */
template <typename T>
__device__ T d_between(
    const T& v0, const T& v1, const int p0, const int p1, const float p)
{
    if (v0 == v1)
        return v0;

    float d = p1 - p0;
    return v0 * ((p1 - p) / d) + v1 * ((p - p0) / d);
}

/**
 * Interpolate linarily between v0 and v1
 */
template <typename T>
__device__ T d_interpolate_linear(const T& v0, const T& v1, const uchar m0,
    const uchar m1, const int p0, const int p1, const float p)
{
    return d_between(v0, v1, p0, p1, p);
}

/**
 * Interpolate linarily between v0 and v1, but do not consider masked points
 */
template <typename T>
__device__ T d_interpolate_linear_masked(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
{
    if (!m0 && !m1)
        return make<T>(0.f);
    else if (!m0)
        return v1;
    else if (!m1)
        return v0;
    else
        return d_between(v0, v1, p0, p1, p);
}

/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to
 * p- and p+ Takes only into account valid pixels
 */
template <typename T>
__device__ T d_interpolate_linear_valid(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    // Both pixels valid -> interpolate between
    if (isvalid(v0) && isvalid(v1))
        return d_between(v0, v1, p0, p1, p);
    else if (isvalid(v0))
        return v0;
    else if (isvalid(v1))
        return v1;
    else
        return make<T>(nanf(""));
}

/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to
 * p- and p+ Takes only into account valid pixels
 */
template <typename T>
__device__ T d_interpolate_linear_nonzero(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    // Both pixels valid -> interpolate between
    if (isvalid(v0) && !iszero(v0) && isvalid(v1) && !iszero(v1))
        return d_between(v0, v1, p0, p1, p);
    else if (isvalid(v0) && !iszero(v0))
        return v0;
    else if (isvalid(v1) && !iszero(v1))
        return v1;
    else
        return make<T>(nanf(""));
}

/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to
 * p- and p+ Takes only into account valid pixels and pixels that are not
 * masked out (mask != 0)
 */
template <typename T>
__device__ T d_interpolate_linear_valid_masked(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
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
        return make<T>(nanf("")); // some type of NaN
}

/**
 * Interpolate linearily between v0 and v1, depending on the distance from p to
 * p- and p+ Takes only into account valid pixels and pixels that are not
 * masked out (mask != 0)
 */
template <typename T>
__device__ T d_interpolate_linear_nonzero_masked(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    // Both or one of the pixels is valid
    if (isvalid(v0) && !iszero(v0) && m0 && isvalid(v1) && m1 && !iszero(v1))
        return d_between(v0, v1, p0, p1, p);
    else if (isvalid(v0) && m0 && !iszero(v0))
        return v0;
    else if (isvalid(v1) && m1 && !iszero(v1))
        return v1;

    // Both pixels are masked
    if (!m0 && !m1)
        return make<T>(0.f);

    // Both pixels are invalid or zero
    return make<T>(nanf(""));
}

/**
 * Interpolate by getting the closest value to p
 */
template <typename T>
__device__ T d_interpolate_nearest(const T& v0, const T& v1, const uchar m0,
    const uchar m1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    if (v0 == v1)
        return v0;

    // Take closest
    if (p1 - p > p - p0)
        return v0; // Influence of v0 bigger
    else
        return v1;
}

/**
 * Interpolate by getting the closest valid value to p
 */
template <typename T>
__device__ T d_interpolate_nearest_valid(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
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

/**
 * Interpolate by getting the closest value to p that is not masked
 * Returns 0 equivalent if both values are masked out
 */
template <typename T>
__device__ T d_interpolate_nearest_masked(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    if (v0 == v1)
        return v0;

    // Both pixel valid -> take closest
    if (!m0)
        return v1; // might be NaN
    else if (!m1)
        return v0; // might be NaN
    else if (p1 - p > p - p0)
        return v0; // Influence of v0 bigger
    else
        return v1;
}

/**
 * Interpolate by getting the closest valid value to p that is not masked
 * Returns 0 equivalent if both values are masked or if both are invalid
 */
template <typename T>
__device__ T d_interpolate_nearest_valid_masked(const T& v0, const T& v1,
    const uchar m0, const uchar m1, const int p0, const int p1, const float p)
{
    assert(p1 > p0);

    if (v0 == v1)
        return v0;

    if (!m0 && !m1)
        return make<T>(0.f);
    else if (!isvalid(v0) && !isvalid(v1))
        return make<T>(0.f);

    // Both pixel valid -> take closest
    if (!m0 || !isvalid(v0))
        return v1; // might be NaN
    else if (!m1 || !isvalid(v1))
        return v0; // might be NaN
    else if (p1 - p > p - p0)
        return v0; // Influence of v0 bigger
    else
        return v1;
}

} // image