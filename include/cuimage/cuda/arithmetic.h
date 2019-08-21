#pragma once
#include "type.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <limits>
#include <math.h>

namespace cuimage
{

/**
 * @brief Create a new T from a single input
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T make(const float v);

/**
 * @brief overload < operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ bool operator<(const T& lhs, const T& rhs);

/**
 * @brief overload <= operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ bool operator<=(const T& lhs, const T& rhs);

/**
 * @brief overload > operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ bool operator>(const T& lhs, const T& rhs);

/**
 * @brief overload >= operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ bool operator>=(const T& lhs, const T& rhs);

/**
 * @brief overload == operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ bool operator==(const T& lhs, const T& rhs);

/**
 * @brief overload + operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator+(const T& lhs, const T& rhs);

/**
 * @brief overload + operator to add broadcasted
 * Omit overloading for basic types, so our definition doesnt interfere with
 * enums, ...
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator+(const float& lhs, const T& rhs)
{
    return make<T>(lhs) + rhs;
}

/**
 * @brief overload + operator to add broadcasted
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator+(const T& lhs, const float& rhs)
{
    return make<T>(rhs) + lhs;
}

/**
 * @brief overload += operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T& operator+=(T& lhs, const T& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

/**
 * @brief overload - operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator-(const T& lhs, const T& rhs);

/**
 * @brief overload - operator to subtract broadcasted
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator-(const float& lhs, const T& rhs)
{
    return make<T>(lhs) - rhs;
}

/**
 * @brief overload - operator to subtract broadcasted
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator-(const T& lhs, const float& rhs)
{
    return lhs - make<T>(rhs);
}

/**
 * @brief overload -= operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T& operator-=(T& lhs, const T& rhs)
{
    lhs = lhs - rhs;
    return lhs;
}

/**
 * @brief overload * operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator*(const T& lhs, const T& rhs);

/**
 * @brief overload * operator to multiply broadcasted
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator*(const T& lhs, const float& rhs)
{
    return make<T>(rhs) * lhs;
}

/**
 * @brief overload * operator to multiply broadcasted
 * Omit overloading for basic types, so our definition doesnt interfere with
 * enums, ...
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator*(const float& lhs, const T& rhs)
{
    return rhs * lhs; // use operator with switched types
}

/**
 * @brief overload *= operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T& operator*=(T& lhs, const T& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

/**
 * @brief overload / operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator/(const T& lhs, const T& rhs);

/**
 * @brief overload / operator to divide broadcasted
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator/(const float& lhs, const T& rhs)
{
    return make<T>(lhs) / rhs;
}

/**
 * @brief overload / operator to divide broadcasted
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T operator/(const T& lhs, const float& rhs)
{
    return lhs / make<T>(rhs);
}

/**
 * @brief overload /= operator to work with all vector types
 */
template <typename T,
    typename std::enable_if<is_extended_vector_type<T>::value,
        T>::type* = nullptr>
__host__ __device__ T& operator/=(T& lhs, const T& rhs)
{
    lhs = lhs / rhs;
    return lhs;
}

/**
 * @brief Get the sum of all members
 */
template <typename T> __host__ __device__ float sum(const T v);

/**
 * @brief Get the the minimum of all positions between the to values l and r
 * The output can be a mix between l and r
 */
template <typename T> __host__ __device__ T min(const T l, const T r);

/**
 * @brief Get the the maximum of all positions between the to values l and r
 * The output can be a mix between l and r
 */
template <typename T> __host__ __device__ T max(const T l, const T r);

/**
 * @brief Get the absolute value per member
 */
template <typename T> __host__ __device__ T abs(const T& v);

/**
 * @brief Get the minimum channel of a multichannel v
 */
template <typename T> __host__ __device__ float min(const T& v);

/**
 * @brief Get the maximum channel of a multichannel v
 */
template <typename T> __host__ __device__ float max(const T& v);

/**
 * @brief Check if the value is not a number (NaN)
 */
#ifdef isnan
#undef isnan
#endif
template <typename T> __host__ __device__ bool isnan(const T& v);

/**
 * @brief Check if the value is infinite
 */
#ifdef isinf
#undef isinf
#endif
template <typename T> __host__ __device__ bool isinf(const T& v);

/**
 * @brief Check if the value provided contains any invalid values
 */
template <typename T> __host__ __device__ bool isvalid(const T& v);

/**
 * @brief Check if the value provided equals to zero
 */
template <typename T> inline __host__ __device__ bool iszero(const T& v)
{
    return (v == make<T>(0.f));
}

/**
 * @brief Clamp the value v between minv and maxv for each element of v
 */
template <typename T>
inline __host__ __device__ T clamp(const T v, const T minv, const T maxv)
{
    assert(maxv > minv);
    return min(max(v, minv), maxv);
}

/**
 * @brief Calculate the L2 norm of the elements of v
 */
template <typename T> inline __host__ __device__ float norm(const T v)
{
    return sqrtf(sum(v * v));
}

/**
 * @brief Normalize the value v so the sum of its elements is 1
 */
template <typename T> inline __host__ __device__ T normalize(const T v)
{
    return v / sqrtf(sum(v * v));
}

/***********************************************************************************
 * Specialization for float (if needed)
 ***********************************************************************************/

template <> inline __host__ __device__ float make(const float v) { return v; }

template <> inline __host__ __device__ float sum(const float v) { return v; }

inline __host__ __device__ float min(const float& v) { return v; }

inline __host__ __device__ float min(const float& l, const float& r)
{
    return (l < r ? l : r);
}

inline __host__ __device__ float max(const float& v) { return v; }

inline __host__ __device__ float max(const float& l, const float& r)
{
    return (l > r ? l : r);
}

template <> inline __host__ __device__ float abs(const float& v)
{
    return (v > 0 ? v : -v);
}

template <> inline __host__ __device__ bool isnan(const float& v)
{
    return (v != v);
}

template <> inline __host__ __device__ bool isinf(const float& v)
{
    return (v == INFINITY || v == -INFINITY);
}

template <> inline __host__ __device__ bool isvalid(const float& v)
{
    return (!isnan(v) && !isinf(v));
}

template <> inline __host__ __device__ bool iszero(const float& v)
{
    return (v == 0);
}

/***********************************************************************************
 * Specialization for int (if needed)
 ***********************************************************************************/

template <> inline __host__ __device__ int make(const float v)
{
    return static_cast<int>(v);
}

template <> inline __host__ __device__ float sum(const int v)
{
    return static_cast<float>(v);
}

template <> inline __host__ __device__ int min(const int l, const int r)
{
    return (l < r ? l : r);
}

template <> inline __host__ __device__ int max(const int l, const int r)
{
    return (l > r ? l : r);
}

template <> inline __host__ __device__ float min(const int& v)
{
    return static_cast<float>(v);
}

template <> inline __host__ __device__ float max(const int& v)
{
    return static_cast<float>(v);
}

template <> inline __host__ __device__ int abs(const int& v)
{
    return (v > 0 ? v : -v);
}

template <> inline __host__ __device__ bool isnan(const int& v)
{
    return false; // int cannot be NaN
}

template <> inline __host__ __device__ bool isinf(const int& v)
{
    return (v == INT_MIN || v == INT_MAX); // if it is int_max or int_min
}

template <> inline __host__ __device__ bool isvalid(const int& v)
{
    return (!isnan(v) && !isinf(v));
}

template <> inline __host__ __device__ bool iszero(const int& v)
{
    return (v == 0);
}

/***********************************************************************************
 * Specialization for uchar (if needed)
 ***********************************************************************************/

template <> inline __host__ __device__ uchar make(const float v)
{
    return static_cast<uchar>(v);
}

template <> inline __host__ __device__ float sum(const uchar v)
{
    return static_cast<float>(v);
}

inline __host__ __device__ uchar min(const uchar& l, const uchar& r)
{
    return (l < r ? l : r);
}

inline __host__ __device__ uchar max(const uchar& l, const uchar& r)
{
    return (l > r ? l : r);
}

template <> inline __host__ __device__ float min(const uchar& v)
{
    return static_cast<float>(v);
}

template <> inline __host__ __device__ float max(const uchar& v)
{
    return static_cast<float>(v);
}

template <> inline __host__ __device__ uchar abs(const uchar& v)
{
    return (v > 0 ? v : -v);
}

template <> inline __host__ __device__ bool isnan(const uchar& v)
{
    return false; // uchar cannot be NaN
}

template <> inline __host__ __device__ bool isinf(const uchar& v)
{
    return false; // uchar cannot be Inf
}

template <> inline __host__ __device__ bool isvalid(const uchar& v)
{
    return (!isnan(v) && !isinf(v));
}

template <> inline __host__ __device__ bool iszero(const uchar& v)
{
    return (v == 0);
}

/***********************************************************************************
 * Specialization for uchar1
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const uchar1& lhs, const uchar1& rhs)
{
    return (lhs.x < rhs.x);
}

template <>
inline __host__ __device__ bool operator<=(
    const uchar1& lhs, const uchar1& rhs)
{
    return (lhs.x <= rhs.x);
}

template <>
inline __host__ __device__ bool operator>(const uchar1& lhs, const uchar1& rhs)
{
    return (lhs.x > rhs.x);
}

template <>
inline __host__ __device__ bool operator>=(
    const uchar1& lhs, const uchar1& rhs)
{
    return (lhs.x >= rhs.x);
}

template <>
inline __host__ __device__ bool operator==(
    const uchar1& lhs, const uchar1& rhs)
{
    return (lhs.x == rhs.x);
}

template <>
inline __host__ __device__ uchar1 operator+(
    const uchar1& lhs, const uchar1& rhs)
{
    return make_uchar1(lhs.x + rhs.x);
}

template <>
inline __host__ __device__ uchar1 operator-(
    const uchar1& lhs, const uchar1& rhs)
{
    return make_uchar1(lhs.x - rhs.x);
}

template <>
inline __host__ __device__ uchar1 operator*(
    const uchar1& lhs, const uchar1& rhs)
{
    return make_uchar1(lhs.x * rhs.x);
}

template <>
inline __host__ __device__ uchar1 operator/(
    const uchar1& lhs, const uchar1& rhs)
{
    return make_uchar1(lhs.x / rhs.x);
}

template <>
inline __host__ __device__ uchar1 operator*(
    const uchar1& lhs, const float& rhs)
{
    return make_uchar1(static_cast<uchar>(lhs.x * rhs));
}

template <> inline __host__ __device__ uchar1 make(const float v)
{
    return make_uchar1(static_cast<uchar>(v));
}

template <> inline __host__ __device__ float sum(const uchar1 v)
{
    return static_cast<float>(v.x);
}

template <>
inline __host__ __device__ uchar1 min(const uchar1 l, const uchar1 r)
{
    return (l.x < r.x ? l : r);
}

template <>
inline __host__ __device__ uchar1 max(const uchar1 l, const uchar1 r)
{
    return (l.x > r.x ? l : r);
}

template <> inline __host__ __device__ float min(const uchar1& v)
{
    return static_cast<float>(v.x);
}

template <> inline __host__ __device__ float max(const uchar1& v)
{
    return static_cast<float>(v.x);
}

template <> inline __host__ __device__ uchar1 abs(const uchar1& v)
{
    return make_uchar1(abs(v.x));
}

template <> inline __host__ __device__ bool isnan(const uchar1& v)
{
    return isnan(v.x);
}

template <> inline __host__ __device__ bool isinf(const uchar1& v)
{
    return isinf(v.x);
}

template <> inline __host__ __device__ bool isvalid(const uchar1& v)
{
    return isvalid(v.x);
}

/***********************************************************************************
 * Specialization for uchar2
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const uchar2& lhs, const uchar2& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y);
}

template <>
inline __host__ __device__ bool operator<=(
    const uchar2& lhs, const uchar2& rhs)
{
    return (lhs.x <= rhs.x && lhs.y <= rhs.y);
}

template <>
inline __host__ __device__ bool operator>(const uchar2& lhs, const uchar2& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y);
}

template <>
inline __host__ __device__ bool operator>=(
    const uchar2& lhs, const uchar2& rhs)
{
    return (lhs.x >= rhs.x && lhs.y >= rhs.y);
}

template <>
inline __host__ __device__ bool operator==(
    const uchar2& lhs, const uchar2& rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

template <>
inline __host__ __device__ uchar2 operator+(
    const uchar2& lhs, const uchar2& rhs)
{
    return make_uchar2(lhs.x + rhs.x, lhs.y + rhs.y);
}

template <>
inline __host__ __device__ uchar2 operator-(
    const uchar2& lhs, const uchar2& rhs)
{
    return make_uchar2(lhs.x - rhs.x, lhs.y - rhs.y);
}

template <>
inline __host__ __device__ uchar2 operator*(
    const uchar2& lhs, const uchar2& rhs)
{
    return make_uchar2(lhs.x * rhs.x, lhs.y * rhs.y);
}

template <>
inline __host__ __device__ uchar2 operator/(
    const uchar2& lhs, const uchar2& rhs)
{
    return make_uchar2(lhs.x / rhs.x, lhs.y / rhs.y);
}

template <>
inline __host__ __device__ uchar2 operator*(
    const uchar2& lhs, const float& rhs)
{
    return make_uchar2(
        static_cast<uchar>(lhs.x * rhs), static_cast<uchar>(lhs.y * rhs));
}

template <> inline __host__ __device__ uchar2 make(const float v)
{
    return make_uchar2(static_cast<uchar>(v), static_cast<uchar>(v));
}

template <> inline __host__ __device__ float sum(const uchar2 v)
{
    return static_cast<float>(v.x + v.y);
}

template <>
inline __host__ __device__ uchar2 min(const uchar2 l, const uchar2 r)
{
    return make_uchar2(min(l.x, r.x), min(l.y, r.y));
}

template <>
inline __host__ __device__ uchar2 max(const uchar2 l, const uchar2 r)
{
    return make_uchar2(max(l.x, r.x), max(l.y, r.y));
}

template <> inline __host__ __device__ float min(const uchar2& v)
{
    return static_cast<float>(min(v.x, v.y));
}

template <> inline __host__ __device__ float max(const uchar2& v)
{
    return static_cast<float>(max(v.x, v.y));
}

template <> inline __host__ __device__ uchar2 abs(const uchar2& v)
{
    return make_uchar2(abs(v.x), abs(v.y));
}

template <> inline __host__ __device__ bool isnan(const uchar2& v)
{
    return (isnan(v.x) || isnan(v.y));
}

template <> inline __host__ __device__ bool isinf(const uchar2& v)
{
    return (isinf(v.x) || isinf(v.y));
}

template <> inline __host__ __device__ bool isvalid(const uchar2& v)
{
    return (isvalid(v.x) && isvalid(v.y));
}

/***********************************************************************************
 * Specialization for uchar3
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const uchar3& lhs, const uchar3& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z);
}

template <>
inline __host__ __device__ bool operator<=(
    const uchar3& lhs, const uchar3& rhs)
{
    return (lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z);
}

template <>
inline __host__ __device__ bool operator>(const uchar3& lhs, const uchar3& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z);
}

template <>
inline __host__ __device__ bool operator>=(
    const uchar3& lhs, const uchar3& rhs)
{
    return (lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z);
}

template <>
inline __host__ __device__ bool operator==(
    const uchar3& lhs, const uchar3& rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

template <>
inline __host__ __device__ uchar3 operator+(
    const uchar3& lhs, const uchar3& rhs)
{
    return make_uchar3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template <>
inline __host__ __device__ uchar3 operator-(
    const uchar3& lhs, const uchar3& rhs)
{
    return make_uchar3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template <>
inline __host__ __device__ uchar3 operator*(
    const uchar3& lhs, const uchar3& rhs)
{
    return make_uchar3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

template <>
inline __host__ __device__ uchar3 operator/(
    const uchar3& lhs, const uchar3& rhs)
{
    return make_uchar3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

template <>
inline __host__ __device__ uchar3 operator*(
    const uchar3& lhs, const float& rhs)
{
    return make_uchar3(static_cast<uchar>(lhs.x * rhs),
        static_cast<uchar>(lhs.y * rhs), static_cast<uchar>(lhs.z * rhs));
}

template <> inline __host__ __device__ uchar3 make(const float v)
{
    return make_uchar3(
        static_cast<uchar>(v), static_cast<uchar>(v), static_cast<uchar>(v));
}

template <> inline __host__ __device__ float sum(const uchar3 v)
{
    return static_cast<float>(v.x + v.y + v.z);
}

template <>
inline __host__ __device__ uchar3 min(const uchar3 l, const uchar3 r)
{
    return make_uchar3(min(l.x, r.x), min(l.y, r.y), min(l.z, r.z));
}

template <>
inline __host__ __device__ uchar3 max(const uchar3 l, const uchar3 r)
{
    return make_uchar3(max(l.x, r.x), max(l.y, r.y), max(l.z, r.z));
}

template <> inline __host__ __device__ float min(const uchar3& v)
{
    return static_cast<float>(min(min(v.x, v.y), v.z));
}

template <> inline __host__ __device__ float max(const uchar3& v)
{
    return static_cast<float>(max(max(v.x, v.y), v.z));
}

template <> inline __host__ __device__ uchar3 abs(const uchar3& v)
{
    return make_uchar3(abs(v.x), abs(v.y), abs(v.z));
}

template <> inline __host__ __device__ bool isnan(const uchar3& v)
{
    return (isnan(v.x) || isnan(v.y) || isnan(v.z));
}

template <> inline __host__ __device__ bool isinf(const uchar3& v)
{
    return (isinf(v.x) || isinf(v.y) || isinf(v.z));
}

template <> inline __host__ __device__ bool isvalid(const uchar3& v)
{
    return (isvalid(v.x) && isvalid(v.y) && isvalid(v.z));
}

/***********************************************************************************
 * Specialization for uchar4
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const uchar4& lhs, const uchar4& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w);
}

template <>
inline __host__ __device__ bool operator<=(
    const uchar4& lhs, const uchar4& rhs)
{
    return (
        lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w);
}

template <>
inline __host__ __device__ bool operator>(const uchar4& lhs, const uchar4& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w);
}

template <>
inline __host__ __device__ bool operator>=(
    const uchar4& lhs, const uchar4& rhs)
{
    return (
        lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w);
}

template <>
inline __host__ __device__ bool operator==(
    const uchar4& lhs, const uchar4& rhs)
{
    return (
        lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}

template <>
inline __host__ __device__ uchar4 operator+(
    const uchar4& lhs, const uchar4& rhs)
{
    return make_uchar4(
        lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

template <>
inline __host__ __device__ uchar4 operator-(
    const uchar4& lhs, const uchar4& rhs)
{
    return make_uchar4(
        lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

template <>
inline __host__ __device__ uchar4 operator*(
    const uchar4& lhs, const uchar4& rhs)
{
    return make_uchar4(
        lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}

template <>
inline __host__ __device__ uchar4 operator/(
    const uchar4& lhs, const uchar4& rhs)
{
    return make_uchar4(
        lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}

template <>
inline __host__ __device__ uchar4 operator*(
    const uchar4& lhs, const float& rhs)
{
    return make_uchar4(static_cast<uchar>(lhs.x * rhs),
        static_cast<uchar>(lhs.y * rhs), static_cast<uchar>(lhs.z * rhs),
        static_cast<uchar>(lhs.w * rhs));
}

template <> inline __host__ __device__ uchar4 make(const float v)
{
    return make_uchar4(static_cast<uchar>(v), static_cast<uchar>(v),
        static_cast<uchar>(v), static_cast<uchar>(v));
}

template <> inline __host__ __device__ float sum(const uchar4 v)
{
    return static_cast<float>(v.x + v.y + v.z + v.w);
}

template <>
inline __host__ __device__ uchar4 min(const uchar4 l, const uchar4 r)
{
    return make_uchar4(
        min(l.x, r.x), min(l.y, r.y), min(l.z, r.z), min(l.w, r.w));
}

template <>
inline __host__ __device__ uchar4 max(const uchar4 l, const uchar4 r)
{
    return make_uchar4(
        max(l.x, r.x), max(l.y, r.y), max(l.z, r.z), max(l.w, r.w));
}

template <> inline __host__ __device__ float min(const uchar4& v)
{
    return static_cast<float>(min(min(min(v.x, v.y), v.z), v.w));
}

template <> inline __host__ __device__ float max(const uchar4& v)
{
    return static_cast<float>(max(max(max(v.x, v.y), v.z), v.w));
}

template <> inline __host__ __device__ uchar4 abs(const uchar4& v)
{
    return make_uchar4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

template <> inline __host__ __device__ bool isnan(const uchar4& v)
{
    return (isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w));
}

template <> inline __host__ __device__ bool isinf(const uchar4& v)
{
    return (isinf(v.x) || isinf(v.y) || isinf(v.z) || isnan(v.w));
}

template <> inline __host__ __device__ bool isvalid(const uchar4& v)
{
    return (isvalid(v.x) && isvalid(v.y) && isvalid(v.z) && isvalid(v.w));
}

/***********************************************************************************
 * Specialization for float1
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const float1& lhs, const float1& rhs)
{
    return (lhs.x < rhs.x);
}

template <>
inline __host__ __device__ bool operator<=(
    const float1& lhs, const float1& rhs)
{
    return (lhs.x <= rhs.x);
}

template <>
inline __host__ __device__ bool operator>(const float1& lhs, const float1& rhs)
{
    return (lhs.x > rhs.x);
}

template <>
inline __host__ __device__ bool operator>=(
    const float1& lhs, const float1& rhs)
{
    return (lhs.x >= rhs.x);
}

template <>
inline __host__ __device__ bool operator==(
    const float1& lhs, const float1& rhs)
{
    return (lhs.x == rhs.x);
}

template <>
inline __host__ __device__ float1 operator+(
    const float1& lhs, const float1& rhs)
{
    return make_float1(lhs.x + rhs.x);
}

template <>
inline __host__ __device__ float1 operator-(
    const float1& lhs, const float1& rhs)
{
    return make_float1(lhs.x - rhs.x);
}

template <>
inline __host__ __device__ float1 operator*(
    const float1& lhs, const float1& rhs)
{
    return make_float1(lhs.x * rhs.x);
}

template <>
inline __host__ __device__ float1 operator/(
    const float1& lhs, const float1& rhs)
{
    return make_float1(lhs.x / rhs.x);
}

template <> inline __host__ __device__ float1 make(const float v)
{
    return make_float1(v);
}

template <> inline __host__ __device__ float sum(const float1 v)
{
    return v.x;
}

template <>
inline __host__ __device__ float1 min(const float1 l, const float1 r)
{
    return (l.x < r.x ? l : r);
}

template <>
inline __host__ __device__ float1 max(const float1 l, const float1 r)
{
    return (l.x > r.x ? l : r);
}

template <> inline __host__ __device__ float min(const float1& v)
{
    return v.x;
}

template <> inline __host__ __device__ float max(const float1& v)
{
    return v.x;
}

template <> inline __host__ __device__ float1 abs(const float1& v)
{
    return make_float1(abs(v.x));
}

template <> inline __host__ __device__ bool isnan(const float1& v)
{
    return (v.x != v.x);
}

template <> inline __host__ __device__ bool isinf(const float1& v)
{
    return (v.x == INFINITY || v.x == -INFINITY);
}

template <> inline __host__ __device__ bool isvalid(const float1& v)
{
    return isvalid(v.x);
}

/***********************************************************************************
 * Specialization for float2
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const float2& lhs, const float2& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y);
}

template <>
inline __host__ __device__ bool operator<=(
    const float2& lhs, const float2& rhs)
{
    return (lhs.x <= rhs.x && lhs.y <= rhs.y);
}

template <>
inline __host__ __device__ bool operator>(const float2& lhs, const float2& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y);
}

template <>
inline __host__ __device__ bool operator>=(
    const float2& lhs, const float2& rhs)
{
    return (lhs.x >= rhs.x && lhs.y >= rhs.y);
}

template <>
inline __host__ __device__ bool operator==(
    const float2& lhs, const float2& rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

template <>
inline __host__ __device__ float2 operator+(
    const float2& lhs, const float2& rhs)
{
    return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

template <>
inline __host__ __device__ float2 operator-(
    const float2& lhs, const float2& rhs)
{
    return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

template <>
inline __host__ __device__ float2 operator*(
    const float2& lhs, const float2& rhs)
{
    return make_float2(lhs.x * rhs.x, lhs.y * rhs.y);
}

template <>
inline __host__ __device__ float2 operator/(
    const float2& lhs, const float2& rhs)
{
    return make_float2(lhs.x / rhs.x, lhs.y / rhs.y);
}

template <> inline __host__ __device__ float2 make(const float v)
{
    return make_float2(v, v);
}

template <> inline __host__ __device__ float sum(const float2 v)
{
    return v.x + v.y;
}

template <>
inline __host__ __device__ float2 min(const float2 l, const float2 r)
{
    return make_float2(min(l.x, r.x), min(l.y, r.y));
}

template <>
inline __host__ __device__ float2 max(const float2 l, const float2 r)
{
    return make_float2(max(l.x, r.x), max(l.y, r.y));
}

template <> inline __host__ __device__ float min(const float2& v)
{
    return min(v.x, v.y);
}

template <> inline __host__ __device__ float max(const float2& v)
{
    return max(v.x, v.y);
}

template <> inline __host__ __device__ float2 abs(const float2& v)
{
    return make_float2(abs(v.x), abs(v.y));
}

template <> inline __host__ __device__ bool isnan(const float2& v)
{
    return (isnan(v.x) || isnan(v.y));
}

template <> inline __host__ __device__ bool isinf(const float2& v)
{
    return (isinf(v.x) || isinf(v.y));
}

template <> inline __host__ __device__ bool isvalid(const float2& v)
{
    return (isvalid(v.x) && isvalid(v.y));
}

/***********************************************************************************
 * Specialization for float3
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const float3& lhs, const float3& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z);
}

template <>
inline __host__ __device__ bool operator<=(
    const float3& lhs, const float3& rhs)
{
    return (lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z);
}

template <>
inline __host__ __device__ bool operator>(const float3& lhs, const float3& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z);
}

template <>
inline __host__ __device__ bool operator>=(
    const float3& lhs, const float3& rhs)
{
    return (lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z);
}

template <>
inline __host__ __device__ bool operator==(
    const float3& lhs, const float3& rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

template <>
inline __host__ __device__ float3 operator+(
    const float3& lhs, const float3& rhs)
{
    return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template <>
inline __host__ __device__ float3 operator-(
    const float3& lhs, const float3& rhs)
{
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template <>
inline __host__ __device__ float3 operator*(
    const float3& lhs, const float3& rhs)
{
    return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

template <>
inline __host__ __device__ float3 operator/(
    const float3& lhs, const float3& rhs)
{
    return make_float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

template <> inline __host__ __device__ float3 make(const float v)
{
    return make_float3(v, v, v);
}

template <> inline __host__ __device__ float sum(const float3 v)
{
    return v.x + v.y + v.z;
}

template <>
inline __host__ __device__ float3 min(const float3 l, const float3 r)
{
    return make_float3(min(l.x, r.x), min(l.y, r.y), min(l.z, r.z));
}

template <>
inline __host__ __device__ float3 max(const float3 l, const float3 r)
{
    return make_float3(max(l.x, r.x), max(l.y, r.y), max(l.z, r.z));
}

template <> inline __host__ __device__ float min(const float3& v)
{
    return min(min(v.x, v.y), v.z);
}

template <> inline __host__ __device__ float max(const float3& v)
{
    return max(max(v.x, v.y), v.z);
}

template <> inline __host__ __device__ float3 abs(const float3& v)
{
    return make_float3(abs(v.x), abs(v.y), abs(v.z));
}

template <> inline __host__ __device__ bool isnan(const float3& v)
{
    return (isnan(v.x) || isnan(v.y) || isnan(v.z));
}

template <> inline __host__ __device__ bool isinf(const float3& v)
{
    return (isinf(v.x) || isinf(v.y) || isinf(v.z));
}

template <> inline __host__ __device__ bool isvalid(const float3& v)
{
    return (isvalid(v.x) && isvalid(v.y) && isvalid(v.z));
}

/***********************************************************************************
 * Specialization for float4
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const float4& lhs, const float4& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w);
}

template <>
inline __host__ __device__ bool operator<=(
    const float4& lhs, const float4& rhs)
{
    return (
        lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w);
}

template <>
inline __host__ __device__ bool operator>(const float4& lhs, const float4& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w);
}

template <>
inline __host__ __device__ bool operator>=(
    const float4& lhs, const float4& rhs)
{
    return (
        lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w);
}

template <>
inline __host__ __device__ bool operator==(
    const float4& lhs, const float4& rhs)
{
    return (
        lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}

template <>
inline __host__ __device__ float4 operator+(
    const float4& lhs, const float4& rhs)
{
    return make_float4(
        lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

template <>
inline __host__ __device__ float4 operator-(
    const float4& lhs, const float4& rhs)
{
    return make_float4(
        lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

template <>
inline __host__ __device__ float4 operator*(
    const float4& lhs, const float4& rhs)
{
    return make_float4(
        lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}

template <>
inline __host__ __device__ float4 operator/(
    const float4& lhs, const float4& rhs)
{
    return make_float4(
        lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}

template <> inline __host__ __device__ float4 make(const float v)
{
    return make_float4(v, v, v, v);
}

template <> inline __host__ __device__ float sum(const float4 v)
{
    return v.x + v.y + v.z + v.w;
}

template <>
inline __host__ __device__ float4 min(const float4 l, const float4 r)
{
    return make_float4(
        min(l.x, r.x), min(l.y, r.y), min(l.z, r.z), min(l.w, r.w));
}

template <>
inline __host__ __device__ float4 max(const float4 l, const float4 r)
{
    return make_float4(
        max(l.x, r.x), max(l.y, r.y), max(l.z, r.z), max(l.w, r.w));
}

template <> inline __host__ __device__ float min(const float4& v)
{
    return min(min(min(v.x, v.y), v.z), v.w);
}

template <> inline __host__ __device__ float max(const float4& v)
{
    return max(max(max(v.x, v.y), v.z), v.w);
}

template <> inline __host__ __device__ float4 abs(const float4& v)
{
    return make_float4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

template <> inline __host__ __device__ bool isnan(const float4& v)
{
    return (isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w));
}

template <> inline __host__ __device__ bool isinf(const float4& v)
{
    return (isinf(v.x) || isinf(v.y) || isinf(v.z) || isnan(v.w));
}

template <> inline __host__ __device__ bool isvalid(const float4& v)
{
    return (isvalid(v.x) && isvalid(v.y) && isvalid(v.z) && isvalid(v.w));
}

/***********************************************************************************
 * Specialization for int1
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const int1& lhs, const int1& rhs)
{
    return (lhs.x < rhs.x);
}

template <>
inline __host__ __device__ bool operator<=(const int1& lhs, const int1& rhs)
{
    return (lhs.x <= rhs.x);
}

template <>
inline __host__ __device__ bool operator>(const int1& lhs, const int1& rhs)
{
    return (lhs.x > rhs.x);
}

template <>
inline __host__ __device__ bool operator>=(const int1& lhs, const int1& rhs)
{
    return (lhs.x >= rhs.x);
}

template <>
inline __host__ __device__ bool operator==(const int1& lhs, const int1& rhs)
{
    return (lhs.x == rhs.x);
}

template <>
inline __host__ __device__ int1 operator+(const int1& lhs, const int1& rhs)
{
    return make_int1(lhs.x + rhs.x);
}

template <>
inline __host__ __device__ int1 operator-(const int1& lhs, const int1& rhs)
{
    return make_int1(lhs.x - rhs.x);
}

template <>
inline __host__ __device__ int1 operator*(const int1& lhs, const int1& rhs)
{
    return make_int1(lhs.x * rhs.x);
}

template <>
inline __host__ __device__ int1 operator/(const int1& lhs, const int1& rhs)
{
    return make_int1(lhs.x / rhs.x);
}

template <>
inline __host__ __device__ int1 operator*(const int1& lhs, const float& rhs)
{
    return make_int1(static_cast<int>(lhs.x * rhs));
}

template <> inline __host__ __device__ int1 make(const float v)
{
    return make_int1(static_cast<int>(v));
}

template <> inline __host__ __device__ float sum(const int1 v)
{
    return static_cast<float>(v.x);
}

template <> inline __host__ __device__ int1 min(const int1 l, const int1 r)
{
    return (l.x < r.x ? l : r);
}

template <> inline __host__ __device__ int1 max(const int1 l, const int1 r)
{
    return (l.x > r.x ? l : r);
}

template <> inline __host__ __device__ float min(const int1& v)
{
    return static_cast<float>(v.x);
}

template <> inline __host__ __device__ float max(const int1& v)
{
    return static_cast<float>(v.x);
}

template <> inline __host__ __device__ int1 abs(const int1& v)
{
    return make_int1(abs(v.x));
}

template <> inline __host__ __device__ bool isnan(const int1& v)
{
    return isnan(v.x);
}

template <> inline __host__ __device__ bool isinf(const int1& v)
{
    return isinf(v.x);
}

template <> inline __host__ __device__ bool isvalid(const int1& v)
{
    return isvalid(v.x);
}

/***********************************************************************************
 * Specialization for int2
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const int2& lhs, const int2& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y);
}

template <>
inline __host__ __device__ bool operator<=(const int2& lhs, const int2& rhs)
{
    return (lhs.x <= rhs.x && lhs.y <= rhs.y);
}

template <>
inline __host__ __device__ bool operator>(const int2& lhs, const int2& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y);
}

template <>
inline __host__ __device__ bool operator>=(const int2& lhs, const int2& rhs)
{
    return (lhs.x >= rhs.x && lhs.y >= rhs.y);
}

template <>
inline __host__ __device__ bool operator==(const int2& lhs, const int2& rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

template <>
inline __host__ __device__ int2 operator+(const int2& lhs, const int2& rhs)
{
    return make_int2(lhs.x + rhs.x, lhs.y + rhs.y);
}

template <>
inline __host__ __device__ int2 operator-(const int2& lhs, const int2& rhs)
{
    return make_int2(lhs.x - rhs.x, lhs.y - rhs.y);
}

template <>
inline __host__ __device__ int2 operator*(const int2& lhs, const int2& rhs)
{
    return make_int2(lhs.x * rhs.x, lhs.y * rhs.y);
}

template <>
inline __host__ __device__ int2 operator/(const int2& lhs, const int2& rhs)
{
    return make_int2(lhs.x / rhs.x, lhs.y / rhs.y);
}

template <>
inline __host__ __device__ int2 operator*(const int2& lhs, const float& rhs)
{
    return make_int2(
        static_cast<int>(lhs.x * rhs), static_cast<int>(lhs.y * rhs));
}

template <> inline __host__ __device__ int2 make(const float v)
{
    return make_int2(static_cast<int>(v), static_cast<int>(v));
}

template <> inline __host__ __device__ float sum(const int2 v)
{
    return static_cast<float>(v.x + v.y);
}

template <> inline __host__ __device__ int2 min(const int2 l, const int2 r)
{
    return make_int2(min(l.x, r.x), min(l.y, r.y));
}

template <> inline __host__ __device__ int2 max(const int2 l, const int2 r)
{
    return make_int2(max(l.x, r.x), max(l.y, r.y));
}

template <> inline __host__ __device__ float min(const int2& v)
{
    return static_cast<float>(min(v.x, v.y));
}

template <> inline __host__ __device__ float max(const int2& v)
{
    return static_cast<float>(max(v.x, v.y));
}

template <> inline __host__ __device__ int2 abs(const int2& v)
{
    return make_int2(abs(v.x), abs(v.y));
}

template <> inline __host__ __device__ bool isnan(const int2& v)
{
    return (isnan(v.x) || isnan(v.y));
}

template <> inline __host__ __device__ bool isinf(const int2& v)
{
    return (isinf(v.x) || isinf(v.y));
}

template <> inline __host__ __device__ bool isvalid(const int2& v)
{
    return (isvalid(v.x) && isvalid(v.y));
}

/***********************************************************************************
 * Specialization for int3
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const int3& lhs, const int3& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z);
}

template <>
inline __host__ __device__ bool operator<=(const int3& lhs, const int3& rhs)
{
    return (lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z);
}

template <>
inline __host__ __device__ bool operator>(const int3& lhs, const int3& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z);
}

template <>
inline __host__ __device__ bool operator>=(const int3& lhs, const int3& rhs)
{
    return (lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z);
}

template <>
inline __host__ __device__ bool operator==(const int3& lhs, const int3& rhs)
{
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

template <>
inline __host__ __device__ int3 operator+(const int3& lhs, const int3& rhs)
{
    return make_int3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template <>
inline __host__ __device__ int3 operator-(const int3& lhs, const int3& rhs)
{
    return make_int3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template <>
inline __host__ __device__ int3 operator*(const int3& lhs, const int3& rhs)
{
    return make_int3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

template <>
inline __host__ __device__ int3 operator/(const int3& lhs, const int3& rhs)
{
    return make_int3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

template <>
inline __host__ __device__ int3 operator*(const int3& lhs, const float& rhs)
{
    return make_int3(static_cast<int>(lhs.x * rhs),
        static_cast<int>(lhs.y * rhs), static_cast<int>(lhs.z * rhs));
}

template <> inline __host__ __device__ int3 make(const float v)
{
    return make_int3(
        static_cast<int>(v), static_cast<int>(v), static_cast<int>(v));
}

template <> inline __host__ __device__ float sum(const int3 v)
{
    return static_cast<float>(v.x + v.y + v.z);
}

template <> inline __host__ __device__ int3 min(const int3 l, const int3 r)
{
    return make_int3(min(l.x, r.x), min(l.y, r.y), min(l.z, r.z));
}

template <> inline __host__ __device__ int3 max(const int3 l, const int3 r)
{
    return make_int3(max(l.x, r.x), max(l.y, r.y), max(l.z, r.z));
}

template <> inline __host__ __device__ float min(const int3& v)
{
    return static_cast<float>(min(min(v.x, v.y), v.z));
}

template <> inline __host__ __device__ float max(const int3& v)
{
    return static_cast<float>(max(max(v.x, v.y), v.z));
}

template <> inline __host__ __device__ int3 abs(const int3& v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}

template <> inline __host__ __device__ bool isnan(const int3& v)
{
    return (isnan(v.x) || isnan(v.y) || isnan(v.z));
}

template <> inline __host__ __device__ bool isinf(const int3& v)
{
    return (isinf(v.x) || isinf(v.y) || isinf(v.z));
}

template <> inline __host__ __device__ bool isvalid(const int3& v)
{
    return (isvalid(v.x) && isvalid(v.y) && isvalid(v.z));
}

/***********************************************************************************
 * Specialization for int4
 ***********************************************************************************/

template <>
inline __host__ __device__ bool operator<(const int4& lhs, const int4& rhs)
{
    return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w);
}

template <>
inline __host__ __device__ bool operator<=(const int4& lhs, const int4& rhs)
{
    return (
        lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w);
}

template <>
inline __host__ __device__ bool operator>(const int4& lhs, const int4& rhs)
{
    return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w);
}

template <>
inline __host__ __device__ bool operator>=(const int4& lhs, const int4& rhs)
{
    return (
        lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w);
}

template <>
inline __host__ __device__ bool operator==(const int4& lhs, const int4& rhs)
{
    return (
        lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}

template <>
inline __host__ __device__ int4 operator+(const int4& lhs, const int4& rhs)
{
    return make_int4(
        lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

template <>
inline __host__ __device__ int4 operator-(const int4& lhs, const int4& rhs)
{
    return make_int4(
        lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

template <>
inline __host__ __device__ int4 operator*(const int4& lhs, const int4& rhs)
{
    return make_int4(
        lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}

template <>
inline __host__ __device__ int4 operator/(const int4& lhs, const int4& rhs)
{
    return make_int4(
        lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}

template <>
inline __host__ __device__ int4 operator*(const int4& lhs, const float& rhs)
{
    return make_int4(static_cast<int>(lhs.x * rhs),
        static_cast<int>(lhs.y * rhs), static_cast<int>(lhs.z * rhs),
        static_cast<int>(lhs.w * rhs));
}

template <> inline __host__ __device__ int4 make(const float v)
{
    return make_int4(static_cast<int>(v), static_cast<int>(v),
        static_cast<int>(v), static_cast<int>(v));
}

template <> inline __host__ __device__ float sum(const int4 v)
{
    return static_cast<float>(v.x + v.y + v.z + v.w);
}

template <> inline __host__ __device__ int4 min(const int4 l, const int4 r)
{
    return make_int4(
        min(l.x, r.x), min(l.y, r.y), min(l.z, r.z), min(l.w, r.w));
}

template <> inline __host__ __device__ int4 max(const int4 l, const int4 r)
{
    return make_int4(
        max(l.x, r.x), max(l.y, r.y), max(l.z, r.z), max(l.w, r.w));
}

template <> inline __host__ __device__ float min(const int4& v)
{
    return static_cast<float>(min(min(min(v.x, v.y), v.z), v.w));
}

template <> inline __host__ __device__ float max(const int4& v)
{
    return static_cast<float>(max(max(max(v.x, v.y), v.z), v.w));
}

template <> inline __host__ __device__ int4 abs(const int4& v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

template <> inline __host__ __device__ bool isnan(const int4& v)
{
    return (isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w));
}

template <> inline __host__ __device__ bool isinf(const int4& v)
{
    return (isinf(v.x) || isinf(v.y) || isinf(v.z) || isnan(v.w));
}

template <> inline __host__ __device__ bool isvalid(const int4& v)
{
    return (isvalid(v.x) && isvalid(v.y) && isvalid(v.z) && isvalid(v.w));
}

/**
 * Easy limits for custom types
 */
template <typename T> struct Limits
{
    static __host__ T max()
    {
        if (is_float<T>())
            return make<T>(std::numeric_limits<float>::max());
        else if (is_int<T>())
            return make<T>(std::numeric_limits<int>::max());
        else if (is_uchar<T>())
            return make<T>(std::numeric_limits<uchar>::max());
        else
            throw std::runtime_error("Invalid type!");
    }

    static __host__ T min()
    {
        if (is_float<T>())
            return make<T>(std::numeric_limits<float>::min());
        else if (is_int<T>())
            return make<T>(std::numeric_limits<int>::min());
        else if (is_uchar<T>())
            return make<T>(std::numeric_limits<uchar>::min());
        else
            throw std::runtime_error("Invalid type!");
    }
};

} // image