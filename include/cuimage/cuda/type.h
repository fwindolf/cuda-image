#pragma once

#include <ostream>
#include <vector_types.h>
#include <type_traits>

/**
 * X-Macros for tedious template instantiations
 */
#define FOR_EACH_TYPE(func, ...)            \
    func(float,  __VA_ARGS__)               \
    func(float1, __VA_ARGS__)               \
    func(float2, __VA_ARGS__)               \
    func(float3, __VA_ARGS__)               \
    func(float4, __VA_ARGS__)               \
    func(uchar,  __VA_ARGS__)               \
    func(uchar1, __VA_ARGS__)               \
    func(uchar2, __VA_ARGS__)               \
    func(uchar3, __VA_ARGS__)               \
    func(uchar4, __VA_ARGS__)               \
    func(int,    __VA_ARGS__)               \
    func(int1,   __VA_ARGS__)               \
    func(int2,   __VA_ARGS__)               \
    func(int3,   __VA_ARGS__)               \
    func(int4,   __VA_ARGS__)   

#define FOR_EACH_FLOAT_TYPE(func, ...)      \
    func(float, __VA_ARGS__)                \
    func(float1, __VA_ARGS__)               \
    func(float2, __VA_ARGS__)               \
    func(float3, __VA_ARGS__)               \
    func(float4, __VA_ARGS__)               \

#define FOR_EACH_UCHAR_TYPE(func, ...)      \
    func(uchar, __VA_ARGS__)                \
    func(uchar1, __VA_ARGS__)               \
    func(uchar2, __VA_ARGS__)               \
    func(uchar3, __VA_ARGS__)               \
    func(uchar4, __VA_ARGS__)               \

#define FOR_EACH_INT_TYPE(func, ...)        \
    func(int, __VA_ARGS__)                  \
    func(int1, __VA_ARGS__)                 \
    func(int2, __VA_ARGS__)                 \
    func(int3, __VA_ARGS__)                 \
    func(int4, __VA_ARGS__)                 \

#define FOR_EACH_1CHANNEL_TYPE(func, ...)   \
    func(float, __VA_ARGS__)                \
    func(float1, __VA_ARGS__)               \
    func(uchar, __VA_ARGS__)                \
    func(uchar1, __VA_ARGS__)               \
    func(int, __VA_ARGS__)                  \
    func(int1, __VA_ARGS__)                 \
    
#define FOR_EACH_2CHANNEL_TYPE(func, ...)   \
    func(float2, __VA_ARGS__)               \
    func(uchar2, __VA_ARGS__)               \
    func(int2, __VA_ARGS__)                 \

#define FOR_EACH_3CHANNEL_TYPE(func, ...)   \
    func(float3, __VA_ARGS__)               \
    func(uchar3, __VA_ARGS__)               \
    func(int3, __VA_ARGS__)                 \

#define FOR_EACH_4CHANNEL_TYPE(func, ...)   \
    func(float4, __VA_ARGS__)               \
    func(uchar4, __VA_ARGS__)               \
    func(int4, __VA_ARGS__)                 \


namespace cuimage
{

typedef unsigned char uchar;

/***********************************************************************************
 * Enableing templates only if they are vector type
 ***********************************************************************************/

template <typename T>
struct is_vector_type : std::false_type {};

template<> struct is_vector_type<float1> : std::true_type {};
template<> struct is_vector_type<float2> : std::true_type {};
template<> struct is_vector_type<float3> : std::true_type {};
template<> struct is_vector_type<float4> : std::true_type {};

template<> struct is_vector_type<uchar1> : std::true_type {};
template<> struct is_vector_type<uchar2> : std::true_type {};
template<> struct is_vector_type<uchar3> : std::true_type {};
template<> struct is_vector_type<uchar4> : std::true_type {};

template<> struct is_vector_type<int1  > : std::true_type {};
template<> struct is_vector_type<int2  > : std::true_type {};
template<> struct is_vector_type<int3  > : std::true_type {};
template<> struct is_vector_type<int4  > : std::true_type {};

/***********************************************************************************
 * Vector type base type
 ***********************************************************************************/

template <typename T>
struct is_float_type : std::false_type {};

template<> struct is_float_type<float > : std::true_type {};
template<> struct is_float_type<float1> : std::true_type {};
template<> struct is_float_type<float2> : std::true_type {};
template<> struct is_float_type<float3> : std::true_type {};
template<> struct is_float_type<float4> : std::true_type {};

template <typename T>
struct is_uchar_type : std::false_type {};

template<> struct is_uchar_type<uchar > : std::true_type {};
template<> struct is_uchar_type<uchar1> : std::true_type {};
template<> struct is_uchar_type<uchar2> : std::true_type {};
template<> struct is_uchar_type<uchar3> : std::true_type {};
template<> struct is_uchar_type<uchar4> : std::true_type {};

template <typename T>
struct is_int_type : std::false_type {};

template<> struct is_int_type<int > : std::true_type {};
template<> struct is_int_type<int1> : std::true_type {};
template<> struct is_int_type<int2> : std::true_type {};
template<> struct is_int_type<int3> : std::true_type {};
template<> struct is_int_type<int4> : std::true_type {};

/***********************************************************************************
 * Vector type number of channels
 ***********************************************************************************/

template <typename T>
struct has_4_channels : std::false_type {};

template <> struct has_4_channels<float4> : std::true_type {};
template <> struct has_4_channels<uchar4> : std::true_type {};
template <> struct has_4_channels<int4  > : std::true_type {};

template <typename T>
struct has_3_channels : std::false_type {};

template <> struct has_3_channels<float3> : std::true_type {};
template <> struct has_3_channels<uchar3> : std::true_type {};
template <> struct has_3_channels<int3  > : std::true_type {};

template <typename T>
struct has_2_channels : std::false_type {};

template <> struct has_2_channels<float2> : std::true_type {};
template <> struct has_2_channels<uchar2> : std::true_type {};
template <> struct has_2_channels<int2  > : std::true_type {};

template <typename T>
struct has_1_channels : std::false_type {};

template <> struct has_1_channels<float1> : std::true_type {};
template <> struct has_1_channels<uchar1> : std::true_type {};
template <> struct has_1_channels<int1  > : std::true_type {};

template <typename T>
struct has_0_channels : std::false_type {};

template <> struct has_0_channels<float > : std::true_type {};
template <> struct has_0_channels<uchar > : std::true_type {};
template <> struct has_0_channels<int   > : std::true_type {};

/***********************************************************************************
 * Conversion between data types
 ***********************************************************************************/

template <typename T>
bool is_float();

template <> inline bool is_float<float >() { return true;  }
template <> inline bool is_float<float1>() { return true;  }
template <> inline bool is_float<float2>() { return true;  }
template <> inline bool is_float<float3>() { return true;  }
template <> inline bool is_float<float4>() { return true;  }

template <> inline bool is_float<uchar >() { return false; }
template <> inline bool is_float<uchar1>() { return false; }
template <> inline bool is_float<uchar2>() { return false; }
template <> inline bool is_float<uchar3>() { return false; }
template <> inline bool is_float<uchar4>() { return false; }

template <> inline bool is_float<int   >() { return false; }
template <> inline bool is_float<int1  >() { return false; }
template <> inline bool is_float<int2  >() { return false; }
template <> inline bool is_float<int3  >() { return false; }
template <> inline bool is_float<int4  >() { return false; }

template <typename T>
bool is_uchar();

template <> inline bool is_uchar<float >() { return false; }
template <> inline bool is_uchar<float1>() { return false; }
template <> inline bool is_uchar<float2>() { return false; }
template <> inline bool is_uchar<float3>() { return false; }
template <> inline bool is_uchar<float4>() { return false; }

template <> inline bool is_uchar<uchar >() { return true;  }
template <> inline bool is_uchar<uchar1>() { return true;  }
template <> inline bool is_uchar<uchar2>() { return true;  }
template <> inline bool is_uchar<uchar3>() { return true;  }
template <> inline bool is_uchar<uchar4>() { return true;  }

template <> inline bool is_uchar<int   >() { return false; }
template <> inline bool is_uchar<int1  >() { return false; }
template <> inline bool is_uchar<int2  >() { return false; }
template <> inline bool is_uchar<int3  >() { return false; }
template <> inline bool is_uchar<int4  >() { return false; }

template <typename T>
bool is_int();

template <> inline bool is_int<float >() { return false; }
template <> inline bool is_int<float1>() { return false; }
template <> inline bool is_int<float2>() { return false; }
template <> inline bool is_int<float3>() { return false; }
template <> inline bool is_int<float4>() { return false; }

template <> inline bool is_int<uchar >() { return false; }
template <> inline bool is_int<uchar1>() { return false; }
template <> inline bool is_int<uchar2>() { return false; }
template <> inline bool is_int<uchar3>() { return false; }
template <> inline bool is_int<uchar4>() { return false; }

template <> inline bool is_int<int   >() { return true;  }
template <> inline bool is_int<int1  >() { return true;  }
template <> inline bool is_int<int2  >() { return true;  }
template <> inline bool is_int<int3  >() { return true;  }
template <> inline bool is_int<int4  >() { return true;  }

template <typename T, typename TO>
bool is_same_base()
{
    return (is_float<T>() && is_float<TO>() ||
            is_uchar<T>() && is_uchar<TO>() ||
            is_int<T>() && is_int<TO>());
}

template <typename T>
size_t channels();

template <> inline size_t channels<float >() { return 1; }
template <> inline size_t channels<float1>() { return 1; }
template <> inline size_t channels<float2>() { return 2; }
template <> inline size_t channels<float3>() { return 3; }
template <> inline size_t channels<float4>() { return 4; }

template <> inline size_t channels<uchar >() { return 1; }
template <> inline size_t channels<uchar1>() { return 1; }
template <> inline size_t channels<uchar2>() { return 2; }
template <> inline size_t channels<uchar3>() { return 3; }
template <> inline size_t channels<uchar4>() { return 4; }

template <> inline size_t channels<int   >() { return 1; }
template <> inline size_t channels<int1  >() { return 1; }
template <> inline size_t channels<int2  >() { return 2; }
template <> inline size_t channels<int3  >() { return 3; }
template <> inline size_t channels<int4  >() { return 4; }

/**
 * Ostream operator overload
 */
template <typename T, typename std::enable_if<is_vector_type<T>::value && has_1_channels<T>::value, T>::type* = nullptr>
std::ostream& operator<<(std::ostream& os, const T& v)
{
    os << v.x;
    return os;
}

template <typename T, typename std::enable_if<is_vector_type<T>::value && has_2_channels<T>::value, T>::type* = nullptr>
std::ostream& operator<<(std::ostream& os, const T& v)
{
    os << "(" << v.x << "," << v.y << ")";
    return os;
}

template <typename T, typename std::enable_if<is_vector_type<T>::value && has_3_channels<T>::value, T>::type* = nullptr>
std::ostream& operator<<(std::ostream& os, const T& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << ")";
    return os;
}

template <typename T, typename std::enable_if<is_vector_type<T>::value && has_4_channels<T>::value, T>::type* = nullptr >
std::ostream& operator<<(std::ostream& os, const T& v)
{
    os << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
    return os;
}

/**
 * New template aliases in C++14 only
 *

template <typename T1, typename T2>
using IsSameBase = typename 
    std::enable_if<
        (is_float_type<T1>::value && is_float_type<T2>::value) || 
        (is_uchar_type<T1>::value && is_uchar_type<T2>::value) || 
        (is_int_type<T1>::value   && is_int_type<T2>::value), T1>::type* = nullptr;

template <typename T1, typename T2>
using IsSameChannels = typename 
    std::enable_if<
        (has_4_channels<T>::value && has_4_channels<TO>::value) ||
        (has_3_channels<T>::value && has_3_channels<TO>::value) ||
        (has_2_channels<T>::value && has_2_channels<TO>::value) ||
        (has_1_channels<T>::value && has_1_channels<TO>::value) ||
        (has_0_channels<T>::value && has_0_channels<TO>::value), T1>::type* = nullptr;

template <typename T1, typename T2>
using IsSame = typename 
    std::enable_if<(
        (is_float_type<T1>::value && is_float_type<T2>::value) || 
        (is_uchar_type<T1>::value && is_uchar_type<T2>::value) || 
        (is_int_type<T1>::value   && is_int_type<T2>::value)) && (
        (has_4_channels<T>::value && has_4_channels<TO>::value) ||
        (has_3_channels<T>::value && has_3_channels<TO>::value) ||
        (has_2_channels<T>::value && has_2_channels<TO>::value) ||
        (has_1_channels<T>::value && has_1_channels<TO>::value) ||
        (has_0_channels<T>::value && has_0_channels<TO>::value)), T1::type* = nullptr;
*/

} // namespace cuimage