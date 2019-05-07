#include <gtest/gtest.h>

#include "cuimage/cuda/type.h"

using namespace cuimage;


TEST(FloatTypeTest, is_float_with_channels)
{
    EXPECT_TRUE(is_float<float>());
    EXPECT_TRUE(is_float<float1>());
    EXPECT_TRUE(is_float<float2>());
    EXPECT_TRUE(is_float<float3>());
    EXPECT_TRUE(is_float<float4>());

    EXPECT_EQ(1, channels<float>());
    EXPECT_EQ(1, channels<float1>());
    EXPECT_EQ(2, channels<float2>());
    EXPECT_EQ(3, channels<float3>());
    EXPECT_EQ(4, channels<float4>());
}

TEST(FloatTypeTest, is_not_other_type)
{
    EXPECT_FALSE(is_uchar<float>());
    EXPECT_FALSE(is_uchar<float1>());
    EXPECT_FALSE(is_uchar<float2>());
    EXPECT_FALSE(is_uchar<float3>());
    EXPECT_FALSE(is_uchar<float4>());

    EXPECT_FALSE(is_int<float>());
    EXPECT_FALSE(is_int<float1>());
    EXPECT_FALSE(is_int<float2>());
    EXPECT_FALSE(is_int<float3>());
    EXPECT_FALSE(is_int<float4>());
}

TEST(UcharTypeTest, is_uchar_with_channels)
{
    EXPECT_TRUE(is_uchar<uchar>());
    EXPECT_TRUE(is_uchar<uchar1>());
    EXPECT_TRUE(is_uchar<uchar2>());
    EXPECT_TRUE(is_uchar<uchar3>());
    EXPECT_TRUE(is_uchar<uchar4>());

    EXPECT_EQ(1, channels<uchar>());
    EXPECT_EQ(1, channels<uchar1>());
    EXPECT_EQ(2, channels<uchar2>());
    EXPECT_EQ(3, channels<uchar3>());
    EXPECT_EQ(4, channels<uchar4>());
}

TEST(UcharTypeTest, is_not_other_type)
{
    EXPECT_FALSE(is_float<uchar>());
    EXPECT_FALSE(is_float<uchar1>());
    EXPECT_FALSE(is_float<uchar2>());
    EXPECT_FALSE(is_float<uchar3>());
    EXPECT_FALSE(is_float<uchar4>());

    EXPECT_FALSE(is_int<uchar>());
    EXPECT_FALSE(is_int<uchar1>());
    EXPECT_FALSE(is_int<uchar2>());
    EXPECT_FALSE(is_int<uchar3>());
    EXPECT_FALSE(is_int<uchar4>());
}


TEST(IntTypeTest, is_int_with_channels)
{
    EXPECT_TRUE(is_int<int>());
    EXPECT_TRUE(is_int<int1>());
    EXPECT_TRUE(is_int<int2>());
    EXPECT_TRUE(is_int<int3>());
    EXPECT_TRUE(is_int<int4>());

    EXPECT_EQ(1, channels<int>());
    EXPECT_EQ(1, channels<int1>());
    EXPECT_EQ(2, channels<int2>());
    EXPECT_EQ(3, channels<int3>());
    EXPECT_EQ(4, channels<int4>());
}

TEST(IntTypeTest, is_not_other_type)
{
    EXPECT_FALSE(is_float<int>());
    EXPECT_FALSE(is_float<int1>());
    EXPECT_FALSE(is_float<int2>());
    EXPECT_FALSE(is_float<int3>());
    EXPECT_FALSE(is_float<int4>());

    EXPECT_FALSE(is_uchar<int>());
    EXPECT_FALSE(is_uchar<int1>());
    EXPECT_FALSE(is_uchar<int2>());
    EXPECT_FALSE(is_uchar<int3>());
    EXPECT_FALSE(is_uchar<int4>());
}