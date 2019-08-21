#include "cuimage/cuda/interpolation.h"

#include <gtest/gtest.h>

using namespace cuimage;

TEST(InterpolationLinearTest, returns_interpolated_value_in_range)
{
    float v1 = 0.f;
    float v2 = 1.f;

    int pm = 0, pp = 1;

    float px[11] = {.0f, .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, 1.f};

    // interpolated value is on linear function between 0 - 1
    for (int i = 0; i < 11; i++)
        EXPECT_FLOAT_EQ(
            px[i], d_interpolate_linear(v1, v2, 1, 1, pm, pp, px[i]));
}

TEST(InterpolationLinearTest, works_for_large_range)
{
    float v1 = 0.f;
    float v2 = 1.f;

    int pm = 0, pp = 100;

    float px[6] = {0.f, 20.f, 40.f, 60.f, 80.f, 100.f};

    // interpolated value is on linear function between 0 - 1
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(
            px[i] / 100.f, d_interpolate_linear(v1, v2, 1, 1, pm, pp, px[i]));
}

TEST(InterpolationLinearTest, works_for_uchar)
{
    uchar v1 = 1;
    uchar v2 = 5;

    int pm = 0, pp = 1;
    float px = 0.3f;

    // 0.7 + 1.5 = 2.2 -> 2
    uchar res = 2;
    EXPECT_EQ(res, d_interpolate_linear(v1, v2, 1, 1, pm, pp, px));
}

TEST(InterpolationLinearTest, works_for_float_vector_types)
{
    float3 v1 = make_float3(1.f, 0.f, -1.f);
    float3 v2 = make_float3(0.f, 1.f, 0.f);

    int pm = 0, pp = 1;
    float px = 0.5f;

    float3 res = d_interpolate_linear(v1, v2, 1, 1, pm, pp, px);
    float3 exp = make_float3(0.5f, 0.5f, -0.5f);

    EXPECT_FLOAT_EQ(exp.x, res.x);
    EXPECT_FLOAT_EQ(exp.y, res.y);
    EXPECT_FLOAT_EQ(exp.z, res.z);
}

TEST(InterpolationLinearTest, works_for_uchar_vector_types)
{
    uchar4 v1 = make_uchar4(0, 50, 100, 200);
    uchar4 v2 = make_uchar4(50, 0, 150, 150);

    int pm = 0, pp = 1;
    float px = 0.5f;

    uchar4 res = d_interpolate_linear(v1, v2, 1, 1, pm, pp, px);
    uchar4 exp = make_uchar4(25, 25, 125, 175);

    EXPECT_FLOAT_EQ(exp.x, res.x);
    EXPECT_FLOAT_EQ(exp.y, res.y);
    EXPECT_FLOAT_EQ(exp.z, res.z);
    EXPECT_FLOAT_EQ(exp.w, res.w);
}

TEST(InterpolationLinearTest, works_for_int_vector_types)
{
    int2 v1 = make_int2(0, 10);
    int2 v2 = make_int2(50, 0);

    int pm = 0, pp = 1;
    float px = 0.5f;

    int2 res = d_interpolate_linear(v1, v2, 1, 1, pm, pp, px);
    int2 exp = make_int2(25, 5);

    EXPECT_FLOAT_EQ(exp.x, res.x);
    EXPECT_FLOAT_EQ(exp.y, res.y);
}