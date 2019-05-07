#include <gtest/gtest.h>

#include "cuimage/cuda/conversion.h"

using namespace cuimage;


TEST(RgbaToGrayTest, works_float)
{
    float4 c1 = make_float4(1.f, 1.f, 1.f, 1.f);
    float4 c2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c3 = make_float4(0.3f, 0.2f, 0.4f, 0.f);
    float4 c4 = make_float4(0.3f, 0.2f, 0.4f, 1.f);

    EXPECT_FLOAT_EQ(1.f, rgba2gray(c1));
    EXPECT_FLOAT_EQ(0.f, rgba2gray(c2));
    EXPECT_FLOAT_EQ(0.f, rgba2gray(c3));
    EXPECT_GT(rgba2gray(c4), 0.f);
    EXPECT_LT(rgba2gray(c4), 1.f);
}

TEST(RgbaToGrayTest, works_uchar)
{
    uchar4 c1 = make_uchar4(255, 255, 255, 255);
    uchar4 c2 = make_uchar4(0, 0, 0, 0);
    uchar4 c3 = make_uchar4(100, 200, 150, 0);
    uchar4 c4 = make_uchar4(100, 200, 150, 255);

    EXPECT_FLOAT_EQ(255, rgba2gray(c1));
    EXPECT_FLOAT_EQ(0, rgba2gray(c2));
    EXPECT_FLOAT_EQ(0, rgba2gray(c3));
    EXPECT_GT(rgba2gray(c4), 0);
    EXPECT_LT(rgba2gray(c4), 255);
}

TEST(RgbaToGrayTest, works_int)
{
    int4 c1 = make_int4(255, 255, 255, 255);
    int4 c2 = make_int4(0, 0, 0, 0);
    int4 c3 = make_int4(100, 200, 150, 0);
    int4 c4 = make_int4(100, 200, 150, 255);

    EXPECT_FLOAT_EQ(255, rgba2gray(c1));
    EXPECT_FLOAT_EQ(0, rgba2gray(c2));
    EXPECT_FLOAT_EQ(0, rgba2gray(c3));
    EXPECT_GT(rgba2gray(c4), 0);
    EXPECT_LT(rgba2gray(c4), 255);
}

TEST(RgbToGrayTest, works_float)
{
    float3 c1 = make_float3(1.f, 1.f, 1.f);
    float3 c2 = make_float3(0.f, 0.f, 0.f);
    float3 c3 = make_float3(0.3f, 0.2f, 0.5f);

    EXPECT_FLOAT_EQ(1.f, rgb2gray(c1));
    EXPECT_FLOAT_EQ(0.f, rgb2gray(c2));
    EXPECT_GT(rgb2gray(c3), 0.f);
    EXPECT_LT(rgb2gray(c3), 1.f);
}

TEST(RgbToGrayTest, works_uchar)
{
    uchar3 c1 = make_uchar3(255, 255, 255);
    uchar3 c2 = make_uchar3(0, 0, 0);
    uchar3 c3 = make_uchar3(100, 200, 150);
    
    EXPECT_FLOAT_EQ(255, rgb2gray(c1));
    EXPECT_FLOAT_EQ(0, rgb2gray(c2));
    EXPECT_GT(rgb2gray(c3), 0);
    EXPECT_LT(rgb2gray(c3), 255);
}

TEST(RgbToGrayTest, works_int)
{
    int3 c1 = make_int3(255, 255, 255);
    int3 c2 = make_int3(0, 0, 0);
    int3 c3 = make_int3(100, 200, 150);

    EXPECT_FLOAT_EQ(255, rgb2gray(c1));
    EXPECT_FLOAT_EQ(0, rgb2gray(c2));
    EXPECT_GT(rgb2gray(c3), 0);
    EXPECT_LT(rgb2gray(c3), 255);
}


TEST(RgbaToRgbTest, works_float)
{
    float4 c1 = make_float4(1.f, 1.f, 1.f, 1.f);
    float4 c2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c3 = make_float4(0.3f, 0.2f, 0.4f, 0.f);
    float4 c4 = make_float4(0.3f, 0.2f, 0.4f, 1.f);

    float3 r1 = rgba2rgb<float3>(c1);
    EXPECT_FLOAT_EQ(1.f, r1.x);
    EXPECT_FLOAT_EQ(1.f, r1.y);
    EXPECT_FLOAT_EQ(1.f, r1.z);

    float3 r2 = rgba2rgb<float3>(c2);
    EXPECT_FLOAT_EQ(0.f, r2.x);
    EXPECT_FLOAT_EQ(0.f, r2.y);
    EXPECT_FLOAT_EQ(0.f, r2.z);

    float3 r3 = rgba2rgb<float3>(c3);
    EXPECT_FLOAT_EQ(0.f, r3.x);
    EXPECT_FLOAT_EQ(0.f, r3.y);
    EXPECT_FLOAT_EQ(0.f, r3.z);

    float3 r4 = rgba2rgb<float3>(c4);
    EXPECT_FLOAT_EQ(c4.x, r4.x);
    EXPECT_FLOAT_EQ(c4.y, r4.y);
    EXPECT_FLOAT_EQ(c4.z, r4.z);
}


TEST(RgbaToRgbTest, works_uchar)
{
    uchar4 c1 = make_uchar4(255, 255, 255, 255);
    uchar4 c2 = make_uchar4(0, 0, 0, 0);
    uchar4 c3 = make_uchar4(100, 200, 150, 0);
    uchar4 c4 = make_uchar4(100, 200, 150, 255);

    uchar3 r1 = rgba2rgb<uchar3>(c1);
    EXPECT_EQ(255, r1.x);
    EXPECT_EQ(255, r1.y);
    EXPECT_EQ(255, r1.z);

    uchar3 r2 = rgba2rgb<uchar3>(c2);
    EXPECT_EQ(0, r2.x);
    EXPECT_EQ(0, r2.y);
    EXPECT_EQ(0, r2.z);

    uchar3 r3 = rgba2rgb<uchar3>(c3);
    EXPECT_EQ(0, r3.x);
    EXPECT_EQ(0, r3.y);
    EXPECT_EQ(0, r3.z);

    uchar3 r4 = rgba2rgb<uchar3>(c4);
    EXPECT_EQ(c4.x, r4.x);
    EXPECT_EQ(c4.y, r4.y);
    EXPECT_EQ(c4.z, r4.z);
}

TEST(RgbaToRgbTest, works_int)
{
    int4 c1 = make_int4(255, 255, 255, 255);
    int4 c2 = make_int4(0, 0, 0, 0);
    int4 c3 = make_int4(100, 200, 150, 0);
    int4 c4 = make_int4(100, 200, 150, 255);

    int3 r1 = rgba2rgb<int3>(c1);
    EXPECT_EQ(255, r1.x);
    EXPECT_EQ(255, r1.y);
    EXPECT_EQ(255, r1.z);

    int3 r2 = rgba2rgb<int3>(c2);
    EXPECT_EQ(0, r2.x);
    EXPECT_EQ(0, r2.y);
    EXPECT_EQ(0, r2.z);

    int3 r3 = rgba2rgb<int3>(c3);
    EXPECT_EQ(0, r3.x);
    EXPECT_EQ(0, r3.y);
    EXPECT_EQ(0, r3.z);

    int3 r4 = rgba2rgb<int3>(c4);
    EXPECT_EQ(c4.x, r4.x);
    EXPECT_EQ(c4.y, r4.y);
    EXPECT_EQ(c4.z, r4.z);
}

TEST(RgbToRgbaTest, works_float)
{
    float3 c1 = make_float3(1.f, 1.f, 1.f);
    float3 c2 = make_float3(0.f, 0.f, 0.f);
    float3 c3 = make_float3(0.3f, 0.2f, 0.4f);

    float4 r1 = rgb2rgba<float4>(c1);
    EXPECT_FLOAT_EQ(1.f, r1.x);
    EXPECT_FLOAT_EQ(1.f, r1.y);
    EXPECT_FLOAT_EQ(1.f, r1.z);
    EXPECT_FLOAT_EQ(1.f, r1.w);

    float4 r2 = rgb2rgba<float4>(c2);
    EXPECT_FLOAT_EQ(0.f, r2.x);
    EXPECT_FLOAT_EQ(0.f, r2.y);
    EXPECT_FLOAT_EQ(0.f, r2.z);
    EXPECT_FLOAT_EQ(1.f, r2.w);

    float4 r3 = rgb2rgba<float4>(c3);
    EXPECT_FLOAT_EQ(c3.x, r3.x);
    EXPECT_FLOAT_EQ(c3.y, r3.y);
    EXPECT_FLOAT_EQ(c3.z, r3.z);
    EXPECT_FLOAT_EQ(1.f, r3.w);
}



TEST(RgbToRgbaTest, works_uchar)
{
    uchar3 c1 = make_uchar3(255, 255, 255);
    uchar3 c2 = make_uchar3(0, 0, 0);
    uchar3 c3 = make_uchar3(100, 200, 150);

    uchar4 r1 = rgb2rgba<uchar4>(c1);
    EXPECT_EQ(255, r1.x);
    EXPECT_EQ(255, r1.y);
    EXPECT_EQ(255, r1.z);
    EXPECT_EQ(255, r1.w);

    uchar4 r2 = rgb2rgba<uchar4>(c2);
    EXPECT_EQ(0, r2.x);
    EXPECT_EQ(0, r2.y);
    EXPECT_EQ(0, r2.z);
    EXPECT_EQ(255, r1.w);

    uchar4 r3 = rgb2rgba<uchar4>(c3);
    EXPECT_EQ(c3.x, r3.x);
    EXPECT_EQ(c3.y, r3.y);
    EXPECT_EQ(c3.z, r3.z);
    EXPECT_EQ(255, r1.w);
}

TEST(RgbToRgbaTest, works_int)
{
    int3 c1 = make_int3(255, 255, 255);
    int3 c2 = make_int3(0, 0, 0);
    int3 c3 = make_int3(100, 200, 150);

    int4 r1 = rgb2rgba<int4>(c1);
    EXPECT_EQ(255, r1.x);
    EXPECT_EQ(255, r1.y);
    EXPECT_EQ(255, r1.z);
    EXPECT_EQ(255, r1.w);

    int4 r2 = rgb2rgba<int4>(c2);
    EXPECT_EQ(0, r2.x);
    EXPECT_EQ(0, r2.y);
    EXPECT_EQ(0, r2.z);
    EXPECT_EQ(255, r1.w);

    int4 r3 = rgb2rgba<int4>(c3);
    EXPECT_EQ(c3.x, r3.x);
    EXPECT_EQ(c3.y, r3.y);
    EXPECT_EQ(c3.z, r3.z);
    EXPECT_EQ(255, r1.w);
}

TEST(GrayToRgbTest, works_float)
{
    float v1 = 1.f, v2 = 0.f, v3 = 0.3f;

    float3 r1 = gray2rgb<float3>(v1);
    EXPECT_FLOAT_EQ(v1, r1.x);
    EXPECT_FLOAT_EQ(v1, r1.y);
    EXPECT_FLOAT_EQ(v1, r1.z);

    float3 r2 = gray2rgb<float3>(v2);
    EXPECT_FLOAT_EQ(v2, r2.x);
    EXPECT_FLOAT_EQ(v2, r2.y);
    EXPECT_FLOAT_EQ(v2, r2.z);

    float3 r3 = gray2rgb<float3>(v3);
    EXPECT_FLOAT_EQ(v3, r3.x);
    EXPECT_FLOAT_EQ(v3, r3.y);
    EXPECT_FLOAT_EQ(v3, r3.z);
}

TEST(GrayToRgbTest, works_uchar)
{
    float v1 = 255.f, v2 = 0.f, v3 = 30.f;
    uchar u1 = 255, u2 = 0, u3 = 30;

    uchar3 r1 = gray2rgb<uchar3>(v1);
    EXPECT_EQ(u1, r1.x);
    EXPECT_EQ(u1, r1.y);
    EXPECT_EQ(u1, r1.z);

    uchar3 r2 = gray2rgb<uchar3>(v2);
    EXPECT_EQ(u2, r2.x);
    EXPECT_EQ(u2, r2.y);
    EXPECT_EQ(u2, r2.z);

    uchar3 r3 = gray2rgb<uchar3>(v3);
    EXPECT_EQ(u3, r3.x);
    EXPECT_EQ(u3, r3.y);
    EXPECT_EQ(u3, r3.z);
}

TEST(GrayToRgbTest, works_int)
{
    float v1 = 255.f, v2 = 0.f, v3 = 30.f;
    int u1 = 255, u2 = 0, u3 = 30;

    int3 r1 = gray2rgb<int3>(v1);
    EXPECT_EQ(u1, r1.x);
    EXPECT_EQ(u1, r1.y);
    EXPECT_EQ(u1, r1.z);

    int3 r2 = gray2rgb<int3>(v2);
    EXPECT_EQ(u2, r2.x);
    EXPECT_EQ(u2, r2.y);
    EXPECT_EQ(u2, r2.z);

    int3 r3 = gray2rgb<int3>(v3);
    EXPECT_EQ(u3, r3.x);
    EXPECT_EQ(u3, r3.y);
    EXPECT_EQ(u3, r3.z);
}

TEST(GrayToRgbaTest, works_float)
{
    float v1 = 1.f, v2 = 0.f, v3 = 0.3f;

    float4 r1 = gray2rgba<float4>(v1);
    EXPECT_FLOAT_EQ(v1, r1.x);
    EXPECT_FLOAT_EQ(v1, r1.y);
    EXPECT_FLOAT_EQ(v1, r1.z);
    EXPECT_FLOAT_EQ(1.f, r1.w);

    float4 r2 = gray2rgba<float4>(v2);
    EXPECT_FLOAT_EQ(v2, r2.x);
    EXPECT_FLOAT_EQ(v2, r2.y);
    EXPECT_FLOAT_EQ(v2, r2.z);
    EXPECT_FLOAT_EQ(1.f, r2.w);

    float4 r3 = gray2rgba<float4>(v3);
    EXPECT_FLOAT_EQ(v3, r3.x);
    EXPECT_FLOAT_EQ(v3, r3.y);
    EXPECT_FLOAT_EQ(v3, r3.z);
    EXPECT_FLOAT_EQ(1.f, r3.w);
}

TEST(GrayToRgbaTest, works_uchar)
{
    float v1 = 255.f, v2 = 0.f, v3 = 30.f;
    uchar u1 = 255, u2 = 0, u3 = 30;

    uchar4 r1 = gray2rgba<uchar4>(v1);
    EXPECT_EQ(u1, r1.x);
    EXPECT_EQ(u1, r1.y);
    EXPECT_EQ(u1, r1.z);
    EXPECT_EQ(255, r1.w);

    uchar4 r2 = gray2rgba<uchar4>(v2);
    EXPECT_EQ(u2, r2.x);
    EXPECT_EQ(u2, r2.y);
    EXPECT_EQ(u2, r2.z);
    EXPECT_EQ(255, r2.w);

    uchar4 r3 = gray2rgba<uchar4>(v3);
    EXPECT_EQ(u3, r3.x);
    EXPECT_EQ(u3, r3.y);
    EXPECT_EQ(u3, r3.z);
    EXPECT_EQ(255, r3.w);
}

TEST(GrayToRgbaTest, works_int)
{
    float v1 = 255.f, v2 = 0.f, v3 = 30.f;
    int u1 = 255, u2 = 0, u3 = 30;

    int4 r1 = gray2rgba<int4>(v1);
    EXPECT_EQ(u1, r1.x);
    EXPECT_EQ(u1, r1.y);
    EXPECT_EQ(u1, r1.z);
    EXPECT_EQ(255, r1.w);

    int4 r2 = gray2rgba<int4>(v2);
    EXPECT_EQ(u2, r2.x);
    EXPECT_EQ(u2, r2.y);
    EXPECT_EQ(u2, r2.z);
    EXPECT_EQ(255, r2.w);

    int4 r3 = gray2rgba<int4>(v3);
    EXPECT_EQ(u3, r3.x);
    EXPECT_EQ(u3, r3.y);
    EXPECT_EQ(u3, r3.z);
    EXPECT_EQ(255, r3.w);
}