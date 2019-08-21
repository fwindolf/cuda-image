#include "cuimage/cuda/arithmetic.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace cuimage;

TEST(MakeTest, templated_make_works)
{
    float f = -23.3f;
    auto f1 = make<float1>(f);
    EXPECT_EQ(f1.x, f);

    auto f2 = make<float2>(f);
    EXPECT_EQ(f2.x, f);
    EXPECT_EQ(f2.y, f);

    auto f3 = make<float3>(f);
    EXPECT_EQ(f3.x, f);
    EXPECT_EQ(f3.y, f);
    EXPECT_EQ(f3.z, f);

    auto f4 = make<float4>(f);
    EXPECT_EQ(f4.x, f);
    EXPECT_EQ(f4.y, f);
    EXPECT_EQ(f4.z, f);
    EXPECT_EQ(f4.w, f);

    uchar u = 22;
    auto u1 = make<uchar1>(u);
    EXPECT_EQ(u1.x, u);

    auto u2 = make<uchar2>(u);
    EXPECT_EQ(u2.x, u);
    EXPECT_EQ(u2.y, u);

    auto u3 = make<uchar3>(u);
    EXPECT_EQ(u3.x, u);
    EXPECT_EQ(u3.y, u);
    EXPECT_EQ(u3.z, u);

    auto u4 = make<uchar4>(u);
    EXPECT_EQ(u4.x, u);
    EXPECT_EQ(u4.y, u);
    EXPECT_EQ(u4.z, u);
    EXPECT_EQ(u4.w, u);

    int i = -3;
    auto i1 = make<int1>(i);
    EXPECT_EQ(i1.x, i);

    auto i2 = make<int2>(i);
    EXPECT_EQ(i2.x, i);
    EXPECT_EQ(i2.y, i);

    auto i3 = make<int3>(i);
    EXPECT_EQ(i3.x, i);
    EXPECT_EQ(i3.y, i);
    EXPECT_EQ(i3.z, i);

    auto i4 = make<int4>(i);
    EXPECT_EQ(i4.x, i);
    EXPECT_EQ(i4.y, i);
    EXPECT_EQ(i4.z, i);
    EXPECT_EQ(i4.w, i);
}

template <typename T> struct CompareOpTest
{
public:
    void test()
    {
        // <
        EXPECT_FALSE(val_ref < val_eq);
        EXPECT_TRUE(val_ref <= val_eq);
        EXPECT_TRUE(val_ref == val_eq);
        EXPECT_TRUE(val_ref >= val_eq);
        EXPECT_FALSE(val_ref > val_eq);

        EXPECT_TRUE(val_ref <= val_g);
        EXPECT_TRUE(val_ref < val_g);
        EXPECT_FALSE(val_ref == val_g);
        EXPECT_FALSE(val_ref >= val_g);
        EXPECT_FALSE(val_ref > val_g);

        if (channels<T>() > 1 || is_float<T>())
        {
            EXPECT_FALSE(val_ref < val_gm);
            EXPECT_FALSE(val_ref <= val_gm);
            EXPECT_FALSE(val_ref == val_gm);
            EXPECT_FALSE(val_ref >= val_gm);
            EXPECT_FALSE(val_ref > val_gm);
        }

        EXPECT_FALSE(val_ref < val_l);
        EXPECT_FALSE(val_ref <= val_l);
        EXPECT_FALSE(val_ref == val_l);
        EXPECT_TRUE(val_ref >= val_l);
        EXPECT_TRUE(val_ref > val_l);

        if (channels<T>() > 1 || is_float<T>())
        {
            EXPECT_FALSE(val_ref < val_lm);
            EXPECT_FALSE(val_ref <= val_lm);
            EXPECT_FALSE(val_ref == val_lm);
            EXPECT_FALSE(val_ref >= val_lm);
            EXPECT_FALSE(val_ref > val_lm);
        }
    }

    T val_ref;
    T val_eq; // Value that is equal to ref
    T val_g, val_gm; // Value that is greater than ref, and mixed greater
    T val_l, val_lm; // Value that is less than ref, and mixed less
};

TEST(CompareOpTest, float1_works)
{
    CompareOpTest<float1> t;
    t.val_ref = make_float1(0.f);
    t.val_eq = make_float1(0.f);
    t.val_g = make_float1(1.f);
    t.val_gm = make_float1(std::nanf("")); // meh
    t.val_l = make_float1(-1.f);
    t.val_lm = make_float1(std::nanf("")); // meh

    t.test();
}

TEST(CompareOpTest, float2_works)
{
    CompareOpTest<float2> t;
    t.val_ref = make_float2(0.f, 0.f);
    t.val_eq = make_float2(0.f, 0.f);
    t.val_g = make_float2(1.f, 2.f);
    t.val_gm = make_float2(-1.f, 1.f);
    t.val_l = make_float2(-1.f, -2.f);
    t.val_lm = make_float2(-1.f, 1.f);

    t.test();
}

TEST(CompareOpTest, float3_works)
{
    CompareOpTest<float3> t;
    t.val_ref = make_float3(0.f, 0.f, 0.f);
    t.val_eq = make_float3(0.f, 0.f, 0.f);
    t.val_g = make_float3(1.f, 2.f, 3.f);
    t.val_gm = make_float3(-1.f, 1.f, 0.f);
    t.val_l = make_float3(-1.f, -2.f, -3.f);
    t.val_lm = make_float3(-1.f, 1.f, 0.f);

    t.test();
}

TEST(CompareOpTest, float4_works)
{
    CompareOpTest<float4> t;
    t.val_ref = make_float4(0.f, 0.f, 0.f, 0.f);
    t.val_eq = make_float4(0.f, 0.f, 0.f, 0.f);
    t.val_g = make_float4(1.f, 2.f, 3.f, 4.f);
    t.val_gm = make_float4(-1.f, 1.f, 0.f, -1.f);
    t.val_l = make_float4(-1.f, -2.f, -3.f, -4.f);
    t.val_lm = make_float4(-1.f, 1.f, 0.f, 1.f);

    t.test();
}

TEST(CompareOpTest, uchar1_works)
{
    CompareOpTest<uchar1> t;
    t.val_ref = make_uchar1(1);
    t.val_eq = make_uchar1(1);
    t.val_g = make_uchar1(2);
    t.val_l = make_uchar1(0);

    t.test();
}

TEST(CompareOpTest, uchar2_works)
{
    CompareOpTest<uchar2> t;
    t.val_ref = make_uchar2(1, 2);
    t.val_eq = make_uchar2(1, 2);
    t.val_g = make_uchar2(3, 4);
    t.val_gm = make_uchar2(3, 1);
    t.val_l = make_uchar2(0, 1);
    t.val_lm = make_uchar2(0, 4);

    t.test();
}

TEST(CompareOpTest, uchar3_works)
{
    CompareOpTest<uchar3> t;
    t.val_ref = make_uchar3(3, 1, 4);
    t.val_eq = make_uchar3(3, 1, 4);
    t.val_g = make_uchar3(5, 2, 6);
    t.val_gm = make_uchar3(5, 0, 4);
    t.val_l = make_uchar3(2, 0, 2);
    t.val_lm = make_uchar3(3, 0, 5);

    t.test();
}

TEST(CompareOpTest, uchar4_works)
{
    CompareOpTest<uchar4> t;
    t.val_ref = make_uchar4(3, 1, 4, 1);
    t.val_eq = make_uchar4(3, 1, 4, 1);
    t.val_g = make_uchar4(5, 2, 5, 2);
    t.val_gm = make_uchar4(5, 1, 2, 5);
    t.val_l = make_uchar4(0, 0, 0, 0);
    t.val_lm = make_uchar4(0, 1, 2, 2);

    t.test();
}

TEST(CompareOpTest, int1_works)
{
    CompareOpTest<int1> t;
    t.val_ref = make_int1(0);
    t.val_eq = make_int1(0);
    t.val_g = make_int1(1);
    t.val_l = make_int1(-1);

    t.test();
}

TEST(CompareOpTest, int2_works)
{
    CompareOpTest<int2> t;
    t.val_ref = make_int2(0, 0);
    t.val_eq = make_int2(0, 0);
    t.val_g = make_int2(1, 2);
    t.val_gm = make_int2(-1, 1);
    t.val_l = make_int2(-1, -2);
    t.val_lm = make_int2(-1, 1);

    t.test();
}

TEST(CompareOpTest, int3_works)
{
    CompareOpTest<int3> t;
    t.val_ref = make_int3(0, 0, 0);
    t.val_eq = make_int3(0, 0, 0);
    t.val_g = make_int3(1, 2, 3);
    t.val_gm = make_int3(-1, 1, 0);
    t.val_l = make_int3(-1, -2, -3);
    t.val_lm = make_int3(-1, 1, 0);

    t.test();
}

TEST(CompareOpTest, int4_works)
{
    CompareOpTest<int4> t;
    t.val_ref = make_int4(0, 0, 0, 0);
    t.val_eq = make_int4(0, 0, 0, 0);
    t.val_g = make_int4(1, 2, 3, 4);
    t.val_gm = make_int4(-1, 1, 0, -1);
    t.val_l = make_int4(-1, -2, -3, -4);
    t.val_lm = make_int4(-1, 1, 0, 1);

    t.test();
}

template <typename T> struct MathOpTest
{
public:
    void test(const T sum, const T sub, const T mul, const T div)
    {
        T l_c = l;
        T r_c = r;

        EXPECT_TRUE(sum == l + r);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);

        EXPECT_TRUE(sub == l - r);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);

        EXPECT_TRUE(mul == l * r);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);

        EXPECT_TRUE(div == l / r);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);

        T l_l = l;
        T r_l = r;
        l_l += r;
        r_l += l;
        EXPECT_TRUE(sum == l_l);
        EXPECT_TRUE(sum == r_l);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);

        l_l = l;
        l_l -= r;
        // not associative
        EXPECT_TRUE(sub == l_l);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);

        l_l = l;
        r_l = r;
        l_l *= r;
        r_l *= l;
        EXPECT_TRUE(mul == l_l);
        EXPECT_TRUE(mul == r_l);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);

        l_l = l;
        l_l /= r;
        // not associative
        EXPECT_TRUE(div == l_l);
        ASSERT_TRUE(l_c == l);
        ASSERT_TRUE(r_c == r);
    }

    T l, r;
};

TEST(MathOpTest, float1_works)
{
    float f11 = 14.f, f21 = 2.f;

    MathOpTest<float1> t;
    t.l = make_float1(f11);
    t.r = make_float1(f21);

    t.test(make_float1(f11 + f21), make_float1(f11 - f21),
        make_float1(f11 * f21), make_float1(f11 / f21));
}

TEST(MathOpTest, float2_works)
{
    float f11 = 14.f, f12 = 5.f, f21 = 2.f, f22 = 4.f;

    MathOpTest<float2> t;
    t.l = make_float2(f11, f12);
    t.r = make_float2(f21, f22);

    t.test(make_float2(f11 + f21, f12 + f22),
        make_float2(f11 - f21, f12 - f22), make_float2(f11 * f21, f12 * f22),
        make_float2(f11 / f21, f12 / f22));
}

TEST(MathOpTest, float3_works)
{
    float f11 = 14.f, f12 = 5.f, f13 = -4.f, f21 = 2.f, f22 = 4.f, f23 = 0.f;

    MathOpTest<float3> t;
    t.l = make_float3(f11, f12, f13);
    t.r = make_float3(f21, f22, f23);

    t.test(make_float3(f11 + f21, f12 + f22, f13 + f23),
        make_float3(f11 - f21, f12 - f22, f13 - f23),
        make_float3(f11 * f21, f12 * f22, f13 * f23),
        make_float3(f11 / f21, f12 / f22, f13 / f23));
}

TEST(MathOpTest, float4_works)
{
    float f11 = 14.f, f12 = 5.f, f13 = -4.f, f14 = 0.f, f21 = 2.f, f22 = 4.f,
          f23 = 0.f, f24 = -1.f;

    MathOpTest<float4> t;
    t.l = make_float4(f11, f12, f13, f14);
    t.r = make_float4(f21, f22, f23, f24);

    t.test(make_float4(f11 + f21, f12 + f22, f13 + f23, f14 + f24),
        make_float4(f11 - f21, f12 - f22, f13 - f23, f14 - f24),
        make_float4(f11 * f21, f12 * f22, f13 * f23, f14 * f24),
        make_float4(f11 / f21, f12 / f22, f13 / f23, f14 / f24));
}

TEST(MathOpTest, uchar1_works)
{
    uchar f11 = 14, f21 = 2;

    MathOpTest<uchar1> t;
    t.l = make_uchar1(f11);
    t.r = make_uchar1(f21);

    t.test(make_uchar1(f11 + f21), make_uchar1(f11 - f21),
        make_uchar1(f11 * f21), make_uchar1(f11 / f21));
}

TEST(MathOpTest, uchar2_works)
{
    uchar f11 = 14, f12 = 5, f21 = 2, f22 = 4;

    MathOpTest<uchar2> t;
    t.l = make_uchar2(f11, f12);
    t.r = make_uchar2(f21, f22);

    t.test(make_uchar2(f11 + f21, f12 + f22),
        make_uchar2(f11 - f21, f12 - f22), make_uchar2(f11 * f21, f12 * f22),
        make_uchar2(f11 / f21, f12 / f22));
}

TEST(MathOpTest, uchar3_works)
{
    uchar f11 = 14, f12 = 5, f13 = 4, f21 = 2, f22 = 4, f23 = 1;

    MathOpTest<uchar3> t;
    t.l = make_uchar3(f11, f12, f13);
    t.r = make_uchar3(f21, f22, f23);

    t.test(make_uchar3(f11 + f21, f12 + f22, f13 + f23),
        make_uchar3(f11 - f21, f12 - f22, f13 - f23),
        make_uchar3(f11 * f21, f12 * f22, f13 * f23),
        make_uchar3(f11 / f21, f12 / f22, f13 / f23));
}

TEST(MathOpTest, uchar4_works)
{
    uchar f11 = 14, f12 = 5, f13 = 4, f14 = 3, f21 = 2, f22 = 4, f23 = 5,
          f24 = 1;

    MathOpTest<uchar4> t;
    t.l = make_uchar4(f11, f12, f13, f14);
    t.r = make_uchar4(f21, f22, f23, f24);

    t.test(make_uchar4(f11 + f21, f12 + f22, f13 + f23, f14 + f24),
        make_uchar4(f11 - f21, f12 - f22, f13 - f23, f14 - f24),
        make_uchar4(f11 * f21, f12 * f22, f13 * f23, f14 * f24),
        make_uchar4(f11 / f21, f12 / f22, f13 / f23, f14 / f24));
}

TEST(MathOpTest, int1_works)
{
    uchar f11 = 14, f21 = 2;

    MathOpTest<int1> t;
    t.l = make_int1(f11);
    t.r = make_int1(f21);

    t.test(make_int1(f11 + f21), make_int1(f11 - f21), make_int1(f11 * f21),
        make_int1(f11 / f21));
}

TEST(MathOpTest, int2_works)
{
    int f11 = 14, f12 = 5, f21 = 2, f22 = 4;

    MathOpTest<int2> t;
    t.l = make_int2(f11, f12);
    t.r = make_int2(f21, f22);

    t.test(make_int2(f11 + f21, f12 + f22), make_int2(f11 - f21, f12 - f22),
        make_int2(f11 * f21, f12 * f22), make_int2(f11 / f21, f12 / f22));
}

TEST(MathOpTest, int3_works)
{
    int f11 = 14, f12 = 5, f13 = -4, f21 = 2, f22 = 4, f23 = -6;

    MathOpTest<int3> t;
    t.l = make_int3(f11, f12, f13);
    t.r = make_int3(f21, f22, f23);

    t.test(make_int3(f11 + f21, f12 + f22, f13 + f23),
        make_int3(f11 - f21, f12 - f22, f13 - f23),
        make_int3(f11 * f21, f12 * f22, f13 * f23),
        make_int3(f11 / f21, f12 / f22, f13 / f23));
}

TEST(MathOpTest, int4_works)
{
    int f11 = 14, f12 = 5, f13 = -4, f14 = 1, f21 = 2, f22 = 4, f23 = 2,
        f24 = -1;

    MathOpTest<int4> t;
    t.l = make_int4(f11, f12, f13, f14);
    t.r = make_int4(f21, f22, f23, f24);

    t.test(make_int4(f11 + f21, f12 + f22, f13 + f23, f14 + f24),
        make_int4(f11 - f21, f12 - f22, f13 - f23, f14 - f24),
        make_int4(f11 * f21, f12 * f22, f13 * f23, f14 * f24),
        make_int4(f11 / f21, f12 / f22, f13 / f23, f14 / f24));
}

template <typename T> struct MiscFunctionTest
{
public:
    void test(float minv, float maxv, float sumv, T absv)
    {
        EXPECT_TRUE(isvalid(val));
        EXPECT_FALSE(isnan(val));
        EXPECT_FALSE(isinf(val));
        EXPECT_FALSE(iszero(val));

        EXPECT_TRUE(min(val) == minv);
        EXPECT_TRUE(max(val) == maxv);
        EXPECT_TRUE(abs(val) == absv);
        EXPECT_TRUE(sum(val) == sumv);
    }

    T val;
};

TEST(MiscFunctionTest, float1_works)
{
    MiscFunctionTest<float1> t;

    float v = -2.f;
    t.val = make_float1(v);

    t.test(v, v, v, make_float1(-v));
}

TEST(MiscFunctionTest, float2_works)
{
    MiscFunctionTest<float2> t;

    float v1 = -2.f, v2 = 3.f;
    t.val = make_float2(v1, v2);

    t.test(v1, v2, v1 + v2, make_float2(-v1, v2));
}

TEST(MiscFunctionTest, float3_works)
{
    MiscFunctionTest<float3> t;

    float v1 = -2.f, v2 = 3.f, v3 = 1.f;
    t.val = make_float3(v1, v2, v3);

    t.test(v1, v2, v1 + v2 + v3, make_float3(-v1, v2, v3));
}

TEST(MiscFunctionTest, float4_works)
{
    MiscFunctionTest<float4> t;

    float v1 = -2.f, v2 = 3.f, v3 = 1.f, v4 = -6.2f;
    t.val = make_float4(v1, v2, v3, v4);

    t.test(v4, v2, v1 + v2 + v3 + v4, make_float4(-v1, v2, v3, -v4));
}

TEST(MiscFunctionTest, uchar1_works)
{
    MiscFunctionTest<uchar1> t;

    uchar v = 3;
    t.val = make_uchar1(v);

    t.test(v, v, v, make_uchar1(v));
}

TEST(MiscFunctionTest, uchar2_works)
{
    MiscFunctionTest<uchar2> t;

    uchar v1 = 2, v2 = 3;
    t.val = make_uchar2(v1, v2);

    t.test(v1, v2, v1 + v2, make_uchar2(v1, v2));
}

TEST(MiscFunctionTest, uchar3_works)
{
    MiscFunctionTest<uchar3> t;

    uchar v1 = 2, v2 = 5, v3 = 100;
    t.val = make_uchar3(v1, v2, v3);

    t.test(v1, v3, v1 + v2 + v3, make_uchar3(v1, v2, v3));
}

TEST(MiscFunctionTest, uchar4_works)
{
    MiscFunctionTest<uchar4> t;

    uchar v1 = 25, v2 = 32, v3 = 10, v4 = 90;
    t.val = make_uchar4(v1, v2, v3, v4);

    t.test(v3, v4, v1 + v2 + v3 + v4, make_uchar4(v1, v2, v3, v4));
}

TEST(MiscFunctionTest, int1_works)
{
    MiscFunctionTest<int1> t;

    int v = -23;
    t.val = make_int1(v);

    t.test(v, v, v, make_int1(-v));
}

TEST(MiscFunctionTest, int2_works)
{
    MiscFunctionTest<int2> t;

    int v1 = 2, v2 = -3;
    t.val = make_int2(v1, v2);

    t.test(v2, v1, v1 + v2, make_int2(v1, -v2));
}

TEST(MiscFunctionTest, int3_works)
{
    MiscFunctionTest<int3> t;

    int v1 = 2, v2 = -5, v3 = 100;
    t.val = make_int3(v1, v2, v3);

    t.test(v2, v3, v1 + v2 + v3, make_int3(v1, -v2, v3));
}

TEST(MiscFunctionTest, int4_works)
{
    MiscFunctionTest<int4> t;

    int v1 = 25, v2 = -32, v3 = 10, v4 = -90;
    t.val = make_int4(v1, v2, v3, v4);

    t.test(v4, v1, v1 + v2 + v3 + v4, make_int4(v1, -v2, v3, -v4));
}