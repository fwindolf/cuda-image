#include "cuimage/image.h"

#include <gtest/gtest.h>
#include <random>

using namespace cuimage;
using namespace ::testing;

class ResizeDepth : public Test
{
public:
    ResizeDepth()
        : w(20)
        , h(12)
        , dis(1.0, 1000.0)
    {
        gen.seed(1);

        float data[w * h];
        for (int i = 0; i < w * h; i++)
            data[i] = dis(gen);

        depth.upload(data, w, h);
    }

    // Random number engine
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis;

    Image<float> depth;
    const int w, h;
};

TEST_F(ResizeDepth, linear_resize_produces_right_dims)
{
    const int w_new = 100, h_new = 200;
    auto depth_resized = depth.resized(w_new, h_new, ResizeMode::LINEAR);

    // New resolution
    EXPECT_EQ(depth_resized.width(), w_new);
    EXPECT_EQ(depth_resized.height(), h_new);

    // Old resolution didnt change
    EXPECT_EQ(depth.width(), w);
    EXPECT_EQ(depth.height(), h);
}

TEST_F(ResizeDepth, linear_resize_produces_resized_data)
{
    float val = 1.f;
    depth.setTo(val);
    const int w_new = 100, h_new = 200;
    auto depth_resized = depth.resized(w_new, h_new, ResizeMode::LINEAR);

    // All data is still 1
    EXPECT_FLOAT_EQ(depth_resized.min(), val);
    EXPECT_FLOAT_EQ(depth_resized.max(), val);
}

TEST_F(ResizeDepth, linear_resize_produces_no_invalid_data) 
{
    const int w_new = 100, h_new = 200;
    auto depth_resized = depth.resized(w_new, h_new, ResizeMode::LINEAR);

    // Data should be interpolated between the existing
    EXPECT_GE(depth_resized.min(), depth.min());
    EXPECT_LE(depth_resized.max(), depth.max());

    // No zeros, no nans
    EXPECT_EQ(depth_resized.nan(), 0);
    EXPECT_EQ(depth_resized.nonzero(), depth_resized.size());
}

TEST_F(ResizeDepth, linear_resize_uses_only_data_in_mask)
{
    const int w_new = 30, h_new = 18;
    ASSERT_EQ(depth.nonzero(), depth.size());

    uchar mdata[w * h];
    for (int i = 0; i < w * h; i++)
        mdata[i] = dis(gen) > 800 ? 1 : 0;

    Image<uchar> mask(w, h);
    mask.upload(mdata, w, h);

    // Create the same image, but with values that should occur in the final image if used
    Image<float> depth_masked = depth;
    depth_masked.mask(mask);
    depth_masked.replace(0.f, -1e20f);

    // The resized depth should not contain the values that were masked out
    auto depth_resized = depth_masked.resized(w_new, h_new, mask, ResizeMode::LINEAR);
    EXPECT_FLOAT_EQ(depth_resized.min(), 0.f);
}

TEST_F(ResizeDepth, linear_resize_interpolates_with_zero)
{
    const int w_new = 30, h_new = 18;
    ASSERT_EQ(depth.nonzero(), depth.size());

    uchar mdata[w * h];
    for (int i = 0; i < w * h; i++)
        mdata[i] = dis(gen) > 500 ? 1 : 0;

    Image<uchar> mask(w, h);
    mask.upload(mdata, w, h);

    // Create the same image, but with 0s where the mask is
    Image<float> depth_masked = depth;
    depth_masked.mask(mask);

    // Depth should have lower values, as 0s were interpolated with
    auto depth_resized = depth_masked.resized(w_new, h_new, ResizeMode::LINEAR);
    EXPECT_LT(depth_resized.min(), depth.min());
}

TEST_F(ResizeDepth, linear_resize_interpolates_with_nan) 
{
    const int w_new = 30, h_new = 18;
    ASSERT_EQ(depth.nonzero(), depth.size());

    uchar mdata[w * h];
    for (int i = 0; i < w * h; i++)
        mdata[i] = dis(gen) > 500 ? 1 : 0;

    Image<uchar> mask(w, h);
    mask.upload(mdata, w, h);

    // Create the same image, but with 0s where the mask is
    Image<float> depth_masked = depth;
    depth_masked.mask(mask);
    depth_masked.replace(0.f, nanf(""));

    // Should contain nans now
    auto depth_resized = depth_masked.resized(w_new, h_new, ResizeMode::LINEAR);
    EXPECT_LT(depth_resized.valid(), depth_resized.size());
}

TEST_F(ResizeDepth, linear_nonzero_resize_produces_right_dims) 
{
    const int w_new = 100, h_new = 200;
    auto depth_resized = depth.resized(w_new, h_new, ResizeMode::LINEAR_NONZERO);

    // New resolution
    EXPECT_EQ(depth_resized.width(), w_new);
    EXPECT_EQ(depth_resized.height(), h_new);

    // Old resolution didnt change
    EXPECT_EQ(depth.width(), w);
    EXPECT_EQ(depth.height(), h);
}

TEST_F(ResizeDepth, linear_nonzero_resize_produces_resized_data) 
{
    float val = 1.f;
    depth.setTo(val);
    const int w_new = 100, h_new = 200;
    auto depth_resized = depth.resized(w_new, h_new, ResizeMode::LINEAR_NONZERO);

    // All data is still 1
    EXPECT_FLOAT_EQ(depth_resized.min(), val);
    EXPECT_FLOAT_EQ(depth_resized.max(), val);
}

TEST_F(ResizeDepth, linear_nonzero_resize_produces_no_invalid_data) 
{
    const int w_new = 100, h_new = 200;
    auto depth_resized = depth.resized(w_new, h_new, ResizeMode::LINEAR_NONZERO);

    // Data should be interpolated between the existing
    EXPECT_GE(depth_resized.min(), depth.min());
    EXPECT_LE(depth_resized.max(), depth.max());

    // No zeros, no nans
    EXPECT_EQ(depth_resized.nan(), 0);
    EXPECT_EQ(depth_resized.nonzero(), depth_resized.size());
}

TEST_F(ResizeDepth, linear_nonzero_resize_uses_only_data_in_mask) 
{
    const int w_new = 100, h_new = 200;
    ASSERT_EQ(depth.nonzero(), depth.size());

    uchar mdata[w * h];
    for (int i = 0; i < w * h; i++)
        mdata[i] = dis(gen) > 800 ? 1 : 0;

    Image<uchar> mask(w, h);
    mask.upload(mdata, w, h);

    // Create the same image, but with values that should occur in the final image if used
    Image<float> depth_masked = depth;
    depth_masked.mask(mask);
    depth_masked.replace(0.f, -1e20f);

    // The resized depth should not contain the values that were masked out
    auto depth_resized = depth_masked.resized(w_new, h_new, mask, ResizeMode::LINEAR_NONZERO);
    EXPECT_FLOAT_EQ(depth_resized.min(), 0.f);
}

TEST_F(ResizeDepth, linear_nonzero_resize_doesnt_interpolate_with_zero)
{
    const int w_new = 100, h_new = 200;

    auto depth_zeros = depth;
    depth_zeros.thresholdInv(500, 0.f);

    ASSERT_LT(depth_zeros.nonzero(), depth_zeros.size());

    // The resized depth should not contain values lower than 500
    auto depth_resized = depth_zeros.resized(w_new, h_new, ResizeMode::LINEAR_NONZERO);
    EXPECT_GT(depth_resized.min(), 500.f);
}

TEST_F(ResizeDepth, linear_nonzero_resize_doesnt_interpolate_with_nan) 
{
    const int w_new = 100, h_new = 200;

    auto depth_zeros = depth;
    auto depth_nans = depth;
    depth_zeros.thresholdInv(500, 0.f);
    depth_nans.thresholdInv(500, nanf(""));

    ASSERT_LT(depth_zeros.nonzero(), depth_zeros.size());
    ASSERT_LT(depth_nans.valid(), depth_nans.size());

    // The resized zeros image contains nans where there couldnt be a value interpolated
    auto depth_resized_ = depth_zeros.resized(w_new, h_new, ResizeMode::LINEAR_NONZERO);

    // The resized nans image should only contain the same nans, not more (that would mean
    // that there were valid values mixed with the nans, resulting in nans)
    auto depth_resized = depth_nans.resized(w_new, h_new, ResizeMode::LINEAR_NONZERO);

    EXPECT_EQ(depth_resized_.valid(), depth_resized.valid());
}