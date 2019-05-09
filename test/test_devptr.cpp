#include <gtest/gtest.h>

#include "cuimage/cuda/devptr.h"

using namespace cuimage;

TEST(DevPtrTest, can_be_created_with_data)
{
    int w = 10;
    int h = 8;
    float* data;
    ASSERT_NO_THROW(cudaSafeCall(cudaMalloc(&data, w * h * sizeof(float))));
    DevPtr<float> dptr(data, w, h);

    EXPECT_EQ(h, dptr.height);
    EXPECT_EQ(w, dptr.width);
    EXPECT_EQ(data, dptr.data);

    ASSERT_NO_THROW(cudaSafeCall(cudaFree(data)));
}

TEST(DevPtrTest, doesnt_free_data_on_delete)
{
    int w = 10;
    int h = 8;
    float* data;
    ASSERT_NO_THROW(cudaSafeCall(cudaMalloc(&data, w * h * sizeof(float))));
    {
        DevPtr<float> dptr(data, w, h);
        // ~DevPtr
    }
    EXPECT_NO_THROW(cudaSafeCall(cudaFree(data)));
}

TEST(DevPtrTest, allocates_data)
{
    int w = 10;
    int h = 8;
    DevPtr<float> dptr(w, h);

    EXPECT_NE(nullptr, dptr.data);
    EXPECT_NO_THROW(dptr.free());
}

TEST(DevPtrTest, can_be_assigned_to_empty)
{
    int w = 10;
    int h = 8;
    DevPtr<float> dptr1(nullptr, w, h);
    DevPtr<float> dptr2(w, h);

    dptr1 = dptr2;
    EXPECT_NE(dptr1.data, dptr2.data);
    EXPECT_NO_THROW(dptr1.free());
    EXPECT_NO_THROW(dptr2.free());
}

TEST(DevPtrTest, can_be_assigned_to_existing)
{
    int w = 10;
    int h = 8;
    DevPtr<float> dptr1(w, h);
    DevPtr<float> dptr2(w, h);

    dptr1 = dptr2;
    EXPECT_NE(dptr1.data, dptr2.data);
    EXPECT_NO_THROW(dptr1.free());
    EXPECT_NO_THROW(dptr2.free());
}