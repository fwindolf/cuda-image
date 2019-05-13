#include "cuimage/operations/transform_cu.h"

#include "cuimage/cuda/utils.h"
#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/devptr.h"

#include "cuimage/operations/reduce_cu.h"

#include <nvfunctional>

namespace cuimage
{

template <typename T, typename Op>
__global__ void g_Transform(DevPtr<T> image, Op operation)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= image.width || pos.y >= image.height)
        return;

    T& pxl = image(pos.x, pos.y);
    operation.getOp()(pxl);
}

template <typename T, typename Op>
void cu_Transform(DevPtr<T>& image, Op operation)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(image.width, image.height, block);

    g_Transform <<< grid, block >>> (image, operation);

    cudaCheckLastCall();
    cudaDeviceSynchronize();
}

template <typename T>
struct ReplaceValue
{
    ReplaceValue(const T& val, const T& with) 
    : val_(val), with_(with) {}

    // Return a function ptr from lambda that replaces the pixel with with_ if it is val_
    nvstd::function<void(T&)> __device__ getOp()
    {
        return [*this] __device__ (T& v) 
        {
            if (v == val_)            
                v = with_;  
        };
    }
private:
    const T val_, with_;
};

template <typename T>
void cu_Replace(DevPtr<T> image, const T& value, const T& with)
{
    if (isnan(value))
        return cu_ReplaceNan(image, with);

    ReplaceValue<T> replace_op(value, with);
    cu_Transform(image, replace_op);
}

template <typename T>
struct ReplaceNan
{
    ReplaceNan(const T& with) 
    : with_(with) {}

    // Return a function ptr from lambda that replaces the pixel with with_ if it is val_
    nvstd::function<void(T&)> __device__ getOp()
    {
        return [*this] __device__ (T& v) 
        {
            if (isnan(v))
                v = with_;
        };
    }
private:
    const T with_;
};

template <typename T>
void cu_ReplaceNan(DevPtr<T> image, const T& with)
{
    ReplaceNan<T> replace_op(with);
    cu_Transform(image, replace_op);
}

template <typename T>
struct SetValue
{
    SetValue(const T& val)
     : val_(val) {}

    // Return a function ptr from lambda that sets the pixel to value
    nvstd::function<void(T&)> __device__ getOp()
    {
        return [*this] __device__ (T& v) 
        {
            v = val_;
        };
    }
private:
    const T val_;
};

template <typename T>
void cu_SetTo(DevPtr<T> image, const T& value)
{
    SetValue<T> set_op(value);
    cu_Transform(image, set_op);
}

template <typename T>
struct Threshold
{
    Threshold(const T& threshold, const T& val)
     : thres_(threshold), val_(val) {}

    // Return a function ptr from lambda that sets the pixel to value
    nvstd::function<void(T&)> __device__ getOp()
    {
        return [*this] __device__ (T& v) 
        {
            if (v > thres_)
                v = val_;
        };
    }
private:
    const T thres_, val_;
};

template <typename T>
void cu_Threshold(DevPtr<T> image, const T& threshold, const T& value)
{
    Threshold<T> thres_op(threshold, value);
    cu_Transform(image, thres_op);
}

template <typename T>
struct ThresholdInv
{
    ThresholdInv(const T& threshold, const T& val)
     : thres_(threshold), val_(val) {}

    // Return a function ptr from lambda that sets the pixel to value
    nvstd::function<void(T&)> __device__ getOp()
    {
        return [*this] __device__ (T& v) 
        {
            if (v <= thres_)
                v = val_;
        };
    }
private:
    const T thres_, val_;
};

template <typename T>
void cu_ThresholdInv(DevPtr<T> image, const T& threshold, const T& value)
{
    ThresholdInv<T> thres_op(threshold, value);
    cu_Transform(image, thres_op);
}

template <typename T>
struct ThresholdLowHigh
{
    ThresholdLowHigh(const T& threshold, const T& low, const T& high)
     : thres_(threshold), low_(low), high_(high) {}

    // Return a function ptr from lambda that sets the pixel to value
    nvstd::function<void(T&)> __device__ getOp()
    {
        return [*this] __device__ (T& v) 
        {
            if (v >= thres_)
                v = high_;
            else
                v = low_;
        };
    }
private:
    const T thres_, low_, high_;
};

template <typename T>
void cu_Threshold(DevPtr<T> image, const T& threshold, const T& low, const T& high)
{
    ThresholdLowHigh<T> thres_op(threshold, low, high);
    cu_Transform(image, thres_op);
}

template <typename T, typename std::enable_if<has_0_channels<T>::value, T>::type* = nullptr>
struct MedianOp
{
    MedianOp(const T& minv, const T& maxv, int* data, const size_t& numBuckets, const float& bucketSize)
     : minv_(minv), 
       maxv_(maxv),
       buckets_(data),
       numBuckets_(numBuckets),
       bucketSize_(bucketSize)
    {
    }

    nvstd::function<void(T&)> __device__ getOp()
    {
        return [*this] __device__ (T& v)
        {
            // Add the element to the correct bucket
            const size_t idx = static_cast<size_t>((v - minv_) / bucketSize_);
            atomicAdd(&buckets_[idx], 1);
        };
    }
private:
    T minv_, maxv_;
    size_t numBuckets_;
    float bucketSize_;
    int* buckets_;
};

template <typename T>
T cu_Median(DevPtr<T> image)
{
    // Find dimensions for bucket
    T minv = cu_Min(image);
    T maxv = cu_Max(image);
    float bucketSize = is_float<T>() ? 0.01f : 1.f;
    size_t numBuckets = ((maxv - minv) / bucketSize) + 1;

    // Initialize buckets
    DevPtr<int> buckets(numBuckets, 1);
    cu_SetTo<int>(buckets, 0);

    // Fill buckets
    MedianOp<T> medianOp(minv, maxv, buckets.data, numBuckets, bucketSize);
    cu_Transform(image, medianOp);  
    
    // Transfer to CPU
    int* h_buckets = new int[numBuckets]();
    cudaSafeCall(cudaMemcpy(h_buckets, buckets.data, numBuckets * sizeof(int), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(buckets.data));
    
    // Add until at least half of pixels in buckets -> median
    int i = 0;
    size_t sum = 0;
    while(sum < (image.width * image.height) / 2)
    {
        sum += h_buckets[i];
        i++;
    }
    return static_cast<T>(minv + (i - 1) * bucketSize);
}




/**
 * Explicit template instantiation
 */

#define DECLARE_TRANSFORM_FUNCTION(type, operation) \
    template void cu_Transform(DevPtr<type>&, operation<type>);

#define DECLARE_SET_FUNCTION(type, name) \
    template void name(DevPtr<type>, const type&);

#define DECLARE_REPLACE_FUNCTION(type, name) \
    template void name(DevPtr<type>, const type&, const type&);

#define DECLARE_THRESHOLD_FUNCTION(type, name) \
    template void name(DevPtr<type>, const type&, const type&, const type&);

FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, SetValue)
FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, ReplaceValue)
FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, ReplaceNan)
FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, Threshold)
FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, ThresholdInv)
FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, ThresholdLowHigh)

FOR_EACH_TYPE(DECLARE_SET_FUNCTION, cu_SetTo)
FOR_EACH_TYPE(DECLARE_SET_FUNCTION, cu_ReplaceNan)
FOR_EACH_TYPE(DECLARE_REPLACE_FUNCTION, cu_Replace)
FOR_EACH_TYPE(DECLARE_REPLACE_FUNCTION, cu_Threshold)
FOR_EACH_TYPE(DECLARE_REPLACE_FUNCTION, cu_ThresholdInv)
FOR_EACH_TYPE(DECLARE_THRESHOLD_FUNCTION, cu_Threshold)

#define DECLARE_MEDIAN_FUNCTION(type, name) \
    template type name(DevPtr<type>);

FOR_EACH_0CHANNEL_TYPE(DECLARE_TRANSFORM_FUNCTION, MedianOp)
FOR_EACH_0CHANNEL_TYPE(DECLARE_MEDIAN_FUNCTION, cu_Median)


} // image