#include "image/operations/transform_cu.h"

#include "image/cuda/utils.h"
#include "image/cuda/arithmetic.h"
#include "image/cuda/kernel.h"
#include "image/cuda/devptr.h"

#include <nvfunctional>

namespace image
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

/**
 * Explicit template instantiation
 */

#define DECLARE_TRANSFORM_FUNCTION(type, operation) \
    template void cu_Transform(DevPtr<type>&, operation<type>);

#define DECLARE_SET_FUNCTION(type, name) \
    template void name(DevPtr<type>, const type&);

#define DECLARE_REPLACE_FUNCTION(type, name) \
    template void name(DevPtr<type>, const type&, const type&);

FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, SetValue)
FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, ReplaceValue)
FOR_EACH_TYPE(DECLARE_TRANSFORM_FUNCTION, ReplaceNan)

FOR_EACH_TYPE(DECLARE_SET_FUNCTION, cu_SetTo)
FOR_EACH_TYPE(DECLARE_SET_FUNCTION, cu_ReplaceNan)
FOR_EACH_TYPE(DECLARE_REPLACE_FUNCTION, cu_Replace)

} // image