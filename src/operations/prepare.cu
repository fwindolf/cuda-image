#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/utils.h"
#include "cuimage/operations/reduce_cu.h"

#include <nvfunctional>

namespace cuimage
{

template <typename T, typename Q, typename Op>
__global__ void g_Prepare(
    DevPtr<Q> result, const DevPtr<T> image, Op operation)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= image.width || pos.y >= image.height)
        return;

    result(pos.x, pos.y) = operation.getOp()(image(pos.x, pos.y));
}

template <typename T, typename Q, typename Op>
void cu_Prepare(DevPtr<Q>& result, const DevPtr<T>& image, Op operation)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(image.width, image.height, block);

    g_Prepare<<<grid, block>>>(result, image, operation);

    cudaCheckLastCall();
#ifndef DNDEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T, typename Q> struct SquareNormOp
{
    nvstd::function<Q(const T&)> __device__ getOp()
    {
        return [*this] __device__(const T& v) -> Q {
            if (!isvalid(v))
                return make<Q>(0.f);

            return static_cast<Q>(sum(v * v));
        };
    }
};

template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_SquareNorm(DevPtr<Q> result, const DevPtr<T> image)
{
    SquareNormOp<T, Q> op;
    cu_Prepare(result, image, op);
}

template <typename T> struct SquareOp
{
    nvstd::function<T(const T&)> __device__ getOp()
    {
        return [*this] __device__(const T& v) -> T {
            if (!isvalid(v))
                return make<T>(0.f);

            return (v * v);
        };
    }
};

template <typename T> void cu_Square(DevPtr<T> result, const DevPtr<T> image)
{
    SquareOp<T> op;
    cu_Prepare(result, image, op);
}

template <typename T, typename Q> struct PixelSumOp
{
    nvstd::function<Q(const T&)> __device__ getOp()
    {
        return [*this] __device__(const T& v) -> Q {
            if (!isvalid(v))
                return make<Q>(0.f);

            return static_cast<Q>(sum(v));
        };
    }
};

template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_PixelSum(DevPtr<Q> result, const DevPtr<T> image)
{
    PixelSumOp<T, Q> op;
    cu_Prepare(result, image, op);
}

template <typename T, typename Q> struct MarkValidOp
{
    nvstd::function<Q(const T&)> __device__ getOp()
    {
        return [*this] __device__(const T& v) -> Q {
            if (isvalid(v))
                return static_cast<Q>(1.f);

            return static_cast<Q>(0.f);
        };
    }
};

template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_MarkValid(DevPtr<Q> result, const DevPtr<T> image)
{
    MarkValidOp<T, Q> op;
    cu_Prepare(result, image, op);
}

template <typename T, typename Q> struct MarkNonZeroOp
{
    nvstd::function<Q(const T&)> __device__ getOp()
    {
        return [*this] __device__(const T& v) -> Q {
            if (isvalid(v) && !iszero(v))
                return static_cast<Q>(1.f);

            return static_cast<Q>(0.f);
        };
    }
};

template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_MarkNonzero(DevPtr<Q> result, const DevPtr<T> image)
{
    MarkNonZeroOp<T, Q> op;
    cu_Prepare(result, image, op);
}

template <typename T, typename Q> struct MarkAboveOp
{
    MarkAboveOp(const T threshold)
        : threshold_(threshold)
    {
    }

    nvstd::function<Q(const T&)> __device__ getOp()
    {
        return [*this] __device__(const T& v) -> Q {
            if (isvalid(v) && v > threshold_)
                return static_cast<Q>(1.f);

            return static_cast<Q>(0.f);
        };
    }

private:
    T threshold_;
};

template <typename T, typename Q,
    typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
void cu_MarkAboveValue(DevPtr<Q> result, const DevPtr<T> image, const T& value)
{
    MarkAboveOp<T, Q> op(value);
    cu_Prepare(result, image, op);
}

#define DECLARE_PREPARE_FUNCTION(type, typeo, name)                           \
    template void name(DevPtr<typeo>, const DevPtr<type>);

#define DECLARE_PREPARE_OPERATION(type, typeo, operation)                     \
    template void cu_Prepare(                                                 \
        DevPtr<typeo>&, const DevPtr<type>&, operation<type, typeo>);

FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, float, cu_SquareNorm)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, float, SquareNormOp)
FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, int, cu_SquareNorm)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, int, SquareNormOp)

FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, float, cu_PixelSum)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, float, PixelSumOp)
FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, int, cu_PixelSum)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, int, PixelSumOp)

FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, float, cu_MarkValid)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, float, MarkValidOp)
FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, int, cu_MarkValid)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, int, MarkValidOp)

FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, float, cu_MarkNonzero)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, float, MarkNonZeroOp)
FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, int, cu_MarkNonzero)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, int, MarkNonZeroOp)

#undef DECLARE_PREPARE_FUNCTION
#define DECLARE_PREPARE_FUNCTION(type, typeo, name)                           \
    template void name(DevPtr<typeo>, const DevPtr<type>, const type&);

FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, float, cu_MarkAboveValue)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, float, MarkAboveOp)
FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, int, cu_MarkAboveValue)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, int, MarkAboveOp)

#undef DECLARE_PREPARE_FUNCTION
#define DECLARE_PREPARE_FUNCTION(type, name)                                  \
    template void name(DevPtr<type>, const DevPtr<type>);

#undef DECLARE_PREPARE_OPERATION
#define DECLARE_PREPARE_OPERATION(type, operation)                            \
    template void cu_Prepare(                                                 \
        DevPtr<type>&, const DevPtr<type>&, operation<type>);

FOR_EACH_TYPE(DECLARE_PREPARE_FUNCTION, cu_Square)
FOR_EACH_TYPE(DECLARE_PREPARE_OPERATION, SquareOp)

} // image