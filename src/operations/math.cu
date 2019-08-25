#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/devptr.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/utils.h"
#include "cuimage/operations/math_cu.h"

#include <nvfunctional>

namespace cuimage
{

template <typename T, typename Op>
__global__ void g_Apply(DevPtr<T> input1, const DevPtr<T> input2, Op operation)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= input1.width || pos.y >= input1.height)
        return;

    operation.op()(input1(pos.x, pos.y), input2(pos.x, pos.y));
}

template <typename T, typename Op>
__global__ void g_Apply(DevPtr<T> input, Op operation)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= input.width || pos.y >= input.height)
        return;

    operation.op()(input(pos.x, pos.y));
}

template <typename T, typename Op>
void cu_Apply(DevPtr<T> input1, const DevPtr<T> input2, Op operation)
{
    assert(input1.width == input2.width);
    assert(input1.height == input2.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(input1.width, input1.height, block);

    g_Apply<<<grid, block>>>(input1, input2, operation);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T, typename Op> void cu_Apply(DevPtr<T> input, Op operation)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(input.width, input.height, block);

    g_Apply<<<grid, block>>>(input, operation);

    cudaCheckLastCall();
#ifdef DEBUG
    cudaSafeCall(cudaDeviceSynchronize());
#endif
}

template <typename T> struct AddTo
{
    AddTo() {}

    nvstd::function<void(T&, const T&)> __device__ op()
    {
        return [*this] __device__(T & val1, const T& val2) { val1 += val2; };
    }
};

template <typename T> struct SubFrom
{
    SubFrom() {}

    nvstd::function<void(T&, const T&)> __device__ op()
    {
        return [*this] __device__(T & val1, const T& val2) { val1 -= val2; };
    }
};

template <typename T> struct MulBy
{
    MulBy() {}

    nvstd::function<void(T&, const T&)> __device__ op()
    {
        return [*this] __device__(T & val1, const T& val2) { val1 *= val2; };
    }
};

template <typename T> struct DivBy
{
    DivBy() {}

    nvstd::function<void(T&, const T&)> __device__ op()
    {
        return [*this] __device__(T & val1, const T& val2) { val1 /= val2; };
    }
};

template <typename T> void cu_AddTo(DevPtr<T> image, const DevPtr<T>& other)
{
    AddTo<T> add_op;
    cu_Apply(image, other, add_op);
}

template <typename T>
void cu_MultiplyBy(DevPtr<T> image, const DevPtr<T>& other)
{
    MulBy<T> mul_op;
    cu_Apply(image, other, mul_op);
}

template <typename T>
void cu_SubtractFrom(DevPtr<T> image, const DevPtr<T>& other)
{
    SubFrom<T> sub_op;
    cu_Apply(image, other, sub_op);
}

template <typename T> void cu_DivideBy(DevPtr<T> image, const DevPtr<T>& other)
{
    DivBy<T> div_op;
    cu_Apply(image, other, div_op);
}

template <typename T> struct AddValTo
{
    AddValTo(const T& val)
        : val_(val)
    {
    }

    nvstd::function<void(T&)> __device__ op()
    {
        return [*this] __device__(T & val) { val += val_; };
    }

private:
    const T val_;
};

template <typename T> struct SubValFrom
{
    SubValFrom(const T& val)
        : val_(val)
    {
    }

    nvstd::function<void(T&)> __device__ op()
    {
        return [*this] __device__(T & val) { val -= val_; };
    }

private:
    const T val_;
};

template <typename T> struct MulByVal
{
    MulByVal(const T& val)
        : val_(val)
    {
    }

    nvstd::function<void(T&)> __device__ op()
    {
        return [*this] __device__(T & val) { val *= val_; };
    }

private:
    const T val_;
};

template <typename T> struct DivByVal
{
    DivByVal(const T& val)
        : val_(val)
    {
    }

    nvstd::function<void(T&)> __device__ op()
    {
        return [*this] __device__(T & val) { val /= val_; };
    }

private:
    const T val_;
};

template <typename T> void cu_AddTo(DevPtr<T> image, const T& value)
{
    AddValTo<T> add_op(value);
    cu_Apply(image, add_op);
}

template <typename T> void cu_MultiplyBy(DevPtr<T> image, const T& value)
{
    MulByVal<T> mul_op(value);
    cu_Apply(image, mul_op);
}

template <typename T> void cu_SubtractFrom(DevPtr<T> image, const T& value)
{
    SubValFrom<T> sub_op(value);
    cu_Apply(image, sub_op);
}

template <typename T> void cu_DivideBy(DevPtr<T> image, const T& value)
{
    DivByVal<T> div_op(value);
    cu_Apply(image, div_op);
}

/**
 * Explicit instantiations
 */

#define DECLARE_MATH_OPERATION(type, function)                                \
    template void cu_Apply(DevPtr<type>, const DevPtr<type>, function<type>);

#define DECLARE_MATH_FUNCTION(type, function)                                 \
    template void function(DevPtr<type>, const DevPtr<type>&);

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, AddTo)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_AddTo)

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, SubFrom)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_SubtractFrom)

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, MulBy)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_MultiplyBy)

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, DivBy)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_DivideBy)

#undef DECLARE_MATH_OPERATION
#define DECLARE_MATH_OPERATION(type, function)                                \
    template void cu_Apply(DevPtr<type>, function<type>);

#undef DECLARE_MATH_FUNCTION
#define DECLARE_MATH_FUNCTION(type, function)                                 \
    template void function(DevPtr<type>, const type&);

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, AddValTo)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_AddTo)

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, SubValFrom)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_SubtractFrom)

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, MulByVal)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_MultiplyBy)

FOR_EACH_TYPE(DECLARE_MATH_OPERATION, DivByVal)
FOR_EACH_TYPE(DECLARE_MATH_FUNCTION, cu_DivideBy)

} // image
