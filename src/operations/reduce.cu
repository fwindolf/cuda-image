#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/utils.h"
#include "cuimage/operations/reduce_cu.h"

#include <nvfunctional>

namespace cuimage
{

/**
 * Templated shared memory, from https://stackoverflow.com/a/49224531/4658360
 */
template <typename T> __device__ T* shared_mem()
{
    extern __shared__ int memory[];
    return reinterpret_cast<T*>(memory);
}

template <typename T, typename Op>
__global__ void g_Reduce(T* g_idata, T* g_odata, int n, Op operation)
{
    T* sdata = shared_mem<T>();

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // From global to shared memory, already use operation to reduce
    T tmp = operation.initVal();
    if (i < n)
        operation.getOp()(tmp, g_idata[i]);
    sdata[tid] = tmp;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            // Work on tmp to reduce shared memory hits
            operation.getOp()(tmp, sdata[tid + s]);
            sdata[tid] = tmp;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = tmp;
    }
}

template <typename T, typename Op> T cu_Reduce(DevPtr<T> image, Op operation)
{
    // Calculate number of blocks and threads
    size_t numElements = image.width * image.height;
    size_t blockSize = 128;

    T* idata;
    T* odata;
    cudaSafeCall(cudaMalloc(&idata, numElements * sizeof(T)));
    cudaSafeCall(cudaMalloc(&odata, numElements * sizeof(T)));
    cudaSafeCall(cudaMemcpy(
        idata, image.data, numElements * sizeof(T), cudaMemcpyDeviceToDevice));

    int m = numElements;
    while (true)
    {
        dim3 block = dim3(blockSize, 1, 1);
        dim3 grid = dim3((m + block.x - 1) / block.x, 1, 1);

        g_Reduce<T, Op><<<grid, block, blockSize * sizeof(T)>>>(
            idata, odata, m, operation);

        cudaCheckLastCall();
#ifndef DNDEBUG
        cudaSafeCall(cudaDeviceSynchronize());
#endif

        m = (m + blockSize - 1) / blockSize;
        if (m <= 1)
            break;

        std::swap(idata, odata);
    }

    T result;
    cudaSafeCall(
        cudaMemcpy(&result, odata, sizeof(T), cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(idata));
    cudaSafeCall(cudaFree(odata));
    return result;
}

template <typename T> struct MinOp
{
    nvstd::function<void(T&, T&)> __device__ getOp()
    {
        return [] __device__(T & l, T & r) {
            if (isvalid(r))
            {
                l = min(l, r);
            }
        };
    }

    T __device__ initVal() const { return initVal_; }

private:
    T initVal_ = Limits<T>::max();
};

template <typename T> T cu_Min(DevPtr<T> image)
{
    MinOp<T> minOp;
    return cu_Reduce(image, minOp);
}

template <typename T> struct MaxOp
{
    nvstd::function<void(T&, T&)> __device__ getOp()
    {
        return [] __device__(T & l, T & r) {
            if (isvalid(r))
                l = max(l, r);
        };
    }

    T __device__ initVal() const { return initVal_; }

private:
    T initVal_ = Limits<T>::min();
};

template <typename T> T cu_Max(DevPtr<T> image)
{
    MaxOp<T> maxOp;
    return cu_Reduce(image, maxOp);
}

template <typename T> struct SumOp
{
    nvstd::function<void(T&, T&)> __device__ getOp()
    {
        return [] __device__(T & l, T & r) {
            if (isvalid(r))
                l += r;
        };
    }

    T __device__ initVal() const { return initVal_; }

private:
    T initVal_ = make<T>(0);
};

template <typename T> float cu_Sum(DevPtr<T> image)
{
    SumOp<T> sumOp;
    return sum(cu_Reduce(image, sumOp));
}

template <typename T> struct ValidOp
{
    nvstd::function<void(T&, T&)> __device__ getOp()
    {
        return [] __device__(T & l, T & r) {
            if (isvalid(r))
                l += make<T>(1.f);
        };
    }

    T __device__ initVal() const { return initVal_; }

private:
    T initVal_ = make<T>(0);
};

template <typename T> unsigned int cu_Valid(DevPtr<T> image)
{
    ValidOp<T> validOp;
    auto res = cu_Reduce(image, validOp);
    std::cout << "Valid: " << res << std::endl;
    return static_cast<unsigned int>(min(res));
}

template <typename T> struct NonZeroOp
{
    nvstd::function<void(T&, T&)> __device__ getOp()
    {
        return [] __device__(T & l, T & r) {
            if (isvalid(r) && !iszero(r))
                l += make<T>(1.f);
        };
    }

    T __device__ initVal() const { return initVal_; }

private:
    T initVal_ = make<T>(0);
};

template <typename T> unsigned int cu_Nonzero(DevPtr<T> image)
{
    NonZeroOp<T> nonzeroOp;
    return static_cast<unsigned int>(min(cu_Reduce(image, nonzeroOp)));
}

template <typename T> struct AbsOp
{
    nvstd::function<void(T&, T&)> __device__ getOp()
    {
        return [] __device__(T & l, T & r) {
            if (isvalid(r))
                l += abs(r);
        };
    }

    T __device__ initVal() const { return initVal_; }

private:
    T initVal_ = make<T>(0);
};

template <typename T> float cu_Norm1(DevPtr<T> image)
{
    AbsOp<T> norm1Op;
    return sum(cu_Reduce(image, norm1Op));
}

/**
 * Explicit instantiation
 */
#define DECLARE_REDUCE_OPERATION(type, operation)                             \
    template type cu_Reduce(DevPtr<type>, operation<type>);

#define DECLARE_REDUCE_FUNCTION(type, function)                               \
    template type function(DevPtr<type>);

FOR_EACH_TYPE(DECLARE_REDUCE_OPERATION, MinOp);
FOR_EACH_TYPE(DECLARE_REDUCE_FUNCTION, cu_Min);

FOR_EACH_TYPE(DECLARE_REDUCE_OPERATION, MaxOp);
FOR_EACH_TYPE(DECLARE_REDUCE_FUNCTION, cu_Max);

#undef DECLARE_REDUCE_FUNCTION
#define DECLARE_REDUCE_FUNCTION(type, function)                               \
    template float function(DevPtr<type>);

FOR_EACH_TYPE(DECLARE_REDUCE_OPERATION, SumOp);
FOR_EACH_TYPE(DECLARE_REDUCE_FUNCTION, cu_Sum);

FOR_EACH_TYPE(DECLARE_REDUCE_OPERATION, AbsOp);
FOR_EACH_TYPE(DECLARE_REDUCE_FUNCTION, cu_Norm1);

#undef DECLARE_REDUCE_FUNCTION
#define DECLARE_REDUCE_FUNCTION(type, function)                               \
    template unsigned int function(DevPtr<type>);

FOR_EACH_TYPE(DECLARE_REDUCE_OPERATION, ValidOp);
FOR_EACH_TYPE(DECLARE_REDUCE_FUNCTION, cu_Valid);

FOR_EACH_TYPE(DECLARE_REDUCE_OPERATION, NonZeroOp);
FOR_EACH_TYPE(DECLARE_REDUCE_FUNCTION, cu_Nonzero);

} // image
