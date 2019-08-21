
template <typename T, typename TO>
bool copyToArray(const T* data, cudaArray_t array, const size_t w,
    const size_t h, const float fillWith = 1.f)
{
    cudaError_t status;
    if (std::is_same<T, TO>())
    {
        status = cudaMemcpyToArray(
            array, 0, 0, data, w * h * sizeof(TO), cudaMemcpyDeviceToDevice);
        return (status == cudaSuccess);
    }

    assert((float)sizeof(TO) / sizeof(T) < 2);

    // T either has more channels than TO -> leave out channels
    //   or one less channel than TO -> add one channel with <fillWith>
    TO* tmp;
    status = cudaMalloc(&tmp, w * h * sizeof(TO));
    if (status != cudaSuccess)
        return false;

    status = cudaMemset(tmp, fillWith, w * h * sizeof(TO));
    if (status != cudaSuccess)
        return false;

    // Copy every element with pitch (width is 1 element, height is w * h)
    status = cudaMemcpy2D(tmp, sizeof(TO), data, sizeof(T), sizeof(T), w * h,
        cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
        return false;

    return copyToArray<TO, TO>(tmp, array, w, h);
}