
template <typename T>
Image<T>::Image()
    : Image(nullptr, 0, 0)
{
}

template <typename T>
Image<T>::Image(const Image& other)
    : Image(nullptr, other.w_, other.h_)
{   
    cudaSafeCall(cudaMalloc(&data_, sizeBytes()));
    cudaSafeCall(cudaMemcpy(data_, other.data_, sizeBytes(), cudaMemcpyDeviceToDevice));
}

template <typename T>
Image<T>::Image(Image&& other)
    : Image(std::move(other.data_), other.w_, other.h_)
{
    other.data_ = nullptr;
}

template <typename T>
Image<T>::Image(T* data, size_t w, size_t h)
    : data_(data), w_(w), h_(h)
{
    if (!data_)
        cudaSafeCall(cudaMalloc(&data_, sizeBytes()));
}

template <typename T>
Image<T>::Image(size_t w, size_t h, const T initVal)
    : Image(nullptr, w, h)
{
    setTo(initVal);
}

template <typename T>
Image<T>::Image(const std::string& fileName)
{
    FileReader f;
    auto devPtr = f.read<T>(fileName, w_, h_, c_);        
    data_ = devPtr.data;
}

template <typename T>
Image<T>& Image<T>::operator=(const Image<T>& other)
{
    // Use copy of other that will be destructed at the end...
    assert(w_ == other.w_);
    assert(h_ == other.h_);
    std::cout << "Copy assignment operator" << std::endl;
    cudaSafeCall(cudaMemcpy(data_, other.data_, w_ * h_ * sizeof(T), cudaMemcpyDeviceToDevice));
    return *this;
}

template <typename T>
Image<T>& Image<T>::operator=(Image<T>&& other)
{
    assert(w_ == other.w_);
    assert(h_ == other.h_);
    std::cout << "Move assignment operator" << std::endl;
    data_ = std::move(other.data_);
    other.data_ = nullptr;
    return *this;
}

template <typename T>
void Image<T>::swap(Image<T>& i1, Image<T>& i2)
{
    // Instead of expensive copy of 1, just swap the internals
    std::swap(i1.data_, i2.data_);
    std::swap(i1.w_, i2.w_);
    std::swap(i1.h_, i2.h_);
}

template <typename T>
Image<T>::~Image()
{
    if (data_)
        cudaSafeCall(cudaFree(data_));

    data_ = nullptr;
}

template <typename T>
bool Image<T>::empty() const
{
    return (bool)data_;
}

template <typename T>
size_t Image<T>::width() const
{
    return w_;
}

template <typename T>
size_t Image<T>::height() const
{
    return h_;
}

template <typename T>
size_t Image<T>::size() const
{
    return w_ * h_;
}

template <typename T>
size_t Image<T>::sizeBytes() const
{
    return size() * sizeof(T);
}