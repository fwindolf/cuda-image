
template <typename T>
Image<T>::Image()
    : Image(nullptr, 0, 0)
{
}

template <typename T>
Image<T>::Image(const Image& other)
    : Image(nullptr, 0, 0)
{
    copyFrom(other);
}

template <typename T>
Image<T>::Image(Image&& other)
    : Image(std::move(other.data_), other.w_, other.h_)
{
    other.data_ = nullptr;
}

template <typename T>
Image<T>::Image(DevPtr<T>&& devPtr)
    : Image(devPtr.data, devPtr.width, devPtr.height)
{
    // Claim ownership of data
    devPtr.data = nullptr;
}

template <typename T>
Image<T>::Image(T* data, size_t w, size_t h)
    : data_(data)
    , w_(w)
    , h_(h)
{
    // Allocate new memory if data was nullptr
    if (!data_)
        cudaSafeCall(cudaMalloc(&data_, sizeBytes()));
}

template <typename T>
Image<T>::Image(size_t w, size_t h, const T& initVal)
    : Image(nullptr, w, h)
{
    setTo(initVal);
}

template <typename T>
template <class Q,
    typename std::enable_if<
        !(is_float_type<Q>::value && has_0_channels<Q>::value), Q>::type*>
Image<T>::Image(size_t w, size_t h, const float initVal)
    : Image(nullptr, w, h)
{
    setTo(initVal);
}

template <typename T>
inline DevPtr<T> Image<T>::read(const std::string& fileName)
{
    FileReader<T> f;
    return f.read(fileName);
}

template <typename T>
Image<T>::Image(const std::string& fileName)
    : Image<T>(read(fileName))
{
}

template <typename T> Image<T>::~Image()
{
    if (data_)
        cudaSafeCall(cudaFree(data_));

    data_ = nullptr;
}

template <typename T> Image<T>& Image<T>::operator=(const Image<T>& other)
{
    if (empty())
        *this = Image(other.w_, other.h_);

    assert(w_ == other.w_);
    assert(h_ == other.h_);
    cudaSafeCall(cudaMemcpy(
        data_, other.data_, w_ * h_ * sizeof(T), cudaMemcpyDeviceToDevice));
    return *this;
}

template <typename T> Image<T>& Image<T>::operator=(Image<T>&& other)
{
    if (data_ == nullptr && w_ == 0 && h_ == 0)
        realloc(other.w_, other.h_);

    assert(w_ == other.w_);
    assert(h_ == other.h_);
    data_ = std::move(other.data_);
    other.data_ = nullptr;
    return *this;
}

template <typename T> void Image<T>::swap(Image<T>& other)
{
    // Instead of expensive copy of 1, just swap the internals
    std::swap(data_, other.data_);
    std::swap(w_, other.w_);
    std::swap(h_, other.h_);
    vis_.release(); // Just in case the resolution changed, ...
}

template <typename T> void Image<T>::realloc(size_t w, size_t h)
{
    w_ = w;
    h_ = h;
    if (data_)
        cudaSafeCall(cudaFree(data_));

    cudaSafeCall(cudaMalloc(&data_, sizeBytes()));
}

template <typename T> void Image<T>::copyTo(Image& other) const
{
    other.copyFrom(*this);
}

template <typename T> void Image<T>::copyFrom(const Image& other)
{
    // If dimensions change, reallocate (usually quite as fast as extending
    // memory)
    if (empty() || other.width() != w_ || other.height() != h_)
        realloc(other.width(), other.height());

    cudaSafeCall(cudaMemcpy(
        data_, other.data(), sizeBytes(), cudaMemcpyDeviceToDevice));
}

template <typename T> bool Image<T>::empty() const
{
    return (data_ == nullptr);
}

template <typename T> size_t Image<T>::width() const { return w_; }

template <typename T> size_t Image<T>::height() const { return h_; }

template <typename T> size_t Image<T>::size() const { return w_ * h_; }

template <typename T> size_t Image<T>::sizeBytes() const
{
    return size() * sizeof(T);
}

template <typename T> T* Image<T>::data() const { return data_; }

template <typename T> T Image<T>::at(const size_t idx) const
{
    T out;
    cudaSafeCall(
        cudaMemcpy(&out, &data_[idx], sizeof(T), cudaMemcpyDeviceToHost));
    return out;
}

template <typename T> T Image<T>::at(const size_t x, const size_t y) const
{
    T out;
    cudaSafeCall(cudaMemcpy(
        &out, &data_[x + y * w_], sizeof(T), cudaMemcpyDeviceToHost));
    return out;
}

template <typename T> T* Image<T>::download() const
{
    return download(size());
}

template <typename T> T* Image<T>::download(const size_t first_n) const
{
    assert(first_n <= size());
    T* out = new T[first_n];
    cudaSafeCall(
        cudaMemcpy(out, data_, first_n * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
}

template <typename T>
void Image<T>::upload(const T* hdata, const size_t width, const size_t height)
{
    if (data_ == nullptr && w_ == 0 && h_ == 0)
        realloc(width, height);

    assert(w_ == width);
    assert(h_ == height);
    cudaSafeCall(
        cudaMemcpy(data_, hdata, sizeBytes(), cudaMemcpyHostToDevice));
}

template <typename T> bool Image<T>::save(const std::string& fileName)
{
    assert(data_);

    FileWriter<T> f(fileName, w_, h_);
    return f.write(download());
}