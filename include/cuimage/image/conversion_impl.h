template <typename T>
inline Image<T>::operator DevPtr<T>() const
{
    return DevPtr<T>(data_, w_, h_);
}

template <typename T>
DevPtr<T> Image<T>::devPtr() const
{
    return DevPtr<T>(data_, w_, h_);
}


template<typename T>
template<typename TO>
Image<TO> Image<T>::asGray() const
{
    assert(channels<T>() >= 3);
    assert(channels<TO>() == 1);

    Image<TO> output(nullptr, w_, h_);
    cu_ColorToGray<T>(output, *this);

    return output;
}

template<typename T>
template<typename TO>
Image<TO> Image<T>::asColor() const
{
    assert(channels<T>() == 1);
    assert(channels<TO>() >= 3);

    Image<TO> output(nullptr, w_, h_);
    cu_GrayToColor<T>(output, *this);

    return output;
}

template<typename T>
template<typename TO>
Image<TO> Image<T>::reinterpret_as()
{
    assert(sizeof(T) == sizeof(TO)); // allows also for int -> float 
            
    Image<TO> output(reinterpret_cast<TO*>(data_), w_, h_);
    data_ = nullptr;
    return output;
}

template<typename T>
template<typename TO>
Image<TO> Image<T>::as() const
{
    assert(channels<T>() == channels<TO>());

    Image<TO> output(nullptr, w_, h_);
    cu_Convert<T>(output, *this);
    return output;
}

template<typename T>
Image<T> Image<T>::resize(const size_t width, const size_t height) const
{
    Image<T> output(nullptr, width, height);
    cu_ResizeLinear<T>(output, *this);
    return output;
}

template<typename T>
Image<T> Image<T>::resize(const size_t width, const size_t height, const Image<uchar>& mask) const
{
    Image<T> output(nullptr, width, height);
    cu_ResizeLinear<T>(output, *this, mask);
    return output;
}

template<typename T>
void Image<T>::resize(const size_t width, const size_t height)
{
    Image<T> tmp(nullptr, width, height);
    cu_ResizeLinear<T>(tmp, *this);
    swap(tmp);
}

template<typename T>
void Image<T>::resize(const size_t width, const size_t height, const Image<uchar>& mask)
{
    Image<T> tmp(nullptr, width, height);
    cu_ResizeLinear<T>(tmp, *this, mask);
    swap(tmp);
}

template<typename T>
Image<T> Image<T>::resize(const float factor) const
{
    Image<T> output(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(output, *this);
    return output;
}

template<typename T>
Image<T> Image<T>::resize(const float factor, const Image<uchar>& mask) const
{
    Image<T> output(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(output, *this, mask);
    return output;
}

template<typename T>
void Image<T>::resize(const float factor)
{
    Image<T> tmp(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(tmp, *this);
    swap(tmp);
}

template<typename T>
void Image<T>::resize(const float factor, const Image<uchar>& mask)
{
    Image<T> tmp(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(tmp, *this, mask);
    swap(tmp);
}

template<typename T>
void Image<T>::mask(const Image<uchar>& mask)
{
    cu_ApplyMask<T>(*this, mask);
}