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

    // Allocates, but doesnt set initial value
    Image<TO> output(nullptr, w_, h_);
    cu_ColorToGray<T, TO>(output, *this);

    return output;
}

template<typename T>
template<typename TO>
Image<TO> Image<T>::asColor() const
{
    assert(channels<T>() == 1);
    assert(channels<TO>() >= 3);

    // Allocates, but doesnt set initial value
    Image<TO> output(nullptr, w_, h_);
    cu_GrayToColor<T, TO>(output, *this);

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

    // Allocates, but doesnt set initial value
    Image<TO> output(nullptr, w_, h_);
    cu_Convert<T, TO>(output, *this);
    return output;
}

template<typename T>
template<typename TO, typename std::enable_if<is_same_type<T, TO>::value && has_0_channels<TO>::value, TO>::type*>
Image<TO> Image<T>::get(const ushort component) const
{
    assert(component < channels<T>());

    Image<TO> output(nullptr, w_, h_);
    cu_GetComponent<T, TO>(output, *this, component);
    return output;
}



template<typename T>
Image<T> Image<T>::resize(const size_t width, const size_t height) const
{
    // Allocates, but doesnt set initial value
    Image<T> output(nullptr, width, height);
    cu_ResizeLinear<T>(output, *this);
    return output;
}

template<typename T>
Image<T> Image<T>::resize(const size_t width, const size_t height, const Image<uchar>& mask) const
{
    // Mask on input resolution
    assert(mask.width() == w_);
    assert(mask.height() == h_);

    // Allocates, but doesnt set initial value
    Image<T> output(nullptr, width, height);
    cu_ResizeLinear<T>(output, *this, mask);
    return output;
}

template<typename T>
void Image<T>::resize(const size_t width, const size_t height)
{
    if (w_ == width && h_ == height)
        return;

    // Allocates, but doesnt set initial value
    Image<T> tmp(nullptr, width, height);
    cu_ResizeLinear<T>(tmp, *this);
    swap(tmp);
}

template<typename T>
void Image<T>::resize(const size_t width, const size_t height, const Image<uchar>& mask)
{
    if (w_ == width && h_ == height)
        return;
    
    // Mask on input resolution
    assert(mask.width() == w_);
    assert(mask.height() == h_);

    // Allocates, but doesnt set initial value
    Image<T> tmp(nullptr, width, height);
    cu_ResizeLinear<T>(tmp, *this, mask);
    swap(tmp);
}

template<typename T>
Image<T> Image<T>::resize(const float factor) const
{
    // Allocates, but doesnt set initial value
    Image<T> output(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(output, *this);
    return output;
}

template<typename T>
Image<T> Image<T>::resize(const float factor, const Image<uchar>& mask) const
{
    // Mask on input resolution
    assert(mask.width() == w_);
    assert(mask.height() == h_);

    // Allocates, but doesnt set initial value
    Image<T> output(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(output, *this, mask);
    return output;
}

template<typename T>
void Image<T>::resize(const float factor)
{
    if (factor == 1.f)
        return;

    // Allocates, but doesnt set initial value
    Image<T> tmp(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(tmp, *this);
    swap(tmp);
}

template<typename T>
void Image<T>::resize(const float factor, const Image<uchar>& mask)
{
    if (factor == 1.f)
        return;

    // Mask on input resolution
    assert(mask.width() == w_);
    assert(mask.height() == h_);
        
    // Allocates, but doesnt set initial value
    Image<T> tmp(nullptr, factor * w_, factor * h_);
    cu_ResizeLinear<T>(tmp, *this, mask);
    swap(tmp);
}

template<typename T>
void Image<T>::mask(const Image<uchar>& mask)
{
    // Mask on input resolution
    assert(mask.width() == w_);
    assert(mask.height() == h_);

    cu_ApplyMask<T>(*this, mask);
}