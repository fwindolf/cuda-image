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

template <typename T>
Image<T> Image<T>::resized(const size_t width, const size_t height, const ResizeMode mode) const
{
    Image<T> output(nullptr, width, height);
    switch (mode)
    {
    case NEAREST:
        cu_ResizeNearest<T>(output, *this);
        break;
    case LINEAR:
        cu_ResizeLinear<T>(output, *this);
        break;
    case LINEAR_VALID:
        cu_ResizeLinearValid<T>(output, *this);        
        break;
    case LINEAR_NONZERO:
        throw std::runtime_error("NonZero resizing is only possible if a mask is provided!");
    default:
        throw std::runtime_error("Invalid mode for resizing!");
    }
    return output;
}

template <typename T>
Image<T> Image<T>::resized(const size_t width, const size_t height, const Image<uchar>& mask, const ResizeMode mode) const
{
    // Mask on input resolution
    assert(mask.width() == w_);
    assert(mask.height() == h_);

    Image<T> output(nullptr, width, height);
    switch (mode)
    {
    case NEAREST:
        cu_ResizeNearest<T>(output, *this, mask);
        break;
    case LINEAR:
        cu_ResizeLinear<T>(output, *this, mask);
        break;
    case LINEAR_VALID:
        cu_ResizeLinearValid<T>(output, *this, mask);
        break;
    case LINEAR_NONZERO:
        cu_ResizeLinearValid<T>(output, *this, mask);
        break;
    default:
        throw std::runtime_error("Invalid mode for resizing!");
    }

    return output;
}

template<typename T>
Image<T> Image<T>::resized(const float factor, const ResizeMode mode) const
{
    const size_t width = factor * w_;
    const size_t height = factor * h_;
    return resized(width, height, mode);
}

template<typename T>
Image<T> Image<T>::resized(const float factor, const Image<uchar>& mask, const ResizeMode mode) const
{
    const size_t width = factor * w_;
    const size_t height = factor * h_;
    return resized(width, height, mask, mode);
}

template<typename T>
void Image<T>::resize(const size_t width, const size_t height, const ResizeMode mode)
{
    if (w_ == width && h_ == height)
        return;

    auto tmp = resized(width, height, mode);
    swap(tmp);
}

template<typename T>
void Image<T>::resize(const size_t width, const size_t height, const Image<uchar>& mask, const ResizeMode mode)
{
    if (w_ == width && h_ == height)
        return;
    
    auto tmp = resized(width, height, mask, mode);
    swap(tmp);
}


template<typename T>
void Image<T>::resize(const float factor, const ResizeMode mode)
{
    if (factor == 1.f)
        return;

    auto tmp = resized(factor, mode);
    std::cout << "Resized: (" << tmp.width() << "x" << tmp.height() << ")" << std::endl;
    swap(tmp);
}

template<typename T>
void Image<T>::resize(const float factor, const Image<uchar>& mask, const ResizeMode mode)
{
    if (factor == 1.f)
        return;

    auto tmp = resized(factor, mask, mode);
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