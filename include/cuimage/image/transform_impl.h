template <typename T>
void Image<T>::setTo(const T& value)
{
    cu_SetTo<T>(*this, value);
}

template <typename T>
template <class Q, typename std::enable_if<!(is_float_type<Q>::value && has_0_channels<Q>::value), Q>::type*>
void Image<T>::setTo(const float& value)
{
    cu_SetTo<T>(*this, make<T>(value));
}

template <typename T>
void Image<T>::threshold(const T& threshold, const T& value)
{
    cu_Threshold<T>(*this, threshold, value);
}

template <typename T>
void Image<T>::threshold(const T& threshold, const T& low, const T& high)
{
    cu_Threshold<T>(*this, threshold, low, high);
}

template <typename T>
void Image<T>::thresholdInv(const T& threshold, const T& value)
{
    cu_ThresholdInv<T>(*this, threshold, value);
}

template <typename T>
void Image<T>::replace(const T& value, const T& with)
{
    cu_Replace<T>(*this, value, with);
}

template <typename T>
void Image<T>::replaceNan(const T& value)
{
    cu_ReplaceNan<T>(*this, value);
}

template <typename T>
T Image<T>::min() const
{
    return cu_Min<T>(*this);
}

template <typename T>
T Image<T>::max() const
{
    return cu_Max<T>(*this);
}

template <typename T>
float Image<T>::sum() const
{
    return cu_Sum<T>(*this);
}

template <typename T>
float Image<T>::mean() const
{
    // First sum all pixel components and then add.
    Image<float> sum(w_, h_, 0.f);
    cu_PixelSum<T, float>(sum, *this);
    return cu_Sum<float>(sum) / (channels<T>() * size()); // Sum of all components divided by number of components
}

template <typename T>
template <class Q, typename std::enable_if<has_0_channels<Q>::value, Q>::type*>
Q Image<T>::median() const
{
    return cu_Median<Q>(*this);
}

template <typename T>
unsigned int Image<T>::valid() const
{
    // First mark valid pixels, then count. Makes sure data type storage is sufficient
    Image<int> cnt(w_, h_, 0);
    cu_MarkValid<T, int>(cnt, *this);
    return static_cast<unsigned int>(cu_Sum<int>(cnt));
}

template <typename T>
unsigned int Image<T>::nonzero() const
{
    Image<int> cnt(w_, h_, 0);
    cu_MarkNonzero<T, int>(cnt, *this);
    return static_cast<unsigned int>(cu_Sum<int>(cnt));
}

template <typename T>
float Image<T>::norm1() const
{
    return cu_Norm1<T>(*this);
}

template <typename T>
float Image<T>::norm2() const
{
    // This cannot be done as usual reducing operation, so first square, then add
    Image<float> norms(w_, h_, 0.f);
    cu_SquareNorm<T, float>(norms, *this);
    return sqrt(cu_Sum<float>(norms));
}
