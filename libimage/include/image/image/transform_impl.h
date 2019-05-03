template <typename T>
void Image<T>::setTo(const T& value)
{
    cu_SetTo<T>(*this, value);
}

template <typename T>
void Image<T>::threshold(const T& threshold, const T& value)
{
    cu_Threshold<T>(*this, threshold, value);
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
float Image<T>::norm1() const
{
    return cu_Norm1<T>(*this);
}

template <typename T>
float Image<T>::norm2() const
{
    // This cannot be done as usual reducing operation, so first square, then add
    Image<T> tmp = (*this) * (*this);
    return sqrt(cu_Sum(tmp));
}
