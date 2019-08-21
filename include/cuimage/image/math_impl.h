
template <typename T> Image<T>& Image<T>::add(const Image<T>& other)
{
    cu_AddTo<T>(*this, other);
    return *this;
}

template <typename T> Image<T>& Image<T>::subtract(const Image<T>& other)
{
    cu_SubtractFrom<T>(*this, other);
    return *this;
}

template <typename T> Image<T>& Image<T>::multiply(const Image<T>& other)
{
    cu_MultiplyBy<T>(*this, other);
    return *this;
}

template <typename T> Image<T>& Image<T>::divide(const Image<T>& other)
{
    cu_DivideBy<T>(*this, other);
    return *this;
}

template <typename T> Image<T>& Image<T>::add(const T& value)
{
    cu_AddTo<T>(*this, value);
    return *this;
}

template <typename T> Image<T>& Image<T>::subtract(const T& value)
{
    cu_SubtractFrom<T>(*this, value);
    return *this;
}

template <typename T> Image<T>& Image<T>::multiply(const T& value)
{
    cu_MultiplyBy<T>(*this, value);
    return *this;
}

template <typename T> Image<T>& Image<T>::divide(const T& value)
{
    cu_DivideBy<T>(*this, value);
    return *this;
}

/**
 * Template friends are tricky to declare outside class...
 * https://isocpp.org/wiki/faq/templates#template-friends
 */
template <typename T> Image<T> operator+(Image<T> a, const Image<T>& b)
{
    a += b;
    return std::move(a);
}

template <typename T> Image<T>& operator+=(Image<T>& a, const Image<T>& b)
{
    a.add(b);
    return a;
}

template <typename T> Image<T> operator-(Image<T> a, const Image<T>& b)
{
    a -= b;
    return std::move(a);
}

template <typename T> Image<T>& operator-=(Image<T>& a, const Image<T>& b)
{
    a.subtract(b);
    return a;
}

template <typename T> Image<T> operator*(Image<T> a, const Image<T>& b)
{
    a *= b;
    return std::move(a);
}

template <typename T> Image<T>& operator*=(Image<T>& a, const Image<T>& b)
{
    a.multiply(b);
    return a;
}

template <typename T> Image<T> operator/(Image<T> a, const Image<T>& b)
{
    a /= b;
    return std::move(a);
}

template <typename T> Image<T>& operator/=(Image<T>& a, const Image<T>& b)
{
    a.divide(b);
    return a;
}

template <typename T> Image<T> operator+(Image<T> a, const T& b)
{
    a += b;
    return std::move(a);
}

template <typename T> Image<T>& operator+=(Image<T>& a, const T& b)
{
    a.add(b);
    return a;
}

template <typename T> Image<T> operator-(Image<T> a, const T& b)
{
    a -= b;
    return std::move(a);
}

template <typename T> Image<T>& operator-=(Image<T>& a, const T& b)
{
    a.subtract(b);
    return a;
}

template <typename T> Image<T> operator*(Image<T> a, const T& b)
{
    a *= b;
    return std::move(a);
}

template <typename T> Image<T>& operator*=(Image<T>& a, const T& b)
{
    a.multiply(b);
    return a;
}

template <typename T> Image<T> operator/(Image<T> a, const T& b)
{
    a /= b;
    return std::move(a);
}

template <typename T> Image<T>& operator/=(Image<T>& a, const T& b)
{
    a.divide(b);
    return a;
}
