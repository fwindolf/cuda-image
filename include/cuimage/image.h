#pragma once

#include "cuimage/cuda.h"
#include "cuimage/file/reader.h"
#include "cuimage/file/writer.h"
#include "cuimage/operations.h"
#include "cuimage/visualize/visualize.h"

#include <cassert>
#include <limits>
#include <string>

namespace cuimage
{

static int id = 0;

// Forward declare friend operators
template <typename T> class Image;
template <typename T> Image<T> operator+(Image<T> a, const Image<T>& b);
template <typename T> Image<T>& operator+=(Image<T>& a, const Image<T>& b);
template <typename T> Image<T> operator-(Image<T> a, const Image<T>& b);
template <typename T> Image<T>& operator-=(Image<T>& a, const Image<T>& b);
template <typename T> Image<T> operator*(Image<T> a, const Image<T>& b);
template <typename T> Image<T>& operator*=(Image<T>& a, const Image<T>& b);
template <typename T> Image<T> operator/(Image<T> a, const Image<T>& b);
template <typename T> Image<T>& operator/=(Image<T>& a, const Image<T>& b);

template <typename T> Image<T> operator+(Image<T> a, const T& v);
template <typename T> Image<T>& operator+=(Image<T>& a, const T& v);
template <typename T> Image<T> operator-(Image<T> a, const T& v);
template <typename T> Image<T>& operator-=(Image<T>& a, const T& v);
template <typename T> Image<T> operator*(Image<T> a, const T& v);
template <typename T> Image<T>& operator*=(Image<T>& a, const T& v);
template <typename T> Image<T> operator/(Image<T> a, const T& v);
template <typename T> Image<T>& operator/=(Image<T>& a, const T& v);

// Forward declare DevPtr of that type
template <typename T> class DevPtr;

template <typename T> class Image
{
public:
    Image();

    // Need copy constructor for math functions
    Image(const Image& other);

    Image(Image&& other);

    Image(DevPtr<T>&& devPtr);

    Image(T* data, size_t w, size_t h);

    Image(const std::string& fileName);

    Image(size_t w, size_t h, const T& initVal = make<T>(0.f));

    template <class Q = T,
        typename std::enable_if<!(is_float_type<Q>::value
                                    && has_0_channels<Q>::value),
            Q>::type* = nullptr>
    Image(size_t w, size_t h, const float initVal);

    ~Image();

    /**
     * Assignment
     */

    Image& operator=(const Image&);

    Image& operator=(Image&&);

    void swap(Image& other);

    void realloc(size_t w, size_t h);

    void copyTo(Image& other) const;

    void copyFrom(const Image& other);

    /**
     * Accessors
     */

    bool empty() const;

    size_t width() const;

    size_t height() const;

    size_t size() const;

    size_t sizeBytes() const;

    T* data() const;

    T at(const size_t idx) const;

    T at(const size_t x, const size_t y) const;

    T* download() const;

    T* download(const size_t first_n) const;

    void upload(const T* hdata, const size_t width, const size_t height);

    bool save(const std::string& fileName) const;

    /**
     * Conversion to DevPtr
     */
    operator DevPtr<T>() const;

    DevPtr<T> devPtr() const;

    /**
     * Visualization
     */
    template <VisType V> void createWindow(const std::string windowName);

    void closeWindow(bool force = false);

    void show(bool wait = false) const;

    template <VisType V>
    void show(const std::string windowName, bool close = true);

    void visualize() const;

    void print() const;

    /**
     * Data Transformations
     */
    void setTo(const T& value);

    template <class Q = T,
        typename std::enable_if<!(is_float_type<Q>::value
                                    && has_0_channels<Q>::value),
            Q>::type* = nullptr>
    void setTo(const float& value);

    void threshold(const T& threshold, const T& value);

    void threshold(const T& threshold, const T& low, const T& high);

    void thresholdInv(const T& threshold, const T& value);

    void replace(const T& value, const T& with);

    void replaceNan(const T& value);

    Image abs();

    Image square();

    /**
     * Reductions
     */
    T min() const;

    T max() const;

    float sum() const;

    float sqSum() const;

    float mean() const;

    float stdDev() const;

    // Only possible for 0-channel types
    template <class Q = T,
        typename std::enable_if<has_0_channels<Q>::value, Q>::type* = nullptr>
    Q median() const;

    unsigned int valid() const;

    unsigned int nan() const;

    unsigned int nonzero() const;

    float norm1() const;

    float norm2() const;

    /**
     * Conversions
     */
    template <typename TO> Image<TO> asGray() const;

    template <typename TO> Image<TO> asColor() const;

    template <typename TO> Image<TO> reinterpret_as();

    template <typename TO> Image<TO> as() const;

    // Only possible for base type
    template <typename TO,
        typename std::enable_if<is_same_type<T, TO>::value
                && has_0_channels<TO>::value,
            TO>::type* = nullptr>
    Image<TO> get(const ushort component) const;

    /*
    // TODO: Modes
    // TODO: Single filter method
    
    // Filters
    Image blur(int radius, float sigma) const;

    Image bilateral(int radius, float sigmaSpace, float sigmaColor) const;
    */
    /**
     * Resizing
     */
    void resize(const size_t width, const size_t height,
        const ResizeMode mode = LINEAR);

    void resize(const size_t width, const size_t height,
        const Image<uchar>& mask, const ResizeMode mode = LINEAR);

    void resize(const float factor, const ResizeMode mode = LINEAR);

    void resize(const float factor, const Image<uchar>& mask,
        const ResizeMode mode = LINEAR);

    Image resized(const size_t width, const size_t height,
        const ResizeMode mode = LINEAR) const;

    Image resized(const size_t width, const size_t height,
        const Image<uchar>& mask, const ResizeMode mode = LINEAR) const;

    Image resized(const float factor, const ResizeMode mode = LINEAR) const;

    Image resized(const float factor, const Image<uchar>& mask,
        const ResizeMode mode = LINEAR) const;

    void mask(const Image<uchar>& mask);

    /**
     * Math operations
     */

    Image& add(const Image& other);

    Image& subtract(const Image& other);

    Image& multiply(const Image& other);

    Image& divide(const Image& other);

    friend Image<T> operator+<>(Image a, const Image& b);

    friend Image<T>& operator+=<>(Image& a, const Image& b);

    friend Image<T> operator-<>(Image a, const Image& b);

    friend Image<T>& operator-=<>(Image& a, const Image& b);

    friend Image<T> operator*<>(Image a, const Image& b);

    friend Image<T>& operator*=<>(Image& a, const Image& b);

    friend Image<T> operator/<>(Image a, const Image& b);

    friend Image<T>& operator/=<>(Image& a, const Image& b);

    Image& add(const T& value);

    Image& subtract(const T& value);

    Image& multiply(const T& value);

    Image& divide(const T& value);

    friend Image<T> operator+<>(Image a, const T& b);

    friend Image<T>& operator+=<>(Image& a, const T& b);

    friend Image<T> operator-<>(Image a, const T& b);

    friend Image<T>& operator-=<>(Image& a, const T& b);

    friend Image<T> operator*<>(Image a, const T& b);

    friend Image<T>& operator*=<>(Image& a, const T& b);

    friend Image<T> operator/<>(Image a, const T& b);

    friend Image<T>& operator/=<>(Image& a, const T& b);

private:
    DevPtr<T> read(const std::string& fileName);

    std::unique_ptr<Visualizer> vis_;

    T* data_; // contains cuda type data
    size_t w_, h_, c_ = channels<T>();
};

/**
 * Implementation
 */

#include "image/base_impl.h"
#include "image/conversion_impl.h"
#include "image/math_impl.h"
#include "image/transform_impl.h"
#include "image/visualization_impl.h"

} // image