#pragma once

#include <cassert>
#include <string>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "cuimage/cuda.h"
#include "cuimage/operations.h"

#include "cuimage/file/reader.h"

#include "cuimage/visualize/visualize.h"

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

template <typename T>
class Image
{
public:
    Image();

    // Need copy constructor for math functions
    Image(const Image& other); 

    Image(Image&& other);

    Image(T* data, size_t w, size_t h);

    Image(const std::string& fileName);

    Image(size_t w, size_t h, const T initVal = make<T>(0.f));

    Image& operator=(const Image&);

    Image& operator=(Image&&);

    void swap(Image& i1, Image& i2);

    ~Image();

    bool empty() const;

    size_t width() const;

    size_t height() const;

    size_t size() const;

    size_t sizeBytes() const;

    T* data() const;

    /**
     * Conversion to DevPtr
     */
    operator DevPtr<T>() const;

    DevPtr<T> devPtr() const;
    
    /**
     * Visualization
     */
    void setVisualizationStrategy(const VisType type);

    void show(const std::string windowName);

    void show(const std::string windowName, const VisType type);

    void print() const;

    /**
     * Data Transformations
     */
    void setTo(const T& value);

    void threshold(const T& threshold, const T& value);

    void replace(const T& value, const T& with);

    void replaceNan(const T& value);

        
    /** 
     * Reductions
     */
    T min() const;

    T max() const;

    float sum() const;

    float norm1() const;

    float norm2() const;

    /**
     * Conversions
     */
    template<typename TO>
    Image<TO> asGray() const;

    template<typename TO>
    Image<TO> asColor() const;

    template<typename TO>
    Image<TO> reinterpret_as();

    template<typename TO>
    Image<TO> as() const;

    /**
     * Resizing
     */
    Image resize(const size_t width, const size_t height) const;
    
    Image resize(const size_t width, const size_t height, const Image<uchar>& mask) const;

    void resize(const size_t width, const size_t height, const Image<uchar>& mask);

    void resize(const size_t width, const size_t height);

    Image resize(const float factor) const;

    Image resize(const float factor, const Image<uchar>& mask) const;

    void resize(const float factor);

    void resize(const float factor, const Image<uchar>& mask);

    void mask(const Image<uchar>& mask);

    /**
     * Math operations
     */

    Image& add(const Image& other);

    Image& subtract(const Image& other);

    Image& multiply(const Image& other);

    Image& divide(const Image& other);

    friend Image<T> operator+ <>(Image a, const Image& b); 

    friend Image<T>& operator+= <>(Image& a, const Image& b);

    friend Image<T> operator- <>(Image a, const Image& b);
    
    friend Image<T>& operator-= <>(Image& a, const Image& b); 
    
    friend Image<T> operator* <>(Image a, const Image& b);
    
    friend Image<T>& operator*= <>(Image& a, const Image& b); 
    
    friend Image<T> operator/ <>(Image a, const Image& b); 
    
    friend Image<T>& operator/= <>(Image& a, const Image& b); 

    Image& add(const T& value);

    Image& subtract(const T& value);

    Image& multiply(const T& value);

    Image& divide(const T& value);

    friend Image<T> operator+ <>(Image a, const T& b); 

    friend Image<T>& operator+= <>(Image& a, const T& b);

    friend Image<T> operator- <>(Image a, const T& b);
    
    friend Image<T>& operator-= <>(Image& a, const T& b); 
    
    friend Image<T> operator* <>(Image a, const T& b);
    
    friend Image<T>& operator*= <>(Image& a, const T& b); 
    
    friend Image<T> operator/ <>(Image a, const T& b); 
    
    friend Image<T>& operator/= <>(Image& a, const T& b); 
    
private:
    std::unique_ptr<Visualizer> vis_;
    VisType visType_ = NONE;

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