/**
 * @file   visualize.h
 * @brief  Class to visualize different types of images
 * @author Florian Windolf
 */
#pragma once

#include <thread>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include <cuda_gl_interop.h>

#include "image/cuda.h"

#include "types.h"
#include "shaders/default.h"
#include "shaders/ambient.h"
#include "kernels.h"

namespace image
{

/**
 * @class Visualizer
 * @brief Show images
 */

class Visualizer
{
public:
    virtual ~Visualizer(){};
    
    virtual void show(const std::string name, void* data, const size_t dataSize, const bool wait = false) = 0;

    virtual VisType type() const = 0;

protected:
    virtual void render_() = 0;

    virtual void bindTexture_(void* data, const size_t dataSize) = 0;
};

/**
 * @class VisualizerBase
 * @brief Stores common data members 
 * Needed for GlTexture which cannot be accessed in specialization otherways
 */
class VisualizerBase : public Visualizer
{
public:
    VisualizerBase(const size_t w, const size_t h);

    virtual void show(const std::string name, void* data, const size_t dataSize, const bool wait = false) override;

    virtual VisType type() const;

private:
    int run(const std::string name, void* data, const size_t dataSize);

protected:
    template <typename T, typename TO>
    bool copyToArray(const T* data, cudaArray_t array, const size_t w, const size_t h, const float fillWith = 1.f);

    virtual void render_();

    virtual void bindTexture_(void* data, const size_t dataSize);

    virtual bool initTexture_(const size_t dataSize);

    virtual bool copyToTexture_(const void* data, cudaArray_t array, const size_t dataSize);
    
    const size_t w_, h_;

    pangolin::GlTextureCudaArray texture_;

    std::thread runThread_;
};

/**
 * @class TypedVisualizer
 * @brief Show images in a certain way
 */
template <VisType T>
class TypedVisualizer : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h);

    ~TypedVisualizer(){};

    virtual VisType type() const override;

protected:
    virtual void bindTexture_(void* data, const size_t dataSize) override;

    virtual bool initTexture_(const size_t dataSize);

    virtual bool copyToTexture_(const void* data, cudaArray_t array, const size_t dataSize);
};


/**
 * Template definition for VisualizerBase
 */
template <typename T, typename TO>
inline bool VisualizerBase::copyToArray(const T* data, cudaArray_t array, const size_t w, const size_t h, const float fillWith)
{
    cudaError_t status;
    if (std::is_same<T, TO>())
    {
        status = cudaMemcpyToArray(array, 0, 0, data, w * h * sizeof(TO), cudaMemcpyDeviceToDevice);
        return (status == cudaSuccess);
    }

    assert((float)sizeof(TO) / sizeof(T) < 2);

    // T either has more channels than TO -> leave out channels
    //   or one less channel than TO -> add one channel with <fillWith>
    TO* tmp;
    status = cudaMalloc(&tmp, w * h * sizeof(TO));
    if (status != cudaSuccess)
        return false;

    status = cudaMemset(tmp, fillWith, w * h * sizeof(TO));
    if (status != cudaSuccess)
        return false;

    // Copy every element with pitch (width is 1 element, height is w * h)
    status = cudaMemcpy2D(tmp, sizeof(TO), data, sizeof(T), sizeof(T), w * h, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
        return false;

    return copyToArray<TO, TO>(tmp, array, w, h);
}

/**
 * Specialization for DEPTH_TYPE
 */
template <>
class TypedVisualizer<DEPTH_TYPE> : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h);

    virtual void bindTexture_(void* data, const size_t dataSize) override;

    virtual void render_() override;

private:    
    float fx_ = 1320.f;
    float fy_ = 1320.f;
    float cx_ = w_/2.f;
    float cy_ = h_/2.f;

    pangolin::OpenGlRenderState s_cam_;
    pangolin::View d_cam_;

    pangolin::GlBufferCudaPtr vertices_;
    pangolin::GlBufferCudaPtr normals_;
    pangolin::GlBufferCudaPtr indices_;    

    pangolin::GlSlProgram shader_;
};

/**
 * Specialization for COLOR_TYPE_GREY
 */
template <>
class TypedVisualizer<COLOR_TYPE_GREY> : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h)
     : VisualizerBase(w, h)
    {
        std::cout << "Visualizing GREY" << std::endl;
    }

    virtual bool initTexture_(const size_t dataSize) override
    {
        assert(dataSize == w_ * h_ * sizeof(unsigned char));
        texture_.Reinitialise(w_, h_, GL_LUMINANCE8, true, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE);
        return texture_.IsValid();
    }
};

/**
 * Specialization for COLOR_TYPE_RGB
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGB> : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h)
     : VisualizerBase(w, h)
    {
        std::cout << "Visualizing RGB" << std::endl;
    }

    virtual bool initTexture_(const size_t dataSize) override
    {
        assert(dataSize == 3 * w_ * h_ * sizeof(unsigned char));
        texture_.Reinitialise(w_, h_, GL_RGB8, true, 0, GL_RGB, GL_UNSIGNED_BYTE);
        return texture_.IsValid();
    }

    virtual bool copyToTexture_(const void* data, cudaArray_t array, const size_t dataSize) override
    {
        // Texture will be RGBA, so we need to inflate
        assert(dataSize == 3 * w_ * h_ * sizeof(unsigned char));
        return copyToArray<uchar3, uchar4>((uchar3*)data, array, w_, h_);
    }
};

/**
 * Specialization for COLOR_TYPE_RGBA
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGBA> : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h)
     : VisualizerBase(w, h)
    {
        std::cout << "Visualizing RGB" << std::endl;
    }
private:
    virtual bool initTexture_(const size_t dataSize) override
    {
        assert(dataSize == 4 * w_ * h_ * sizeof(unsigned char));
        texture_.Reinitialise(w_, h_, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE);
        return texture_.IsValid();
    }
};

/**
 * Specialization for COLOR_TYPE_GREY_F
 */
template <>
class TypedVisualizer<COLOR_TYPE_GREY_F> : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h)
     : VisualizerBase(w, h)
    {
        std::cout << "Visualizing GREY" << std::endl;
    }

    virtual bool initTexture_(const size_t dataSize) override
    {
        assert(dataSize == w_ * h_ * sizeof(float));
        texture_.Reinitialise(w_, h_, GL_LUMINANCE32F_ARB, true, 0, GL_LUMINANCE, GL_FLOAT);
        return texture_.IsValid();
    }
};

/**
 * Specialization for COLOR_TYPE_RGB_F
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGB_F> : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h)
     : VisualizerBase(w, h)
    {
        std::cout << "Visualizing RGB" << std::endl;
    }
private:
    virtual bool initTexture_(const size_t dataSize) override
    {
        texture_.Reinitialise(w_, h_, GL_RGB32F, true, 0, GL_RGB, GL_FLOAT);
        return texture_.IsValid();
    }

    virtual bool copyToTexture_(const void* data, cudaArray_t array, const size_t dataSize) override
    {
        // Texture will be RGBA, so we need to inflate
        assert(dataSize == 3 * w_ * h_ * sizeof(float));
        return copyToArray<float3, float4>((float3*)data, array, w_, h_);
    }   
};

/**
 * Specialization for COLOR_TYPE_RGBA_F
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGBA_F> : public VisualizerBase
{
public:
    TypedVisualizer(const size_t w, const size_t h)
     : VisualizerBase(w, h)
    {
        std::cout << "Visualizing RGBA" << std::endl;
    }
private:
    virtual bool initTexture_(const size_t dataSize) override
    {
        assert(dataSize == 4 * w_ * h_ * sizeof(float));
        texture_.Reinitialise(w_, h_, GL_RGBA32F, true, 0, GL_RGBA, GL_FLOAT);
        return texture_.IsValid();
    } 
};

/**
 * Base
 */

template<VisType T> TypedVisualizer<T>::TypedVisualizer(const size_t w, const size_t h)
 : VisualizerBase(w, h)
{
}

template<VisType T> VisType TypedVisualizer<T>::type() const
{
    return T;
}

template<VisType T> bool TypedVisualizer<T>::initTexture_(const size_t dataSize)
{
    throw std::runtime_error("Cannot init texture of base class, specialized method might be missing!");
}


/**
 * DEPTH_TYPE
 */

inline TypedVisualizer<DEPTH_TYPE>::TypedVisualizer(const size_t w, const size_t h)
 : VisualizerBase(w, h)
{
    std::cout << "Visualizing Depth" << std::endl;
}

inline void TypedVisualizer<DEPTH_TYPE>::bindTexture_(void* data, const size_t dataSize)
{
    assert(dataSize == w_ * h_ * sizeof(float));

    // Create 3D view
    s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(w_, h_, fx_, fy_, cx_, cy_, 0.1, 5000),
        pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 10, pangolin::AxisNegY)
        //pangolin::ModelViewLookAt(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, pangolin::AxisY)
    );

    d_cam_ = pangolin::Display("cam")
        .SetBounds(0,1.0f,0,1.0f,-(float)w_/h_)
        .SetHandler(new pangolin::Handler3D(s_cam_));

    // Enable custom shader
    shader_.ClearShaders();
    shader_.AddShader(pangolin::GlSlAnnotatedShader, pangolin::ambient_light_shader);
    //shader_.AddShader(pangolin::GlSlAnnotatedShader, pangolin::default_model_shader);
    shader_.Link();

    // Needs to be initialized in order for the run() method to work
    texture_.Reinitialise(1, 1, GL_RGB, false, 0, GL_RGB, GL_FLOAT);
    
    vertices_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,
       2 * w_ * h_, GL_FLOAT, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW);
    pangolin::CudaScopedMappedPtr vertex_array(vertices_);
    cu_backProjectAndCalcNormals((float*)data, (float4*)*vertex_array, w_, h_, fx_, fy_, cx_, cy_);

    // Generate the indices of connected triangles to the right[0 1 0+w], and to the left [1 1+w 1+w-1] to connect all points
    indices_.Reinitialise(pangolin::GlBufferType::GlElementArrayBuffer,
        6 * w_ * h_, GL_UNSIGNED_INT, 1, cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW);
    
    pangolin::CudaScopedMappedPtr index_array(indices_);
    cu_generateVertexIndices((float*)data, (unsigned int*)*index_array, w_, h_);
    
}

inline void TypedVisualizer<DEPTH_TYPE>::render_()
{
    // Render triangles for every 
    d_cam_.Activate(s_cam_);

    glClearColor(0.1f, 0.1f, 0.1f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);    

    if (shader_.Valid())
    {
        shader_.Bind();

        // Uniforms for vertex shader
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity(); // No model transformation
        Eigen::Matrix4f view = s_cam_.GetModelViewMatrix();
        Eigen::Matrix4f projection = s_cam_.GetProjectionMatrix();
        shader_.SetUniform("model", model);
        shader_.SetUniform("view", view);
        shader_.SetUniform("projection", projection);

        // Uniform for fragment shader
        shader_.SetUniform("objectColor", 1.0f, 1.0f, 1.0f);
        shader_.SetUniform("lightColor", 1.f, 1.f, 1.f);
        shader_.SetUniform("lightPos", 0.f, 0.f, 0.f);
        shader_.SetUniform("ambient", .3f, .3f, .3f );

        vertices_.Bind();
        size_t stride = vertices_.count_per_element * pangolin::GlDataTypeBytes(vertices_.datatype);
        
        // Vertices: 0, 1, 2, 3,   8, 9, 10, 11 ...
        glVertexAttribPointer(0, vertices_.count_per_element, vertices_.datatype, GL_FALSE, 2 * stride, (void*)0);
        glEnableVertexAttribArray(0);

        // Normals: 4, 5, 6, 7,   12, 13, 14, 15 ...
        glVertexAttribPointer(1, vertices_.count_per_element, vertices_.datatype, GL_FALSE, 2 * stride, (void*)stride);
        glEnableVertexAttribArray(1);

        indices_.Bind();
        glDrawElements(GL_TRIANGLES, indices_.num_elements, indices_.datatype, 0);
        indices_.Unbind();

        glDisableClientState(GL_VERTEX_ARRAY);
        normals_.Unbind();
        vertices_.Unbind();

        shader_.Unbind();
    }

    glDisable(GL_CULL_FACE);
    
}



} // image
