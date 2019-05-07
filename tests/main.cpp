#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <cuda_runtime.h>


template <typename T, typename TO>
bool copyToArray(T* data, cudaArray_t array, int w, int h, float fillWith = 1.f)
{
    cudaError_t status;
    if (std::is_same<T, TO>())
    {
        status = cudaMemcpyToArray(array, 0, 0, data, w * h * sizeof(TO), cudaMemcpyDeviceToDevice);
        return (status == cudaSuccess);
    }

    float ratio = (float)sizeof(TO) / sizeof(T);
    if (ratio < 2)
    {
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
    else if (ratio >= 2)
    {
        // TO has more than double the channels -> broadcast channel values
        TO* tmp;
        status = cudaMalloc(&tmp, w * h * sizeof(TO));
        if (status != cudaSuccess)
            return false;

        // For every time the data needs to be repeated (4x for 1->4, 3x for 1->3, 2x for 2->4)
        //   copy it to the temporary array from an offsetted position
        for (int i = 0; i <= ratio; i++)
        {
            int off = i * (ratio == 2 ? 2 : 1); // only for 2x we need copy first part to back, else just repeat
            status = cudaMemcpy2D((void*)&tmp[i], sizeof(TO), data, sizeof(T), sizeof(T), w * h, cudaMemcpyDeviceToDevice);
            if (status != cudaSuccess)
                return false;
        }
        return copyToArray<TO, TO>(tmp, array, w, h);
    }
}

void test_show_uchar3_from_array_bound_texture()
{
    std::string color_file = std::string(SOURCE_DIR) + "/data/image.png";
    auto m = cv::imread(color_file);

    auto width = m.cols;
    auto height = m.rows;

    cv::Mat m_f;
    cv::cvtColor(m, m_f, cv::COLOR_BGR2RGB);
    //m_f.convertTo(m_f, CV_32F);

    assert(m_f.isContinuous());

    // Create window before creating textures
    pangolin::CreateWindowAndBind("Test", m.cols, m.rows);

    glEnable(GL_DEPTH_TEST);

    uchar3* data;
    auto status = cudaMalloc(&data, width * height * sizeof(uchar3));
    if(status != cudaSuccess)
    {
        std::cerr << "Unable alloc device data!" << std::endl;
        return;
    }   
    status = cudaMemcpy(data, m_f.data, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if(status != cudaSuccess)
    {
        std::cerr << "Unable to copy to device data!" << std::endl;
        return;
    }        

    pangolin::GlTextureCudaArray tex_f;
    cudaArray_t data_;

    tex_f.Reinitialise(width, height, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE);

    pangolin::CudaScopedMappedArray tex_arr(tex_f);
    data_ = *tex_arr;

    if (!copyToArray<uchar3, uchar4>(data, data_, width, height, 1.f))
    {
        std::cerr << "Unable to copy data to array!" << std::endl;
        return;
    }    

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY)
    );

    // whilst a negative ratio makes the view 'grow to fit' (cropping the view).
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0, 1.0f, 0, 1.0f, (float) m.cols / (float) m.rows)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_image = pangolin::Display("image")
        .SetBounds(0, 1.0f, 0, 1.0f, (float) m.cols / (float) m.rows);

    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        d_image.Activate();
        glColor3f(1.0,1.0,1.0);
        
        tex_f.RenderToViewportFlipY();

        pangolin::FinishFrame();
    }

    pangolin::GetBoundWindow()->RemoveCurrent();
    
}

void test_show_float3_from_array_bound_texture()
{
    std::string color_file = std::string(SOURCE_DIR) + "/data/image.png";
    auto m = cv::imread(color_file);

    auto width = m.cols;
    auto height = m.rows;

    cv::Mat m_f;
    cv::cvtColor(m, m_f, cv::COLOR_BGR2RGB);
    m_f.convertTo(m_f, CV_32F, 1.f/255);

    assert(m_f.isContinuous());

    // Create window before creating textures
    pangolin::CreateWindowAndBind("Test", m.cols, m.rows);

    glEnable(GL_DEPTH_TEST);

    float3* data;
    auto status = cudaMalloc(&data, width * height * sizeof(float3));
    if(status != cudaSuccess)
    {
        std::cerr << "Unable alloc device data!" << std::endl;
        return;
    }   
    status = cudaMemcpy(data, m_f.data, 3 * width * height * sizeof(float), cudaMemcpyHostToDevice);
    if(status != cudaSuccess)
    {
        std::cerr << "Unable to copy to device data!" << std::endl;
        return;
    }        

    pangolin::GlTextureCudaArray tex_f;
    cudaArray_t data_;

    tex_f.Reinitialise(width, height, GL_RGBA32F, true, 0, GL_RGBA, GL_FLOAT);
    assert(tex_f.IsValid());

    pangolin::CudaScopedMappedArray tex_arr(tex_f);
    data_ = *tex_arr;
       
    if (!copyToArray<float3, float4>(data, data_, width, height, 1.f))
    {
        std::cerr << "Unable to copy data to array!" << std::endl;
        return;
    }

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY)
    );

    // whilst a negative ratio makes the view 'grow to fit' (cropping the view).
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0, 1.0f, 0, 1.0f, (float) m.cols / (float) m.rows)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_image = pangolin::Display("image")
        .SetBounds(0, 1.0f, 0, 1.0f, (float) m.cols / (float) m.rows);

    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        d_image.Activate();
        glColor3f(1.0,1.0,1.0);
        
        tex_f.RenderToViewportFlipY();

        pangolin::FinishFrame();
    }

    pangolin::GetBoundWindow()->RemoveCurrent();
    
}


void test_show_all_formats_in_texture()
{
    std::string color_file = std::string(SOURCE_DIR) + "/data/image.png";
    auto m = cv::imread(color_file);

    m.convertTo(m, CV_8U);

    auto width = m.cols;
    auto height = m.rows;

    cv::Mat m_rgb, m_rgb_f, m_rgba, m_g, m_f;
    cv::cvtColor(m, m_rgb, cv::COLOR_BGR2RGB);
    m_rgb.convertTo(m_rgb_f, CV_32F, 1.f/255);
    cv::cvtColor(m, m_rgba, cv::COLOR_BGR2RGBA);
    cv::cvtColor(m, m_g, cv::COLOR_BGR2GRAY);
    m_g.convertTo(m_f, CV_32F, 1.f/255);

    // Create window before creating textures
    pangolin::CreateWindowAndBind("Test", m.cols, m.rows);

    glEnable(GL_DEPTH_TEST);

    pangolin::GlTexture tex_rgb(width, height, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE);    
    tex_rgb.Upload((unsigned char*)m_rgb.data, GL_RGB, GL_UNSIGNED_BYTE);   

    pangolin::GlTexture tex_rgb_f(width, height, GL_RGB, true, 0, GL_RGB, GL_FLOAT);    
    tex_rgb_f.Upload((float*)m_rgb_f.data, GL_RGB, GL_FLOAT);   

    pangolin::GlTexture tex_rgba(width, height, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE);
    tex_rgba.Upload(m_rgba.data, GL_RGBA, GL_UNSIGNED_BYTE);

    pangolin::GlTexture tex_g(width, height, GL_LUMINANCE, true, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE);
    tex_g.Upload(m_g.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);

    pangolin::GlTexture tex_f(width, height, GL_LUMINANCE, true, 0, GL_LUMINANCE, GL_FLOAT);
    tex_f.Upload(m_f.data, GL_LUMINANCE, GL_FLOAT);
    
    assert(tex_rgb.IsValid());
    assert(tex_rgba.IsValid());
    assert(tex_rgb_f.IsValid());
    assert(tex_g.IsValid());
    assert(tex_f.IsValid());

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY)
    );

    // Aspect ratio allows us to constrain width and height whilst fitting within specified
    // bounds. A positive aspect ratio makes a view 'shrink to fit' (introducing empty bars),
    // whilst a negative ratio makes the view 'grow to fit' (cropping the view).
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0, 1.0f, 0, 1.0f, (float) m.cols / (float) m.rows)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_image_rgb = pangolin::Display("image_rgb")
        .SetBounds(0, 1.0f, 0, 1.0f, (float) m.cols / (float) m.rows);

    pangolin::View& d_image_rgba = pangolin::Display("image_rgba")
        .SetBounds(0.f, .5f, 0.f, .5f, (float) m.cols / (float) m.rows);

    pangolin::View& d_image_rgb_f = pangolin::Display("image_rgb_f")
        .SetBounds(.5f, 1.f, .5f, 1.f, (float) m.cols / (float) m.rows);

    pangolin::View& d_image_g = pangolin::Display("image_g")
        .SetBounds(0.f, .5f, .5f, 1.f, (float) m.cols / (float) m.rows);
    
    pangolin::View& d_image_f = pangolin::Display("image_f")
        .SetBounds(0.f, .5f, .5f, 1.f, (float) m.cols / (float) m.rows);

    // Show until user destroys window
    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        /*
        d_image_rgb.Activate();
        glColor3f(1.0,1.0,1.0);
        tex_rgb.RenderToViewportFlipY();

        d_image_rgba.Activate();
        glColor3f(1.0,1.0,1.0);
        tex_rgba.RenderToViewportFlipY();

        d_image_rgb_f.Activate();
        glColor3f(1.0,1.0,1.0);
        tex_rgb_f.RenderToViewportFlipY();
        */
        d_image_f.Activate();
        glColor3f(1.0,1.0,1.0);
        tex_f.RenderToViewportFlipY();


        pangolin::FinishFrame();
    }

    pangolin::GetBoundWindow()->RemoveCurrent();
}


#include "cuimage/cuda/type.h"

struct FileReader
{
    template <typename T> 
    T* read() const;
}

struct PngReader : public FileReader
{
    template <typename T, typename std::enable_if<(has_4_channels<T>::value || has_3_channels<T>::value || has_1_channels<T>::value) && is_uchar_type<T>::value, T>::type* = nullptr>
    T* read() const;
}

struct ExrReader : public FileReader
{
    template <typename T, typename std::enable_if<is_float_type<T>::value, T>::type* = nullptr>
    T* read() const;
}

int main(int argc, char** argv)
{
    // test_show_all_formats_in_texture();

    //test_show_uchar3_from_array_bound_texture();

    //test_show_float3_from_array_bound_texture();

    std::string fileName = "asdf.png";
    std::string fileType = fileName.substr(fileName.find_last_of(".") + 1);
    std::transform(fileType.begin(), fileType.end(), fileType.begin(), ::toupper);

    if (fileName == "PNG")
    {
        PngReader f;
        uchar4* data = f.read();
    }
    else if (fileName == "EXR")
    {
        ExrReader f;
        float* data = f.read();
    }

}