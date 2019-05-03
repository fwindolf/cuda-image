#include "image/visualize/visualize.h"

using namespace image;

VisualizerBase::VisualizerBase(const size_t w, const size_t h)
 : w_(w), 
   h_(h)
{
}

void VisualizerBase::show(const std::string name, void* data, const size_t dataSize, const bool wait)
{
    runThread_ = std::thread(&VisualizerBase::run, this, name, data, dataSize);

    if (wait)
        runThread_.join();
    else
        runThread_.detach();
}


VisType VisualizerBase::type() const
{
    throw std::runtime_error("Unknown type in base class!");
}


int VisualizerBase::run(const std::string name, void* data, const size_t dataSize)
{
    // Create and setup new window
    pangolin::CreateWindowAndBind(name, w_, h_);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& d_image = pangolin::Display("image")
        .SetBounds(0.f, 1.0f, 0.f, 1.f, (float) w_ / (float) h_);

    // Bind the image to the GLTexture
    bindTexture_(data, dataSize);
    if (!texture_.IsValid())
        throw std::runtime_error("Invalid texture! Bind to texture first!");

    // Show until user destroys window
    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        d_image.Activate();

        glColor3f(1.0,1.0,1.0);

        render_();

        pangolin::FinishFrame();
    }

    pangolin::GetBoundWindow()->RemoveCurrent();
    
    pangolin::DestroyWindow(name);
}

void VisualizerBase::bindTexture_(void* data, const size_t dataSize)
{
    // Have to reinitialize texture, as the context was re-initialized in run()
    if (!initTexture_(dataSize))
    {
        std::cerr << "Could not initialize GlTexture!" << std::endl;
        return;
    }

    assert(texture_.width == w_);
    assert(texture_.height == h_);

    // Map and bind, unmaps when running out of scope
    pangolin::CudaScopedMappedArray array(texture_);
    
    // Access in cuda context, write data to texture
    if (!copyToTexture_(data, *array, dataSize))
    {
        std::cerr << "Could not copy to GlTexture!" << std::endl;
        return;
    }    
}

bool VisualizerBase::initTexture_(const size_t dataSize)
{
    throw std::runtime_error("Cannot auto-determine type of texture!");
}

bool VisualizerBase::copyToTexture_(const void* data, cudaArray_t array, const size_t dataSize)
{
    auto status = cudaMemcpyToArray(array, 0, 0, data, dataSize, cudaMemcpyDeviceToDevice);
    return (status == cudaSuccess);
}   

void VisualizerBase::render_()
{
    texture_.RenderToViewportFlipY();
}