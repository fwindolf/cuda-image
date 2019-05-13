#include "cuimage/visualize/visualize.h"

using namespace cuimage;

VisualizerBase::VisualizerBase(const std::string name, const size_t w, const size_t h)
 : name_(name), 
   w_(w), 
   h_(h),
   running_(false)
{
    pangolin::CreateWindowAndBind(name, w, h);
    glEnable(GL_DEPTH_TEST);
    pangolin::GetBoundWindow()->RemoveCurrent();
}

VisualizerBase::~VisualizerBase()
{
    close();  
    if (runThread_.joinable())
        runThread_.join();
}

void VisualizerBase::show(void* data, const size_t dataSize, bool wait)
{
    running_ = true;
    runThread_ = std::thread(&VisualizerBase::run, this, data, dataSize, wait);

    // runThread runs until user input or once if wait is not activated
    runThread_.join(); 
    if (wait)
        close(); // Window must be closed after showing in wait mode
}

void VisualizerBase::close()
{
    pangolin::BindToContext(name_);
    pangolin::DestroyWindow(name_);
}

VisType VisualizerBase::type() const
{
    throw std::runtime_error("Unknown type in base class!");
}

std::string VisualizerBase::windowName() const
{
    return name_;
}


int VisualizerBase::run(void* data, const size_t dataSize, const bool wait)
{
    // Create and setup new window
    pangolin::BindToContext(name_);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& d_image = pangolin::Display("image")
        .SetBounds(0.f, 1.0f, 0.f, 1.f, (float) w_ / (float) h_);

    // Bind the image to the GLTexture
    bindTexture_(data, dataSize);
    if (!texture_.IsValid())
        throw std::runtime_error("Invalid texture! Bind to texture first!");

    // Show until user closes, timer elapses (only if waitMs > 0)
    do 
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_image.Activate();

        glColor3f(1.0, 1.0, 1.0);

        render_();

        pangolin::FinishFrame();
    }
    while(wait && !pangolin::ShouldQuit());

    pangolin::GetBoundWindow()->RemoveCurrent();
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