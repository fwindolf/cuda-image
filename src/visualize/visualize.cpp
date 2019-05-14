#include "cuimage/visualize/visualize.h"

using namespace cuimage;

VisualizerBase::VisualizerBase(const std::string& name, const size_t w, const size_t h)
 : name_(name), 
   w_(w), 
   h_(h),
   contextInitialized_(false),
   renderRequestPending_(false),
   uploadPending_(false),
   running_(false),
   quit_(true)
{
}

VisualizerBase::~VisualizerBase()
{
    deinitialize();
}

void VisualizerBase::initialize()
{
    contextInitialized_ = true;

    // Create window with new context
    window_ = &pangolin::CreateWindowAndBind(name_, w_, h_);
    window_->MakeCurrent();

    glEnable(GL_DEPTH_TEST);
    window_->RemoveCurrent();

    // Start thread
    running_ = true;
    runThread_ = std::thread(&VisualizerBase::run, this);
}

void VisualizerBase::deinitialize()
{
    // If run thread is waiting for render request, send one but let it exit bc not running
    running_ = false;
    renderRequested_.notify_one();

    if(runThread_.joinable())
        runThread_.join();
    
    pangolin::DestroyWindow(name_);
}


void VisualizerBase::create() 
{
    if (!contextInitialized_)
        initialize();
}

void VisualizerBase::show(void* data, const size_t dataSize, bool wait)
{
    if (!contextInitialized_)
    {
        std::cerr << "Trying to show image on missing window!" << std::endl;
        return;
    }

    // Quit current render request, and update user interaction
    quit_ = true;
    waitUserAction_ = wait;
    // Take context from run thread and upload data
    upload(data, dataSize);

    // Request rendering
    renderMutex_.lock();
    renderRequestPending_ = true;
    renderRequested_.notify_one();
    renderMutex_.unlock();

    if (!wait)
        return;

    // Wait for rendering to be complete
    std::unique_lock<std::mutex> lock(completeMutex_);
    while(renderRequestPending_)
        renderComplete_.wait(lock);
}

void VisualizerBase::close(bool force)
{
    if (force)
    {
        // Force quitting by exiting render loop (normally done by user)
        running_ = false;
    }
    else
    {
        // Wait for rendering to be complete
        quit_ = false;
        waitUserAction_ = true;
        std::unique_lock<std::mutex> lock(completeMutex_);
        while(renderRequestPending_)
            renderComplete_.wait(lock);
    }
    
    deinitialize();
}

void VisualizerBase::upload(void* data, const size_t dataSize)
{
    // Interrupt run thread to upload data
    uploadMutex_.lock();

    uploadPending_ = true;
    uploadData_ = data;
    uploadDataSize_ = dataSize;    

    uploadMutex_.unlock();
}


VisType VisualizerBase::type() const
{
    throw std::runtime_error("Unknown type in base class!");
}

std::string VisualizerBase::windowName() const
{
    return name_;
}


int VisualizerBase::run()
{
    // Create and setup new window
    window_->MakeCurrent();

    pangolin::View& d_image = pangolin::Display("image")
        .SetBounds(0.f, 1.0f, 0.f, 1.f, (float) w_ / (float) h_);    

    auto status = initContext_();
    if(!status)
    {
        std::cerr << "Could not initialize context!" << std::endl;
        return -1;
    }

    // Stop rendering on escape press
    auto& quit_ref = quit_;
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_ESCAPE, [&quit_ref](){
        quit_ref = true;
    });

    while(running_)
    {
        quit_ = false;

        // Wait for the show method to produce a render request        
        {
            //  Create lock for just this scope
        std::unique_lock<std::mutex> lock(renderMutex_);
        while(running_ && !renderRequestPending_)
            renderRequested_.wait(lock); // Inside while to counter spurious wakes
        }

        // Run as long as user interaction is required and quit(ESC) is pressed
        //  or forever, when user action is not required
        while(!waitUserAction_ || !quit_)
        {
        // Exit gracefully if requested
        if (!running_)
            return 0;
        

        while(!quit_)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            d_image.Activate();

            glColor3f(1.0, 1.0, 1.0);

            // Defer rendering until new data was uploaded
            if (uploadPending_)
            {
                // (Re-)Init texture
                uploadMutex_.lock();
                if (!initTexture_())
                {
                    std::cerr << "Could not initialize GlTexture!" << std::endl;
                    break;
                }

                if (!bindToTexture_())
                {
                    std::cerr << "Could not bind to Texture!" << std::endl;
                    break;
                }

                // Upload data
                if (!copyToTexture_())
                {
                    std::cerr << "Could not copy to Texture!" << std::endl;
                    break;
                } 

                uploadPending_ = false;
                uploadMutex_.unlock();
            }

            render_();

            pangolin::FinishFrame();
        }

        // User closed window, so this request was fulfilled
        completeMutex_.lock();
        renderRequestPending_ = false;
        renderComplete_.notify_one();
        completeMutex_.unlock();        
    }
}

bool VisualizerBase::initContext_()
{
    return true;
}

bool VisualizerBase::bindToTexture_()
{
    auto status = cudaGraphicsMapResources(1, &texture_.cuda_res, 0);
    if (status != cudaSuccess)
    {
        std::cerr << "Could not map resource of texture" << std::endl;
        return false;
    }
    
    status = cudaGraphicsSubResourceGetMappedArray(&array_, texture_.cuda_res, 0, 0);
    if (status != cudaSuccess)
    {
        std::cerr << "Could not map cuda array to resource" << std::endl;
        return false;
    }

    return true;                
}

bool VisualizerBase::initTexture_()
{
    throw std::runtime_error("Cannot auto-determine type of texture!");
}

bool VisualizerBase::copyToTexture_()
{
    auto status = cudaMemcpyToArray(array_, 0, 0, uploadData_, uploadDataSize_, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "Could not copy data to cuda array" << std::endl;
        return false;
    }
    
    status = cudaGraphicsUnmapResources(1, &texture_.cuda_res, 0);
    if (status != cudaSuccess)
    {
        std::cerr << "Could not unmap texture resource" << std::endl;
        return false;
    }

    return true;
}   

void VisualizerBase::render_()
{
    texture_.RenderToViewportFlipY();
}
