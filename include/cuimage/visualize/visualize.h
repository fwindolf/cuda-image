/**
 * @file   visualize.h
 * @brief  Class to visualize different types of images
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda.h"
#include "cuimage/operations/math_cu.h"
#include "kernels.h"
#include "shaders/ambient.h"
#include "shaders/default.h"
#include "types.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include <cuda_gl_interop.h>


namespace cuimage
{

/**
 * @class Visualizer
 * @brief Show images
 */

class Visualizer
{
public:
    virtual ~Visualizer(){};

    /**
     * Create the window that binds to this context
     */
    virtual void create() = 0;

    /**
     * Show the data in the associated window
     * If wait is set, execution will halt until user input
     */
    virtual void show(void* data, const size_t dataSize, bool wait = true) = 0;

    /**
     * Close the window associated with this Visualizer
     */
    virtual void close(bool force = false) = 0;

    /**
     * Get the type of this Visualizer
     */
    virtual VisType type() const = 0;

    /**
     * Get the name of the associated window
     */
    virtual std::string windowName() const = 0;
};

/**
 * @class VisualizerBase
 * @brief Stores common data members
 * Needed for GlTexture which cannot be accessed in specialization otherways
 */
class VisualizerBase : public Visualizer
{
public:
    VisualizerBase(const std::string& name, const size_t w, const size_t h);

    ~VisualizerBase();

    virtual void create() override;

    virtual void show(
        void* data, const size_t dataSize, bool wait = true) override;

    virtual void close(bool force = false) override;

    virtual VisType type() const override;

    virtual std::string windowName() const override;

private:
    int run();

    void initialize();

    void upload(void* data, const size_t dataSize);

    void deinitialize();

protected:
    // All child class hooks are called within opengl context

    virtual bool initContext_();

    virtual bool bindToTexture_();

    virtual bool initTexture_();

    virtual bool copyToTexture_();

    virtual void render_();

    const size_t w_, h_;
    const std::string name_;

    bool contextInitialized_;

    std::atomic<bool> running_;

    std::atomic<bool> quit_;
    std::atomic<bool> waitUserAction_;

    std::atomic<bool> renderRequestPending_;
    std::condition_variable renderRequested_;
    std::mutex renderMutex_;

    std::condition_variable renderComplete_;
    std::mutex completeMutex_;

    std::atomic<bool> uploadPending_;
    std::mutex uploadMutex_;

    void* uploadData_;
    size_t uploadDataSize_;

    std::thread runThread_;

    pangolin::WindowInterface* window_;
    pangolin::GlTextureCudaArray texture_;
    cudaArray_t array_;
};

/**
 * @class TypedVisualizer
 * @brief Show images in a certain way
 */
template <VisType T> class TypedVisualizer : public VisualizerBase
{
public:
    TypedVisualizer(const std::string& name, const size_t w, const size_t h);

    ~TypedVisualizer(){};

    inline virtual VisType type() const override { return T; }

protected:
    virtual bool initContext_() override;

    virtual bool bindToTexture_() override;

    virtual bool initTexture_() override;

    virtual bool copyToTexture_() override;

    virtual void render_() override;
};

#include "visualizers/copy_impl.h"

#include "visualizers/color_type_impl.h"
#include "visualizers/depth_type_impl.h"
#include "visualizers/depth_with_gradients_impl.h"
#include "visualizers/normals_impl.h"

} // image
