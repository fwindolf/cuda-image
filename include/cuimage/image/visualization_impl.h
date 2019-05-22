template <typename T>
template <VisType V>
void Image<T>::createWindow(const std::string windowName)
{
    if (vis_)
        throw std::runtime_error("Cannot create a window for an image when another window is open");

    static_assert(V != VisType::NONE, "Visualization type must be valid!");

    vis_.reset(new TypedVisualizer<V>(windowName, w_, h_));
    vis_->create();
}

template <typename T>
void Image<T>::closeWindow(bool force)
{
    if(!vis_)
        return;

    vis_->close(force);
    vis_.release();
}

template <typename T>
void Image<T>::show(bool wait) const
{
    if (!vis_)
        throw std::runtime_error("To show an image without explicitly creating a window, use the templated show method!");

    vis_->show(data_, sizeBytes(), wait);
}

template <typename T>
template <VisType V>
void Image<T>::show(const std::string windowName, bool close)
{    
    // Check or create window
    if (vis_)
    {
        if (windowName != vis_->windowName() || 
            V != vis_->type())
            throw std::runtime_error("Cannot show in new window when there is already a different window. Use closeWindow() first!");
    }      
    else
    {
        createWindow<V>(windowName);
    }
    
    // Visualize blocking, then close window again
    show(true);
    if (close)
        closeWindow(true);
}

// Default (const) visualization
template <typename T> 
cuimage::VisualizerBase* getDefaultVisualizer(const std::string& name, const size_t w, const size_t h);

template <>
inline cuimage::VisualizerBase* getDefaultVisualizer<uchar>(const std::string& name, const size_t w, const size_t h)
{
    return new TypedVisualizer<COLOR_TYPE_GREY>(name, w, h);
}

template <>
inline cuimage::VisualizerBase* getDefaultVisualizer<uchar3>(const std::string& name, const size_t w, const size_t h)
{
    return new TypedVisualizer<COLOR_TYPE_RGB>(name, w, h);
}

template <>
inline cuimage::VisualizerBase* getDefaultVisualizer<float>(const std::string& name, const size_t w, const size_t h)
{
    return new TypedVisualizer<COLOR_TYPE_GREY_F>(name, w, h);
}

template <>
inline cuimage::VisualizerBase* getDefaultVisualizer<float3>(const std::string& name, const size_t w, const size_t h)
{
    return new TypedVisualizer<COLOR_TYPE_RGB_F>(name, w, h);
}


template <typename T>
void Image<T>::visualize() const
{
    static int id = 0;
    std::string name = "Anonymous Window" + std::to_string(id);
    id++;
    std::unique_ptr<VisualizerBase> vis(getDefaultVisualizer<T>(name, w_, h_));
    
    vis->create();
    vis->show(data_, sizeBytes());
    vis->close(true);
    
    std::cerr << "Debug visualization with visualize() might kill your cuda context!" << std::endl;
}

template <typename T>
void Image<T>::print() const
{
    assert(data_ && size());

    T* h_data = new T[size()];
    cudaSafeCall(cudaMemcpy(h_data, data_, sizeBytes(), cudaMemcpyDeviceToHost));

    for(int py = 0; py < h_; py++)
    {
        for (int px = 0; px < w_; px++)
        {
            std::cout << h_data[px + py * w_] << " ";
        }   
        std::cout << std::endl;
    }
    delete[] h_data;
}
