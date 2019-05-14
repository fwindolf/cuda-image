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
void Image<T>::show(const std::string windowName)
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
    closeWindow(true);
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
