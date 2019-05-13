template <typename T>
template <VisType V>
void Image<T>::createWindow(const std::string windowName)
{
    if (vis_)
        throw std::runtime_error("Cannot create a window for an image when another window is open");

    static_assert(V != VisType::NONE, "Visualization type must be valid!");

    vis_.reset(new TypedVisualizer<V>(windowName, w_, h_));
}

template <typename T>
void Image<T>::closeWindow()
{
    if(!vis_)
        return;

    vis_->close();
    vis_.release();
}

template <typename T>
void Image<T>::show(int waitMs) const
{
    if (!vis_)
        throw std::runtime_error("To show an image without explicitly creating a window, use the templated show method!");

    vis_->show(data_, sizeBytes(), waitMs);
}

template <typename T>
template <VisType V>
void Image<T>::show(const std::string windowName, int waitMs)
{
    if (vis_)
    {
        if (windowName != vis_->windowName())
        {
            assert(vis_->type() == V);
            vis_->show(data_, sizeBytes(), waitMs);
        }
        else
        {
            throw std::runtime_error("Cannot show in new window when there is already a created window. Use closeWindow() first!");
        }
    }      

    createWindow<V>(windowName);
    show(waitMs);
    closeWindow();
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
