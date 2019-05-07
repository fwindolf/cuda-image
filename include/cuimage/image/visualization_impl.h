

template <typename T>
void Image<T>::setVisualizationStrategy(const VisType type)
{
    assert(type != VisType::NONE);

    // Skip if strategy doesnt change
    if (vis_ && visType_ == type)
        return;

    // Create a new visualizer
    // TODO: This should be solvable by using constexpr... the type is set by user at compile time
    switch(type)
    {
    case DEPTH_TYPE:
        vis_.reset(new TypedVisualizer<DEPTH_TYPE>(w_, h_));
        break;
    case COLOR_TYPE_GREY:
        vis_.reset(new TypedVisualizer<COLOR_TYPE_GREY>(w_, h_));
        break;
    case COLOR_TYPE_GREY_F:
        vis_.reset(new TypedVisualizer<COLOR_TYPE_GREY_F>(w_, h_));
        break;
    case COLOR_TYPE_RGB:
        vis_.reset(new TypedVisualizer<COLOR_TYPE_RGB>(w_, h_));
        break;
    case COLOR_TYPE_RGB_F:
        vis_.reset(new TypedVisualizer<COLOR_TYPE_RGB_F>(w_, h_));
        break;
    case COLOR_TYPE_RGBA:
        vis_.reset(new TypedVisualizer<COLOR_TYPE_RGBA>(w_, h_));
        break;
    case COLOR_TYPE_RGBA_F:
        vis_.reset(new TypedVisualizer<COLOR_TYPE_RGBA_F>(w_, h_));
        break;
    default:
        throw std::runtime_error("Invalid type for visualization: " + std::to_string(type));
    }
    visType_ = type;
}
template <typename T>
void Image<T>::show(const std::string windowName)
{
    return vis_->show(windowName, data_, sizeBytes(), true);
}

template <typename T>
void Image<T>::show(const std::string windowName, const VisType type)
{
    setVisualizationStrategy(type);
    show(windowName);
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
    delete h_data;
}
