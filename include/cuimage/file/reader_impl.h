

template<typename TO>
DevPtr<TO> FileReader::read(const std::string fileName, size_t& width, size_t& height, size_t& c) const
{
    std::string fileType = fileName.substr(fileName.find_last_of(".") + 1);
    std::transform(fileType.begin(), fileType.end(), fileType.begin(), ::toupper);

    if (fileType == "PNG")
    {        
        std::vector<uchar> image = readPng(fileName, width, height, c);
        
        if (c == 3)
        {
            auto devptr = upload<uchar, uchar3>(image.data(), width, height, c);
            return convert<uchar3, TO>(devptr);
        }
        else if (c == 4)
        {
            auto devptr = upload<uchar, uchar4>(image.data(), width, height, c);
            return convert<uchar4, TO>(devptr);
        }
        else
        {
            throw std::runtime_error("Invalid numbers of channels in PNG file");        
        }            
    }
    else if (fileType == "EXR")
    {
        std::vector<float> image = readExr(fileName, width, height, c);
        auto devptr = upload<float, float>(image.data(), width, height, c);
        return convert<float, TO>(devptr);
    }
    else
    {
        throw std::runtime_error("Cannot read files of type " + fileType);        
    }
}

template <typename T, typename TO > 
DevPtr<TO> FileReader::upload(const T* h_data, const size_t width, const size_t height, const size_t chans) const
{
    // If the format matches, upload to gpu
    assert(sizeof(TO) == sizeof(T) * chans);

    TO* d_data;
    cudaSafeCall(cudaMalloc(&d_data, width * height * sizeof(TO)));
    cudaSafeCall(cudaMemcpy(d_data, h_data, width * height * sizeof(TO), cudaMemcpyHostToDevice));

    return DevPtr<TO>(d_data, width, height);
}

/**
 * Same type
 */
template <typename T, typename TO, typename
    std::enable_if<
        (
            (is_float_type<T>::value && is_float_type<TO>::value) ||
            (is_uchar_type<T>::value && is_uchar_type<TO>::value) ||
            (is_int_type<T>::value   && is_int_type<TO>::value)
        ) && (
            (has_4_channels<T>::value && has_4_channels<TO>::value) ||
            (has_3_channels<T>::value && has_3_channels<TO>::value) ||
            (has_2_channels<T>::value && has_2_channels<TO>::value) ||
            (has_1_channels<T>::value && has_1_channels<TO>::value) ||
            (has_0_channels<T>::value && has_0_channels<TO>::value)
        ), T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input)
{   
    DevPtr<TO> output(input);
    return output;
}

/**
 * Same channels, convert only type
 */
template <typename T, typename TO, typename
    std::enable_if<
        (
            (is_float_type<T>::value && !is_float_type<TO>::value) ||
            (is_uchar_type<T>::value && !is_uchar_type<TO>::value) ||
            (is_int_type<T>::value   && !is_int_type<TO>::value)
        ) && (
            (has_4_channels<T>::value && has_4_channels<TO>::value) ||
            (has_3_channels<T>::value && has_3_channels<TO>::value) ||
            (has_2_channels<T>::value && has_2_channels<TO>::value) ||
            (has_1_channels<T>::value && has_1_channels<TO>::value) ||
            (has_0_channels<T>::value && has_0_channels<TO>::value)
        ), T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input_same_channels)
{  
    DevPtr<TO> output(input_same_channels.width, input_same_channels.height); // Allocates
    cu_Convert<T, TO>(output, input_same_channels);
    return output;
}

/**
 * Same type, convert only channels
 */
template <typename T, typename TO, typename
    std::enable_if<
        (
            (is_float_type<T>::value && is_float_type<TO>::value) ||
            (is_uchar_type<T>::value && is_uchar_type<TO>::value) ||
            (is_int_type<T>::value   && is_int_type<TO>::value)
        ) && (
            (has_4_channels<T>::value && !has_4_channels<TO>::value) ||
            (has_3_channels<T>::value && !has_3_channels<TO>::value) ||
            (has_2_channels<T>::value && !has_2_channels<TO>::value) ||
            (has_1_channels<T>::value && !has_1_channels<TO>::value) ||
            (has_0_channels<T>::value && !has_0_channels<TO>::value)
        ), T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input_same_type)
{
    DevPtr<TO> output(input_same_type.width, input_same_type.height); // Allocates
    if (channels<TO>() >= 3 && channels<T>() == 1)
    {
        cu_GrayToColor(output, input_same_type);
        return output;
    }
    else if (channels<TO>() == 1 && channels<T>() >= 3)
    {           
        cu_ColorToGray(output, input_same_type);
        return output;
    }
    else if (channels<T>() >= 3 && channels<T>() >= 3)
    {
        cu_ColorToColor(output, input_same_type);
        return output;
    }
    
    throw std::runtime_error("Could not convert from input (" + std::to_string(channels<T>()) + ") to output (" + std::to_string(channels<TO>()) + ") channels!");
}

/**
 * Different type and channels do not match
 */
template <typename T, typename TO, typename
    std::enable_if<
       (
            (is_float_type<T>::value && !is_float_type<TO>::value) ||
            (is_uchar_type<T>::value && !is_uchar_type<TO>::value) ||
            (is_int_type<T>::value   && !is_int_type<TO>::value)
        ) && (
            (has_4_channels<T>::value && !has_4_channels<TO>::value) ||
            (has_3_channels<T>::value && !has_3_channels<TO>::value) ||
            (has_2_channels<T>::value && !has_2_channels<TO>::value) ||
            (has_1_channels<T>::value && !has_1_channels<TO>::value) ||
            (has_0_channels<T>::value && !has_0_channels<TO>::value)
        ), T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input_different_type)
{
    throw std::runtime_error("No possible conversion between input and output type!");
}

template <typename T, typename TO>
DevPtr<TO> FileReader::convert(const DevPtr<T>& input) const
{
    return convertDevPtr<T, TO>(input);
}