

template <typename TO>
DevPtr<TO> convertPng(const DevPtr<uchar3>& input);

template <>
DevPtr<float3> convertPng(const DevPtr<uchar3>& input)
{
    DevPtr<float3> output(input.w, input.h); // Allocates
    cu_Convert<uchar3, float3>(output, input);
    return output;
}

template <>
DevPtr<float> convertPng(const DevPtr<uchar3>& input)
{
    DevPtr<float> output(input.w, input.h); // Allocates
    DevPtr<float3> tmp(intput.w, input.h); // Allocates
    cu_Convert<uchar3, float3>(tmp, input);
    cu_ColorToGray<float3, float>(output, tmp);
    tmp.free();
    return output;
}

template <typename T, typename TO>
DevPtr<TO> convertPng(const DevPtr<T>& input);

template <typename T, typename TO, typename 
    std::enable_if<std::is_same<T, TO>::value, T>::type* = nullptr>
DevPtr<TO> convertPng(const DevPtr<T>& input)
{
    return input;
}

template <typename T, typename TO, typename 
    std::enable_if<!is_same_type<T, TO>::value && has_same_channels<T, TO>::value, T>::type* = nullptr>
DevPtr<TO> convertPng(const DevPtr<T>& input)
{
    DevPtr<TO> output(input.w, input.h); // Allocates
    cu_Convert<T, TO>(output, input);
    return output;
}

template <typename T, typename TO, typename 
    std::enable_if<is_same_type<T, TO>::value && !has_same_channels<T, TO>::value && 
                   (has_0_channels<TO>::value || has_1_channels<TO>::value), T>::type* = nullptr>
DevPtr<TO> convertPng(const DevPtr<T>& input)
{
    DevPtr<TO> output(input.w, input.h); // Allocates
    cu_ColorToGray<T, TO>(output, input);
    return output;
}

template <typename T, typename TO, typename 
    std::enable_if<is_same_type<T, TO>::value && !has_same_channels<T, TO>::value && 
                   (has_4_channels<TO>::value || has_3_channels<TO>::value), T>::type* = nullptr>
DevPtr<TO> convertPng(const DevPtr<T>& input)
{
    DevPtr<TO> output(input.w, input.h); // Allocates
    cu_ColorToColor<T, TO>(output, input);
    return output;
}

DevPtr<float> convertPng(const DevPtr<uchar4>& input)
{
    
}

DevPtr<TO> readPng(const std::string fileName)
{
    assert(type(fileName) == "PNG");

    size_t w, h, c;
    std::vector<uchar> image = readPng(fileName, w, h, c); 
    assert(c == 4);

    DevPtr<uchar4> devptr = upload<char, char4>(image.data(), w, h, c);
    DevPtr<TO> output = convertPng<TO>(devPtr);
    devptr.free();
    return output;
}

template<typename TO, typename std::enable_if<is_float_type<TO>::value, TO>::type* = nullptr>
DevPtr<TO> FileReader::read(const std::string fileName, size_t& width, size_t& height, size_t& c) const
{
    std::string fileType = fileName.substr(fileName.find_last_of(".") + 1);
    std::transform(fileType.begin(), fileType.end(), fileType.begin(), ::toupper);

    assert(fileType == "EXR");

    std::vector<float> image = readExr(fileName, width, height, c);
    auto devptr = upload<float, float>(image.data(), width, height, c);
    return convert<float, TO>(devptr);
}

template<typename TO, typename std::enable_if<!is_float_type<TO>::value, TO>::type* = nullptr>
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
    std::enable_if<std::is_same<T, TO>::value, T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input)
{   
    DevPtr<TO> output(input);
    return output;
}

/**
 * Same channels, convert only type
 */
template <typename T, typename TO, typename
    std::enable_if<!std::is_same<T, TO>::value && !is_same_type<T, TO>::value && has_same_channels<T, TO>::value, T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input)
{  
    DevPtr<TO> output(input.width, input.height); // Allocates
    cu_Convert<T, TO>(output, input);
    return output;
}

/**
 * Same type, convert color to gray
 */
template <typename T, typename TO, typename
    std::enable_if<is_same_type<T, TO>::value && 
       ((has_4_channels<T>::value && has_1_channels<TO>::value) ||
        (has_3_channels<T>::value && has_1_channels<TO>::value) ||
        (has_4_channels<T>::value && has_0_channels<TO>::value) ||
        (has_3_channels<T>::value && has_0_channels<TO>::value))
    , T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input)
{
    DevPtr<TO> output(input.width, input.height); // Allocates
    cu_ColorToGray<T, TO>(output, input);
    return output;
}

/**
 * Same type, convert gray to color
 */
template <typename T, typename TO, typename
    std::enable_if<is_same_type<T, TO>::value && 
       ((has_0_channels<T>::value && has_3_channels<TO>::value) ||
        (has_0_channels<T>::value && has_4_channels<TO>::value) ||
        (has_1_channels<T>::value && has_3_channels<TO>::value) ||
        (has_1_channels<T>::value && has_4_channels<TO>::value))
    , T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input)
{
    DevPtr<TO> output(input.width, input.height); // Allocates
    cu_GrayToColor<T, TO>(output, input);
    return output;
}

/**
 * Same type, convert color to color
 */
template <typename T, typename TO, typename
    std::enable_if<is_same_type<T, TO>::value && 
       ((has_4_channels<T>::value && has_3_channels<TO>::value) ||
        (has_3_channels<T>::value && has_4_channels<TO>::value))
    , T>::type* = nullptr>
DevPtr<TO> convertDevPtr(const DevPtr<T>& input)
{
    DevPtr<TO> output(input.width, input.height); // Allocates
    cu_ColorToColor<T, TO>(output, input);
        return output;
}

template <typename T, typename TO>
DevPtr<TO> FileReader::convert(const DevPtr<T>& input) const
{
    return convertDevPtr<T, TO>(input);
}