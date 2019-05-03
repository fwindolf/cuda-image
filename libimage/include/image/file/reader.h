/**
 * @file   file.h
 * @brief  Loading image data from file
 * @author Florian Windolf
 */
#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include <stdexcept>
#include <exception>

#include "image/cuda/type.h"
#include "image/cuda/devptr.h"

#include "image/operations/conversions_cu.h"

namespace image
{

/**
 * @class FileReader
 * @brief Read images from file
 */
class FileReader
{
public:
    FileReader();
    ~FileReader(){};

    template<typename TO>
    DevPtr<TO> read(const std::string fileName, size_t& width, size_t& height, size_t& c) const;

private:
    void* upload(const void* src, const size_t dataBytes) const;

    template <typename T, typename TO>
    DevPtr<TO> upload(const T* h_data, const size_t width, const size_t height, const size_t channels) const;

    template <typename T, typename TO>
    DevPtr<TO> convert(const DevPtr<T>& input) const;

    template <typename T, typename TO>
    TO* uploadAs(T* data, const size_t width, const size_t height, const size_t chans) const;

    template <typename T, typename TO>
    TO* uploadReinterpret(T* data, const size_t width, const size_t height) const;

    template <typename T, typename TO>
    TO* uploadConvert(T* data, const size_t width, const size_t height) const;
    
    std::vector<float> readExr(const std::string fileName, size_t& width, size_t& height, size_t& channels) const;

    std::vector<unsigned char> readPng(const std::string fileName, size_t& width, size_t& height, size_t& channels) const;
};


template<typename TO>
DevPtr<TO> FileReader::read(const std::string fileName, size_t& width, size_t& height, size_t& c) const
{
    std::string fileType = fileName.substr(fileName.find_last_of(".") + 1);
    std::transform(fileType.begin(), fileType.end(), fileType.begin(), ::toupper);

    if (fileType == "PNG")
    {        
        std::vector<uchar> image = readPng(fileName, width, height, c);
        auto devptr = upload<uchar, uchar4>(image.data(), width, height, c);
        return convert<uchar4, TO>(devptr);
    }
    else
    {
        std::vector<float> image = readExr(fileName, width, height, c);
        auto devptr = upload<float, float>(image.data(), width, height, c);
        return convert<float, TO>(devptr);
    }
    throw std::runtime_error("Cannot convert from file type T");
}

template <typename T, typename TO > //, typename std::enable_if<is_same_base<T, TO>::value, T>::type* = nullptr>
DevPtr<TO> FileReader::upload(const T* h_data, const size_t width, const size_t height, const size_t chans) const
{
    // If the format matches, upload to gpu
    assert(chans == channels<T>());
    TO* d_data = (TO*)upload((void*)h_data, width * height * sizeof(T));

    return DevPtr<TO>(d_data, width, height);
}

template <typename T, typename TO>
DevPtr<TO> FileReader::convert(const DevPtr<T>& input) const
{   
    if (is_same_base<T, TO>() && is_same_channels<T, TO>())
    {
        DevPtr<TO> output(input);
        return output;
    }
    else if (is_same_channels<T, TO>())
    {  
        DevPtr<TO> output(input.width, input.height); // Allocates
        cu_Convert<T, TO>(output, input);
        return output;
    }
    else if (is_same_base<T, TO>())
    {
        DevPtr<TO> output(input.width, input.height); // Allocates
        if (channels<TO>() >= 3 && channels<T>() == 1)
        {
            cu_GrayToColor(output, input);
            return output;
        }
        else if (channels<TO>() == 1 && channels<T>() >= 3)
        {           
            cu_ColorToGray(output, input);
            return output;
        }
    }

    throw std::runtime_error("Invalid output format or direct conversion not possible!");    
}

/*



template <typename T, typename TO>
TO* FileReader::uploadAs(T* data, const size_t width, const size_t height, const size_t chans) const
{
    if (is_same<T, TO>() && chans == channels<T>() || 
        is_same_base<T, TO>() && chans == channels<TO>())
    {
        // eg data is 3xuchar and will be uchar3
        return (TO*)upload(data, width * height * sizeof(TO));
    }
    else if(is_same_base<T, TO>())
    {
        // eg data is uchar and will be uchar3
        return uploadConvert<T, TO>(data, width, height);
    }
    else
    {
        // eg data is float and will be uchar3
        return uploadReinterpret<T, TO>(data, width, height);
    }
}

template <typename T, typename TO>
TO* FileReader::uploadReinterpret(T* data, const size_t width, const size_t height) const
{
    size_t channelsT = channels<T>();
    size_t channelsTO = channels<TO>();
    
    if (is_float<TO>())
    {
        std::vector<float> data_re(&data[0], &data[width * height * channelsTO]);
        if (is_same_channels<T, TO>())
            return (TO*)upload((void*)data_re.data(), width * height * sizeof(TO));
        else
            return uploadConvert<float, TO>(data_re.data(), width, height);
    }
    else if (is_uchar<TO>())
    {
        std::vector<uchar> data_re(&data[0], &data[width * height * channelsTO]);
        if (is_same_channels<T, TO>())
            return (TO*)upload((void*)data_re.data(), width * height * sizeof(TO));
        else
            return uploadConvert<uchar, TO>(data_re.data(), width, height);
    }
    else if (is_int<TO>())
    {
        std::vector<int> data_re(&data[0], &data[width * height * channelsTO]);
        if (is_same_channels<T, TO>())
            return (TO*)upload((void*)data_re.data(), width * height * sizeof(TO));
        else
            return uploadConvert<int, TO>(data_re.data(), width, height);
    }
    else
    {
        throw std::runtime_error("Invalid output type!");
    }
}

template <typename T, typename TO>
TO* FileReader::uploadConvert(T* data, const size_t width, const size_t height) const
{

    size_t channelsT = channels<T>();
    size_t channelsTO = channels<TO>();

    if (channelsT == 1 && channelsTO == 3)
        return cu_greyToRgb<TO>(data, width, height);
    else if (channelsT == 1 && channelsTO == 4)
        return cu_greyToRgba<TO>(data, width, height);
    else if (channelsT == 3 && channelsTO == 1)
        return cu_rgbToGrey<TO>(data, width, height);
    else if (channelsT == 4 && channelsTO == 1)
        return cu_rgbaToGrey<TO>(data, width, height);
}
    

// Specialization for T=float
template<>
float* FileReader::read(const std::string fileName, size_t& width, size_t& height, size_t& channels) const
{
    std::string fileType = fileName.substr(fileName.find_last_of(".") + 1);
    std::transform(fileType.begin(), fileType.end(), fileType.begin(), ::toupper);

    if (fileType == "PNG")
    {        
        auto vec_uchar = readPng(fileName, width, height, channels);
        return std::vector<float>(vec_uchar.begin(), vec_uchar.end());
    }        
    else if (fileType == "EXR")
    {
        return readExr(fileName, width, height, channels);
    }
    else
        throw std::runtime_error("Unsupported file type " + fileType);
}


// Specialization for T=uchar
template<>
inline std::vector<unsigned char> FileReader::read(const std::string fileName, size_t& width, size_t& height, size_t& channels) const
{
    std::string fileType = fileName.substr(fileName.find_last_of(".") + 1);
    std::transform(fileType.begin(), fileType.end(), fileType.begin(), ::toupper);

    if (fileType == "PNG")
    {        
        return readPng(fileName, width, height, channels);
    }        
    else if (fileType == "EXR")
    {
        auto vec_float =  readExr(fileName, width, height, channels);
        return std::vector<unsigned char>(vec_float.begin(), vec_float.end());
    }
    else
        throw std::runtime_error("Unsupported file type " + fileType);
}
*/


} // image
