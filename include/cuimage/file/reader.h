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

#include "cuimage/cuda/type.h"
#include "cuimage/cuda/devptr.h"

#include "cuimage/file/file_cu.h"

#include "cuimage/operations/conversions_cu.h"

namespace cuimage
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

    /**
     * Read PNG as color
     */
    DevPtr<uchar4> readPng(const std::string fileName);
    /**
     * Read PNG as color
     */
    DevPtr<float3> readPngF(const std::string fileName);

    /**
     * Read PNG as greyscale
     */
    DevPtr<uchar> readPngGrey(const std::string fileName);

    /**
     * Read PNG as greyscale
     */
    DevPtr<float> readPngGreyF(const std::string fileName);   

    /**
     * Read single channel EXR
     */
    DevPtr<float> readExr(const std::string fileName) const;

    /**
     * Determine the file ending in uppercase
     */
    std::string type(const std::string fileName);

private:

    template <typename T, typename TO>
    DevPtr<TO> upload(const T* h_data, const size_t width, const size_t height, const size_t channels) const;

    std::vector<float> readExr(const std::string fileName, size_t& width, size_t& height, size_t& channels) const;

    std::vector<unsigned char> readPng(const std::string fileName, size_t& width, size_t& height, size_t& channels) const;
};


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

} // image
