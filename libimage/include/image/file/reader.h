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

#include "image/file/file_cu.h"

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
    template <typename T, typename TO>
    DevPtr<TO> upload(const T* h_data, const size_t width, const size_t height, const size_t channels) const;

    template <typename T, typename TO>
    DevPtr<TO> convert(const DevPtr<T>& input) const;


    std::vector<float> readExr(const std::string fileName, size_t& width, size_t& height, size_t& channels) const;

    std::vector<unsigned char> readPng(const std::string fileName, size_t& width, size_t& height, size_t& channels) const;
};

#include "reader_impl.h"

} // image
