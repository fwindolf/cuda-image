/**
 * @file   file.h
 * @brief  Writing image data from file
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/file/file.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>

namespace cuimage
{

/**
 * @class FileWriter
 * @brief Write images to  file
 */
template <typename T> class FileWriter : public File
{
public:
    FileWriter(const std::string& fileName, const int width, const int height);

    ~FileWriter(){};

    bool write(T* data);

private:
    const std::string fileName_;
    const int width_, height_;
    const std::string type_;
};

#include "writer_impl.h"

} // cuimage