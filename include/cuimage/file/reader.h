/**
 * @file   file.h
 * @brief  Loading image data from file
 * @author Florian Windolf
 */
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cuimage/cuda/devptr.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace cuimage
{

/**
 * @class FileReader
 * @brief Read images from file
 */
template <typename T>
class FileReader
{
public:
    ~FileReader(){};

    DevPtr<T> read(const std::string& fileName);

private:
    DevPtr<T> upload(cv::Mat& image);
};

#include "reader_impl.h"

} // image
