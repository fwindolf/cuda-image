/**
 * @file   file.h
 * @brief  Loading image data from file
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/devptr.h"
#include "cuimage/file/file.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace cuimage
{

/**
 * @class FileReader
 * @brief Read images from file
 */
template <typename T> class FileReader : public File
{
public:
    ~FileReader(){};

    DevPtr<T> read(const std::string& fileName);

private:
    DevPtr<T> upload(cv::Mat& image);
};

#include "reader_impl.h"

} // image
