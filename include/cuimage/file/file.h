#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace cuimage
{

class File
{
public:
    virtual ~File(){};

protected:
    /**
     * Read the file as EXR file
     */
    cv::Mat _readExr(const std::string& fileName);
};
}