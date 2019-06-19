template <typename T>
FileWriter<T>::FileWriter(
    const std::string& fileName, const int width, const int height)
    : fileName_(fileName)
    , width_(width)
    , height_(height)
    , type_(fileName.substr(fileName.find(".") + 1))
{
}

/**
 * Incomplete writes using OpenCV
 * - float, float3
 * - uchar, uchar3, uchar4
 * TODO: Add less dirty version of re-scaling
 */

template <> 
bool FileWriter<float>::write(float* data)
{
    cv::Mat img(height_, width_, CV_32FC1, data);
    if (type_ == "png" || type_ == "jpg")
        img *= 255.f;

    cv::imwrite(fileName_, img);
}

template <>
bool FileWriter<float3>::write(float3*  data)
{
    cv::Mat img(height_, width_, CV_32FC3, data);
    if (type_ == "png" || type_ == "jpg")
        img *= 255.f;

    cv::imwrite(fileName_, img);
}

template <> 
bool FileWriter<uchar>::write(uchar* data)
{
    cv::Mat img(height_, width_, CV_8UC1, data);
    cv::imwrite(fileName_, img);
}

template <>
bool FileWriter<uchar3>::write(uchar3* data)
{
    cv::Mat img(height_, width_, CV_8UC3, data);
    cv::imwrite(fileName_, img);
}

template <>
bool FileWriter<uchar4>::write(uchar4* data)
{
    cv::Mat img(height_, width_, CV_8UC4, data);
    cv::imwrite(fileName_, img);
}
