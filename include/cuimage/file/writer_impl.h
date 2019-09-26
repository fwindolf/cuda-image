template <typename T>
FileWriter<T>::FileWriter(
    const std::string& fileName, const int width, const int height)
    : fileName_(fileName)
    , width_(width)
    , height_(height)
    , type_(fileName.substr(fileName.rfind(".") + 1))
{
}

/**
 * Incomplete writes using OpenCV
 * - float, float3
 * - uchar, uchar3, uchar4
 * TODO: Add less dirty version of re-scaling
 */

template <> inline bool FileWriter<float>::write(float* data)
{
    cv::Mat img(height_, width_, CV_32FC1, data);
    if (type_ == "png" || type_ == "jpg")
        img.convertTo(img, CV_8UC1, 255.f);

    return cv::imwrite(fileName_, img);
}

template <> inline bool FileWriter<int>::write(int* data)
{
    cv::Mat img(height_, width_, CV_32SC1, data);
    if (type_ == "png" || type_ == "jpg")
        img *= 255.f;

    return cv::imwrite(fileName_, img);
}

template <> inline bool FileWriter<float3>::write(float3* data)
{
    cv::Mat img(height_, width_, CV_32FC3, data);
    if (type_ == "png" || type_ == "jpg")
        img.convertTo(img, CV_8UC3, 255.f);

    return cv::imwrite(fileName_, img);
}

template <> inline bool FileWriter<uchar>::write(uchar* data)
{
    cv::Mat img(height_, width_, CV_8UC1, data);
    return cv::imwrite(fileName_, img);    
}

template <> inline bool FileWriter<uchar3>::write(uchar3* data)
{
    cv::Mat img(height_, width_, CV_8UC3, data);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    return cv::imwrite(fileName_, img);
}

template <> inline bool FileWriter<uchar4>::write(uchar4* data)
{
    cv::Mat img(height_, width_, CV_8UC4, data);
    cv::cvtColor(img, img, cv::COLOR_RGBA2BGR);
    return cv::imwrite(fileName_, img);
}
