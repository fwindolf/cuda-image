template <typename T> inline DevPtr<T> FileReader<T>::upload(cv::Mat& image)
{
    assert(image.isContinuous());
    assert(image.elemSize() == sizeof(T));
    assert(image.cols > 0 && image.rows > 0);

    DevPtr<T> d_data(image.cols, image.rows);
    cudaSafeCall(cudaMemcpy(d_data.data, image.data,
        d_data.width * d_data.height * sizeof(T), cudaMemcpyHostToDevice));
    return d_data;
}

/**
 * Incomplete reads using OpenCV
 * - float, float3
 * - uchar, uchar3, uchar4
 */

template <>
inline DevPtr<float> FileReader<float>::read(const std::string& fileName)
{
    auto type = fileName.substr(fileName.rfind(".") + 1);
    cv::Mat img;
    if (type == "exr")
        img = _readExr(fileName);
    else
        img = cv::imread(fileName, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

    if (img.empty())
        throw std::runtime_error(
            "Could not open image " + fileName + " or image is empty!");

    // Make float
    img.convertTo(img, CV_32FC1);

    if (type == "png")
    {
        // Estimate range and normalize
        double minv, maxv;
        cv::minMaxIdx(img, &minv, &maxv);
        if (minv >= 0 && maxv > 1 && maxv <= 255)
            img /= 255.f;
    }

    return upload(img);
}

template <>
inline DevPtr<float3> FileReader<float3>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (img.empty())
        throw std::runtime_error(
            "Could not open image " + fileName + " or image is empty!");

    img.convertTo(img, CV_32FC3);

    double minv, maxv;
    cv::minMaxIdx(img, &minv, &maxv);
    if (minv >= 0 && maxv > 1 && maxv <= 255)
        img /= 255.f;

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return upload(img);
}

template <>
inline DevPtr<uchar> FileReader<uchar>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        throw std::runtime_error(
            "Could not open image " + fileName + " or image is empty!");

    img.convertTo(img, CV_8UC1);
    return upload(img);
}

template <>
inline DevPtr<uchar3> FileReader<uchar3>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error(
            "Could not open image " + fileName + " or image is empty!");

    img.convertTo(img, CV_8UC3);
    return upload(img);
}

template <>
inline DevPtr<uchar4> FileReader<uchar4>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error(
            "Could not open image " + fileName + " or image is empty!");

    img.convertTo(img, CV_8UC3);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);
    return upload(img);
}
