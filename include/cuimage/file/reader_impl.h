cv::Mat readExr(const std::string& fileName)
{
    EXRVersion exr_version;

    std::vector<float> image;

    int ret = ParseEXRVersionFromFile(&exr_version, fileName.c_str());
    if (ret != 0)
    {
        std::cerr << "Invalid EXR file: " << fileName << std::endl;
        return cv::Mat();
    }

    if (exr_version.multipart)
    {
        // must be multipart flag is true.
        std::cerr << "Invalid EXR file, " << fileName << " is multipart"
                  << std::endl;
        return cv::Mat();
    }

    // 2. Read EXR header
    EXRHeader exr_header;
    InitEXRHeader(&exr_header);

    const char* err = nullptr;
    ParseEXRHeaderFromFile(&exr_header, &exr_version, fileName.c_str(), &err);
    if (err)
    {
        std::cerr << "Parse EXR failed for file " << fileName
                  << ", err: " << err << std::endl;
        FreeEXRErrorMessage(err); // free's buffer for an error message
        return cv::Mat();
    }

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    LoadEXRImageFromFile(&exr_image, &exr_header, fileName.c_str(), &err);
    if (err)
    {
        std::cerr << "Loading EXR failed for file " << fileName
                  << ", err: " << err << std::endl;
        FreeEXRHeader(&exr_header);
        FreeEXRErrorMessage(err); // free's buffer for an error message
        return cv::Mat();
    }

    // 3. Access image data
    // Copy image to vector
    if (!exr_header.tiled && exr_image.images)
    {
        if (exr_image.num_channels == 1)
        {
            // Row format (R, R, R, R, ... )
            image.resize(exr_image.width * exr_image.height);
            float val;
            for (int i = 0; i < image.size(); i++)
            {
                val = reinterpret_cast<float**>(exr_image.images)[0][i];
                if (val < 0 || val > 1e12)
                    val = 0;

                // Add to array
                image.at(i) = val;
            }
        }
        else if (exr_image.num_channels == 3)
        {
            // Scanline format (RGBA, RGBA, ...)
            image.resize(exr_image.width * exr_image.height * 3);
            float val, val_r = 0, val_g = 0, val_b = 0;
            for (int i = 0; i < exr_image.width * exr_image.height; i++)
            {
                for (int c = 0; c < 3; c++)
                {
                    std::string c_name = exr_header.channels[c].name;
                    val = reinterpret_cast<float**>(exr_image.images)[c][i];
                    if (val < 0 || val > 1e12)
                        val = 0;

                    if (c_name == "R")
                        val_r = val;
                    else if (c_name == "G")
                        val_g = val;
                    else if (c_name == "B")
                        val_b = val;
                }

                // Add to array
                image.at(3 * i + 0) = val_r;
                image.at(3 * i + 1) = val_g;
                image.at(3 * i + 2) = val_b;
            }
        }
    }
    else if (exr_header.tiled && exr_image.tiles)
    {
        // tiled format (R, R, ..., G, G ..., ...)
        std::cerr << "EXR in unsupported tiled format" << std::endl;
        return cv::Mat();
    }
    else
    {
        std::cerr << "EXR file " << fileName << " has not/invalid content"
                  << std::endl;
        return cv::Mat();
    }

    return cv::Mat(exr_image.height, exr_image.width,
        CV_32FC(exr_image.num_channels), image.data());
}

template <typename T> DevPtr<T> FileReader<T>::upload(cv::Mat& image)
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

template <> DevPtr<float> FileReader<float>::read(const std::string& fileName)
{
    auto type = fileName.substr(fileName.find(".") + 1);
    cv::Mat img;
    if (type == "exr")
        img = readExr(fileName);
    else
        img = cv::imread(fileName, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

    img.convertTo(img, CV_32FC1);
    return upload(img);
}

template <>
DevPtr<float3> FileReader<float3>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    img.convertTo(img, CV_32FC3);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return upload(img);
}

template <> DevPtr<uchar> FileReader<uchar>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_8UC1);
    return upload(img);
}

template <>
DevPtr<uchar3> FileReader<uchar3>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    img.convertTo(img, CV_8UC3);
    return upload(img);
}

template <>
DevPtr<uchar4> FileReader<uchar4>::read(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    img.convertTo(img, CV_8UC3);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);
    return upload(img);
}
