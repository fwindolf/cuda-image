#include "cuimage/file/reader.h"
#include <iostream>
#include <array>
#include <algorithm>

#include "cuimage/cuda/utils.h"
#include "cuimage/cuda/kernel.h"

using namespace cuimage;

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#else
#include "lodepng.h"
#endif

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

FileReader::FileReader()
{
}

std::vector<float> FileReader::readExr(const std::string fileName, size_t& width, size_t& height, size_t& channels) const
{
    EXRVersion exr_version;

    std::vector<float> image;

    int ret = ParseEXRVersionFromFile(&exr_version, fileName.c_str());
    if (ret != 0)
    {
        std::cerr << "Invalid EXR file: " << fileName << std::endl;
        return image;
    }

    if (exr_version.multipart)
    {
        // must be multipart flag is true.
        std::cerr << "Invalid EXR file, " << fileName << " is multipart" << std::endl;
        return image;
    }

    // 2. Read EXR header
    EXRHeader exr_header;
    InitEXRHeader(&exr_header);

    const char *err = nullptr; 
    ParseEXRHeaderFromFile(&exr_header, &exr_version, fileName.c_str(), &err);
    if (err)
    {
        std::cerr << "Parse EXR failed for file " << fileName << ", err: " << err << std::endl;
        FreeEXRErrorMessage(err); // free's buffer for an error message
        return image;
    }

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    LoadEXRImageFromFile(&exr_image, &exr_header, fileName.c_str(), &err);
    if (err)
    {
        std::cerr << "Loading EXR failed for file " << fileName << ", err: " << err << std::endl;
        FreeEXRHeader(&exr_header);
        FreeEXRErrorMessage(err); // free's buffer for an error message
        return image;
    }

    // 3. Access image data
    height = exr_image.height;
    width = exr_image.width;
    channels = exr_image.num_channels;

    // Copy image to vector
    if(!exr_header.tiled && exr_image.images)
    {
        if (channels == 1)
        {
            // Row format (R, R, R, R, ... )
            image.resize(width * height);
            float val;
            for (int i = 0; i < width * height; i++)
            {
                val = reinterpret_cast<float **>(exr_image.images)[0][i];
                if (val < 0 || val > 1e12)
                    val = 0;

                // Add to array
                image.at(i) = val;
            }
        }
        else if(channels == 3)
        {
            // Scanline format (RGBA, RGBA, ...)
            image.resize(width * height * 3);     
            float val, val_r = 0, val_g = 0, val_b = 0;
            for (int i = 0; i < width * height; i++)
            {
                for (int c = 0; c < channels; c++)
                {
                    std::string c_name = exr_header.channels[c].name;
                    val = reinterpret_cast<float **>(exr_image.images)[c][i];
                    if (val < 0 || val > 1e12)
                        val = 0;

                    if (c_name == "R")
                        val_r = val;
                    else if (c_name =="G")
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
    else if(exr_header.tiled && exr_image.tiles)
    {
        // tiled format (R, R, ..., G, G ..., ...)
        std::cerr << "EXR in unsupported tiled format" << std::endl;            
        return image;
    }
    else
    {
        std::cerr << "EXR file " << fileName << " has not/invalid content" << std::endl;
        return image;
    }

    return image;
}


std::vector<unsigned char> FileReader::readPng(const std::string fileName, size_t& width, size_t& height, size_t& channels) const
{
#if HAVE_OPENCV
    std::vector<unsigned char> image; // Raw pixels

    auto m = cv::imread(fileName);
    assert(!m.empty());

    // To Uchar
    if (m.depth() != CV_8U)
        m.convertTo(m, CV_8U);
    
    width = m.cols;
    height = m.rows;
    channels = m.channels();
    
    if (channels >= 3)
    {   
        std::vector<cv::Mat> vec;
        cv::split(m, vec);

        // BGR -> RGB        
        std::swap(vec[0], vec[2]); 

        cv::merge(vec, m);
    }    
    image.assign(m.data, m.data + width * height * channels);
    
    return image;    
#else
    std::vector<unsigned char> image; //the raw pixels
    unsigned w, h;
    
    unsigned error = lodepng::decode(image, w, h, fileName);
    if (error)
    {
        std::cerr << "Could not load " << fileName << ": " << lodepng_error_text(error) << std::endl;
        return image;
    }

    width = w;
    height = h;
    channels = image.size() / (w * h);

    return image;
#endif
}
