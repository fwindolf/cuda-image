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

 /**
 * Read PNG as color
 */
DevPtr<uchar4> FileReader::readPng(const std::string fileName)
{
    assert(type(fileName) == "PNG");

    size_t w, h;
    auto image = readPng(fileName, w, h, 4, 8);
    return upload<uchar, uchar4>(image.data(), w, h, 4);
}

/**
 * Read PNG as color
 */
DevPtr<float3> FileReader::readPngF(const std::string fileName)
{
    assert(type(fileName) == "PNG");

    size_t w, h;
    auto raw = readPng(fileName, w, h, 3, 16);

    // 16 bit per pixel in big endian -> reorder
    std::vector<float> image(w * h * 3);
    uchar components[2];
    for (int i = 0; i < w * h * 3; i++)
    {
        // Make little endian
        components[0] = raw[2 * i + 1];
        components[1] = raw[2 * i + 0];
        unsigned short val = *((unsigned short*) components);
        image[i] = static_cast<float>(val) / (255.f * 255.f);
    }

    return upload<float, float3>(image.data(), w, h, 3);
}

/**
 * Read PNG as greyscale
 */
DevPtr<uchar> FileReader::readPngGrey(const std::string fileName)
{
    assert(type(fileName) == "PNG");

    DevPtr<uchar4> tmp = readPng(fileName);
    DevPtr<uchar> output(tmp.width, tmp.height);
    cu_ColorToGray<uchar4, uchar>(output, tmp);
    tmp.free();
    return output;
}

/**
 * Read PNG as greyscale
 */
DevPtr<float> FileReader::readPngGreyF(const std::string fileName)
{
    assert(type(fileName) == "PNG");

    size_t w, h;
    auto raw = readPng(fileName, w, h, 1, 16);

    // 16 bit per pixel in big endian -> reorder
    std::vector<float> image(w * h);
    uchar components[2];
    for (int i = 0; i < w * h; i++)
    {
        // Make little endian
        components[0] = raw[2 * i + 1];
        components[1] = raw[2 * i + 0];
        unsigned short val = *((unsigned short*) components);
        image[i] = static_cast<float>(val) / (255.f * 255.f);
    }

    return upload<float, float>(image.data(), w, h, 1);
}


/**
 * Read single channel EXR
 */
DevPtr<float> FileReader::readExr(const std::string fileName) const
{
    size_t w, h, c;
    std::vector<float> image = readExr(fileName, w, h, c);
    return upload<float, float>(image.data(), w, h, c);
}

std::string FileReader::type(const std::string fileName)
{
    std::string fileType = fileName.substr(fileName.find_last_of(".") + 1);
    std::transform(fileType.begin(), fileType.end(), fileType.begin(), ::toupper);
    return fileType;
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


std::vector<unsigned char> FileReader::readPng(const std::string fileName, size_t& width, size_t& height, size_t channels, unsigned bitdepth) const
{
    std::vector<unsigned char> image; // the raw pixels

    LodePNGColorType type;
    
    if (channels == 1) 
        type = LCT_GREY;
    else if (channels == 3) 
        type = LCT_RGB;
    else if (channels = 4)
        type = LCT_RGBA;

    unsigned w, h;

    auto error = lodepng::decode(image, w, h, fileName, type, bitdepth);
    if(error)
        std::cerr << "Could not load " << fileName << ": " << lodepng_error_text(error) << std::endl;

    width = w;
    height = h;
    return image;
}
