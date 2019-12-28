#include "cuimage/image.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cctype>

using namespace cuimage;

int main(int argc, char** argv)
{
    // Init context
    cudaSafeCall(cudaSetDevice(0));
    cudaFree(0);

    if (argc < 3)
    {
        std::cerr << "Invalid number of parameters!" << std::endl;
        return -1;
    }

    std::string directory = argv[1];

    std::cout << "Reading from directory " << directory << std::endl;

    for (int i = 2; i < argc; i+=2)
    {
        char vis = argv[i][0];
        if (vis != 'c' && vis != 'd' && vis != 'h' && vis != 'u')
        {
            std::cerr << "Invalid visualization type " << vis << "!" << std::endl;
            continue;
        }

        std::string file = argv[i + 1];

        auto pos = file.rfind('.');
        if (pos == file.npos)
        {
            std::cerr << "Invalid file!" << std::endl;
            continue;
        }

        std::string type = file.substr(pos);
        std::transform(type.begin(), type.end(), type.begin(),
            [](unsigned char c){ return std::tolower(c); });

        std::cout << "Opening image of type " << type << std::endl;
        Image<float> img(directory + "/" + file);
        std::cout << "Image has size (" << img.width() << "x" << img.height() << ")" << std::endl;

        if (vis == 'h')
        {
            std::cout << "Saving as half-size to " << directory + "/" + "halved" + type << std::endl;
            auto img_h = img.resized(0.5, cuimage::LINEAR_NONZERO);
            img_h.replace(0, std::nanf(""));
            if (type == ".exr")
                img_h.save(directory + "/" + "halved" + type);
            else
                img_h.save(directory + "/" + "halved.exr");
        }
        if (vis == 'u')
        {
            std::cout << "Saving as double-size to " << directory + "/" + "doubled" + type << std::endl;
            auto img_h = img.resized(2, cuimage::LINEAR_NONZERO);
            if (type == ".exr")
                img_h.save(directory + "/" + "doubled" + type);
            else
                img_h.save(directory + "/" + "doubled.exr");   
        }

        if (type == ".exr")
        {
            std::cout << "Depth has range (" << img.min() << "x" << img.max() << ")" << std::endl;
            if (vis == 'c')
                img.show<cuimage::COLOR_TYPE_GREY_F>(file);
            else if(vis == 'd')
                img.show<cuimage::DEPTH_TYPE>(file);
        }
        else if (type == ".png")
        {
            Image<float3> img(directory + "/" + file);
            if (vis == 'c')
            {
                img.show<cuimage::COLOR_TYPE_RGB_F>(file);
            }
            else if(vis == 'd')
            {
                auto d = img.asGray<float>() / 1000.f;
                std::cout << "Depth has range (" << d.min() << "x" << d.max() << ")" << std::endl;
                d.show<cuimage::DEPTH_TYPE>(file);
            }
        }
    }

    return 0;
}