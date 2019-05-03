#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "image/image.h"

using namespace image;

int main(int argc, char** argv)
{
    
    std::string depth_file = std::string(SOURCE_DIR) + "/data/depth.exr";
    std::string color_file = std::string(SOURCE_DIR) + "/data/image.png";
    Image<float> depth(depth_file);
    Image<uchar3> color(color_file);

    color.show("Color", COLOR_TYPE_RGB);
}