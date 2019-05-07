#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuimage/image.h"

using namespace cuimage;

int main(int argc, char** argv)
{
    // Init context
    cudaSafeCall(cudaSetDevice(0));
    cudaFree(0);

    std::string color_file = std::string(SOURCE_DIR) + "/examples/data/image.png";
    Image<uchar3> color(color_file);

    std::cout << "Color image (" << color.width() << "x" << color.height() << ")" << std::endl;

    std::cout << "min = " << color.min() << std::endl 
        << "max = " << color.max() << std::endl
        << "mean =" << color.sum() / color.size() << std::endl;

    color.show("Color", COLOR_TYPE_RGB);
}