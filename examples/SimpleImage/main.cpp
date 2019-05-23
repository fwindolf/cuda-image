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

    color.show<COLOR_TYPE_RGB>("Color");
    std::cout << "Color image (" << color.width() << "x" << color.height() << ")" << std::endl;

    std::cout << "num = " << color.size() << std::endl;
    std::cout << "valid = " << color.valid() << std::endl;
    std::cout << "nonzero = " << color.nonzero() << std::endl;
   
    std::cout << "min = " << color.min() << std::endl;
    std::cout << "max = " << color.max() << std::endl;
    std::cout << "mean = " << color.mean() << std::endl;
    std::cout << "norm1 = " << color.norm1() << std::endl;
    std::cout << "norm2 = " << color.norm2() << std::endl;

    Image<uchar> color_component_b = color.get<uchar>(2);
    color_component_b.show<COLOR_TYPE_GREY>("Color B Component");
    
    Image<float> grey = color.asGray<uchar>().as<float>() / 255.f;
    grey.show<COLOR_TYPE_GREY_F>("Grey");    

    std::cout << "min = " << grey.min() << std::endl;
    std::cout << "max = " << grey.max() << std::endl;
    std::cout << "mean = " << grey.mean() << std::endl;
    //std::cout << "median = " << grey.median() << std::endl;
    std::cout << "norm1 = " << grey.norm1() << std::endl;
    std::cout << "norm2 = " << grey.norm2() << std::endl;
    
    Image<float> copy = grey.resized(400, 400);
    copy.show<COLOR_TYPE_GREY_F>("Resized");

    Image<uchar> mask = copy.as<uchar>() * (uchar)255;
    mask.threshold(255, 255, 0); // replace white area with 0s, everything grey with 255s
    copy.mask(mask);

    copy.show<COLOR_TYPE_GREY_F>("Masked");       

    copy.resize(3.f);
    copy.show<COLOR_TYPE_GREY_F>("Resized again");       
}