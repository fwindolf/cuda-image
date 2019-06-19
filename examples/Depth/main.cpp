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

    std::string depth_png = std::string(SOURCE_DIR) + "/examples/data/depth.png";
    Image<float> depth_lq(depth_png);
    depth_lq.show<DEPTH_TYPE>("Depth Low Quality");

    Image<float> depth_mask(depth_lq);
    depth_mask.threshold(0.1f, 0.f, 1.f); // Everything that has a valid value is set to 1, else to 0
    Image<uchar> mask = depth_mask.as<uchar>() * (uchar)255;
    mask.show<COLOR_TYPE_GREY>("Mask");
    
    depth_lq.resize(0.4, mask);
    depth_lq.show<DEPTH_TYPE>("Depth LQ Downsampled");

    mask.resize(0.4, mask);
    depth_lq.resize(2.f, mask, LINEAR_NONZERO);
    depth_lq.show<DEPTH_TYPE>("Depth LQ Upsampled");

    std::string depth_file = std::string(SOURCE_DIR) + "/examples/data/depth.exr";
    Image<float> depth(depth_file);

    depth.createWindow<DEPTH_TYPE>("Depth");
    depth.show(true); // blocks

    std::cout << "Depth image (" << depth.width() << "x" << depth.height() << ")" << std::endl;

    std::cout << "num = " << depth.size() << std::endl;
    std::cout << "valid = " << depth.valid() << std::endl;
    std::cout << "nonzero = " << depth.nonzero() << std::endl;
   
    std::cout << "min = " << depth.min() << std::endl;
    std::cout << "max = " << depth.max() << std::endl;
    std::cout << "mean = " << depth.mean() << std::endl;
    std::cout << "norm1 = " << depth.norm1() << std::endl;
    std::cout << "norm2 = " << depth.norm2() << std::endl;

    for(int i = depth.max(); i > 800; i--)
    {
        depth.threshold(i, 0);
        depth.show();
    }   

    depth.closeWindow(true);
}