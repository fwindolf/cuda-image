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

    std::string depth_file = std::string(SOURCE_DIR) + "/examples/data/depth.exr";
    Image<float> depth(depth_file);

    depth.createWindow<DEPTH_TYPE>("Depth");
    depth.show();

    std::cout << "Depth image (" << depth.width() << "x" << depth.height() << ")" << std::endl;

    std::cout << "num = " << depth.size() << std::endl;
    std::cout << "valid = " << depth.valid() << std::endl;
    std::cout << "nonzero = " << depth.nonzero() << std::endl;
   
    std::cout << "min = " << depth.min() << std::endl;
    std::cout << "max = " << depth.max() << std::endl;
    std::cout << "mean = " << depth.mean() << std::endl;
    std::cout << "norm1 = " << depth.norm1() << std::endl;
    std::cout << "norm2 = " << depth.norm2() << std::endl;

    depth.threshold(1400, 0);
    depth.show();
}