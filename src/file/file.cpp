#include "cuimage/file/file_cu.h"

#include "cuimage/cuda/utils.h"

namespace cuimage
{

void* cu_Upload(const void* h_data, const size_t sizeBytes)
{
    void* dst;
    cudaSafeCall(cudaMalloc(&dst, sizeBytes));
    cudaSafeCall(cudaMemcpy(dst, h_data, sizeBytes, cudaMemcpyHostToDevice));
    return dst;
}
    
} // image
