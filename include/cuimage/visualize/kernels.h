#include "cuimage/cuda/devptr.h"

namespace cuimage
{

void cu_VerticesFromDepth(DevPtr<float4>& vertices, const DevPtr<float>& depth,
    const float fx, const float fy, const float cx, const float cy);

void cu_VerticesFromDepthGradients(DevPtr<float4>& vertices,
    const DevPtr<float3>& depth_gradients, const float fx, const float fy,
    const float cx, const float cy);

template <typename T>
void cu_VertexIndices(DevPtr<uint>& indices, const DevPtr<T>& data);

} // namespace cuimage