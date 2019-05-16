#include "cuimage/visualize/kernels.h"

#include "cuimage/cuda/kernel.h"
#include "cuimage/cuda/conversion.h"
#include "cuimage/cuda/arithmetic.h"
#include "cuimage/cuda/depth.h"


namespace cuimage
{

__global__ void g_VerticesFromDepth(DevPtr<float4> vertices, DevPtr<float> data, 
                                    const float fx, const float fy, 
                                    const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= data.width || pos.y >= data.height)
        return;

    const float d = data(pos.x, pos.y);    

    // Backproject
    float4 v = d_backproject(d, pos.x, pos.y, fx, fy, cx, cy);

    // Normals    
    const float2 dg = d_gradient_forward(data, pos.x, pos.y);
    float4 n = d_normals(make_float3(d, dg.x, dg.y), pos.x, pos.y, fx, fy, cx, cy);

    const int idx = getIndex(pos, data.width, data.height);
    vertices(2 * idx) = v;
    vertices(2 * idx + 1) = n;
}

__global__ void g_VerticesFromDepthGradients(DevPtr<float4> vertices, DevPtr<float3> data, 
                                             const float fx, const float fy, 
                                             const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= data.width || pos.y >= data.height)
    return;

    const float3 d = data(pos.x, pos.y);    

    // Backproject
    float4 v = d_backproject(d.x, pos.x, pos.y, fx, fy, cx, cy);

    // Normals    
    float4 n = d_normals(d, pos.x, pos.y, fx, fy, cx, cy);

    const int idx = getIndex(pos, data.width, data.height);
    vertices(2 * idx) = v;
    vertices(2 * idx + 1) = n;
}


struct Triangle
{
    __device__ __host__ Triangle(const uint val) 
     : c1(val), c2(val), c3(val) {}
    
    uint c1 = 0, c2 = 0, c3 = 0;
};

// ◤ Filling left upper corner (corner 3 of quad is missing)
__device__ void d_generate_triangle_1(Triangle& t, const uint idx, const int w)
{
    t.c3 = idx;
    t.c2 = idx + 1;
    t.c1 = idx + w;
}

// ◥ Filling right upper corner (corner 4 of quad is missing)
__device__ void  d_generate_triangle_2(Triangle& t, const uint idx, const int w)
{
    t.c3 = idx;
    t.c2 = idx + 1;
    t.c1 = idx + w + 1;
}

// ◢ Filling right lower corner (corner 1 of quad is missing)
__device__ void  d_generate_triangle_3(Triangle& t, const uint idx, const int w)
{
    t.c3 = idx;
    t.c2 = idx + w + 1;
    t.c1 = idx + w;
}

// ◣ Filling left lower corner (corner 2 of quad is missing)
__device__ void  d_generate_triangle_4(Triangle& t, const uint idx, const int w)
{
    t.c3 = idx + 1;
    t.c2 = idx + w + 1;
    t.c1 = idx + w;
}


template <typename T>
__global__ void g_VertexIndicesTriangles(DevPtr<uint> indices, const DevPtr<T> data)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);
    const int idx = getIndex(pos, data.width, data.height);

    if (pos.x >= data.width - 1 || pos.y >= data.height - 1)
        return;

    T d[4];
    d[0] = data(pos.x, pos.y);
    d[1] = data(pos.x + 1, pos.y);
    d[2] = data(pos.x, pos.y + 1);
    d[3] = data(pos.x + 1, pos.y + 1);

    int valid = 0;
    float df[4];
    float minv = 1e37f;
    float maxv = -1e37f;
    float sum = 0.f;
    for (int i = 0; i < 4; i++)
    {
        df[i] = d_get<T>(d[i], 0); // get 0th component if necessary
        if (df[i] > 0)
        {
            valid++;
            minv = min(df[i], minv);
            maxv = max(df[i], maxv);
            sum += df[i];
        }
    }

    // Discard triangles that would connect points very far apart
    if(maxv - minv >= 0.1f * sum / valid)
        return;

    const int w = data.width;
    const uint uidx = (uint) idx;
    Triangle t1(uidx), t2(uidx); // degenerate on creation

    // Generate triangles for valid points
    if (valid == 4)
    {
        // All points valid -> fill with two triangles (2 / 4)
        d_generate_triangle_2(t1, idx, w);
        d_generate_triangle_3(t2, idx, w);
    }
    else if (valid == 3)
    {
        // One point is missing -> only one triangle
        if (df[0] <= 0)
            d_generate_triangle_4(t1, idx, w);
        else if (df[1] <= 0)
            d_generate_triangle_3(t1, idx, w);
        else if (df[2] <= 0)
            d_generate_triangle_2(t1, idx, w);
        else if (df[3] <= 0)
            d_generate_triangle_1(t1, idx, w);
    }
    
    // Add to triangles
    indices(6 * idx + 0) = t1.c1;
    indices(6 * idx + 1) = t1.c2;
    indices(6 * idx + 2) = t1.c3;
    
    indices(6 * idx + 3) = t2.c1;
    indices(6 * idx + 4) = t2.c2;
    indices(6 * idx + 5) = t2.c3;
    
}


void cu_VerticesFromDepth(DevPtr<float4>& vertices,
                          const DevPtr<float>& depth, 
                          const float fx, const float fy, 
                          const float cx, const float cy)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth.width, depth.height, block);

    g_VerticesFromDepth <<< grid, block >>> (vertices, depth, fx, fy, cx, cy);

    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());   
}

void cu_VerticesFromDepthGradients(DevPtr<float4>& vertices,
                                   const DevPtr<float3>& depth_gradients, 
                                   const float fx, const float fy, 
                                   const float cx, const float cy)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth_gradients.width, depth_gradients.height, block);

    g_VerticesFromDepthGradients <<< grid, block >>> (vertices, depth_gradients, fx, fy, cx, cy);

    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());   
}

template <typename T>
void cu_VertexIndices(DevPtr<uint>& indices, const DevPtr<T>& data)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(data.width, data.height, block);

    g_VertexIndicesTriangles<T> <<< grid, block >>> (indices, data);
    
    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());   
}

/**
 * Explicit instantiation
 */
template void cu_VertexIndices(DevPtr<uint>& indices, const DevPtr<float>& data);
template void cu_VertexIndices(DevPtr<uint>& indices, const DevPtr<float3>& data);

} // cuimage