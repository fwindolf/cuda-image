#include "image/visualize/kernels.h"

#include "image/cuda/kernel.h"
#include "image/cuda/arithmetic.h"

using namespace image;

__global__ void g_backProject(float* data, float4* vertices, const int w, const int h,
                              const float fx, const float fy, const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);
    const int idx = getIndex(pos, w, h);

    if (pos.x >= w || pos.y >= h)
        return;
    
    // Backproject
    const float x = ((float)pos.x - cx) / fx;
    const float y = ((float)pos.y - cy) / fy;
    const float d = data[idx];    
    float4 v = make_float4(x * d, y * d, d, 1.f);

    vertices[idx] = v;
}

__global__ void g_calculateNormals(float* data, float4* normals, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);
    const int idx = getIndex(pos, w, h);

    if (pos.x >= w || pos.y >= h)
        return;
    
    const float d = data[idx];    

    if (d <= 0)
    {
        normals[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    const float dx = pos.x < w - 1 ? data[idx + 1] - data[idx] : 0.f;
    const float dy = pos.y < h - 1 ? data[idx + w] - data[idx] : 0.f;
    float4 n = make_float4(dx * fx, dy * fy, -d - (pos.x - cx) * dx - (pos.y - cy) * dy, 1.f);

    normals[idx] = n;
}

__global__ void g_backProjectAndCalcNormals(float* data, float4* vertices, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);
    const int idx = getIndex(pos, w, h);

    if (pos.x >= w || pos.y >= h)
    return;

    // Backproject
    const float x = ((float)pos.x - cx) / fx;
    const float y = ((float)pos.y - cy) / fy;
    const float d = data[idx];    
    float4 v = make_float4(x * d, y * d, d, 1.f);

    // Normals
    float4 n;
    const float dx = pos.x < w - 1 ? data[idx + 1] - data[idx] : 0.f;
    const float dy = pos.y < h - 1 ? data[idx + w] - data[idx] : 0.f;
    n = make_float4(dx * fx, dy * fy, -d - (pos.x - cx) * dx - (pos.y - cy) * dy, 1.f);
    if (d <= 0)
        n = make_float4(0.f, 0.f, 0.f, 0.f);

    vertices[2 * idx] = v;
    vertices[2 * idx + 1] = n;
}



__global__ void g_generateVertexIndicesNoStrip(float* data, int3* indices, const int w, const int h)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);
    const int idx = getIndex(pos, w, h);

    if (pos.x >= w || pos.y >= h)
        return;

    int3 i_forward = make<int3>(0);
    int3 i_backward = make<int3>(0);

    // Forward facing triangles [x, x+1, x+w]
    if (pos.x < w - 1 && pos.y < h - 1)
    {
        i_forward.x = idx;
        i_forward.y = idx + 1;
        i_forward.z = idx + w;
    }

    // Backwards facing trianges [x, x+w, x+w-1]
    if (pos.x > 0 && pos.y < h - 1)
    {
        i_backward.x = idx;
        i_backward.y = idx + w;
        i_backward.z = idx + w - 1;
    }

    // Interleave (fwd_0, bwd_0, fwd_1, bwd_1, ...)
    indices[2 * idx] = i_forward;
    indices[2 * idx + 1] = i_backward;
}

__global__ void g_generateVertexIndicesTriangles(float* data, unsigned int* indices, const int w, const int h)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);
    const int idx = getIndex(pos, w, h);

    if (pos.x >= w - 1 || pos.y >= h - 1)
        return;

    // Vertices are numbered from 0

    float d[4];
    d[0] = data[idx];
    d[1] = data[idx + 1];
    d[2] = data[idx + w];
    d[3] = data[idx + w + 1];

    int valid = 0;
    for (int i = 0; i < 4; i++)
        valid += (d[i] > 0 ? 1 : 0);
    
    // There is room for 2 triangles, but initialize as degenerate
    unsigned int uidx = (unsigned int) idx;
    unsigned int t1[3] = { uidx, uidx, uidx };
    unsigned int t2[3] = { uidx, uidx, uidx };
    
    // 0 / 1 valid point -> no triangles possible
    if (valid == 4)
    {
        // 4 valid points -> 2 triangles
        t1[2] = idx;
        t1[1] = idx + 1;
        t1[0] = idx + w + 1;

        t2[2] = idx;
        t2[1] = idx + w + 1;
        t2[0] = idx + w;
    }
    
    else if (valid == 3)
    {
        // 3 valid points -> 1 triangle
        if (d[0] <= 0)
        {   
            t1[2] = idx + 1;
            t1[1] = idx + w + 1;
            t1[0] = idx + w;    
        }
        else if (d[1] <= 0)
        {
            t1[2] = idx;
            t1[1] = idx + w + 1;
            t1[0] = idx + w;

        }
        else if (d[2] <= 0)
        {
            t1[2] = idx;
            t1[1] = idx + 1;
            t1[0] = idx + w + 1;
        }
        else
        {
            t1[2] = idx;
            t1[1] = idx + 1;
            t1[0] = idx + w;
        }
    }

    // Add to triangles

    indices[6 * idx + 0] = t1[0];
    indices[6 * idx + 1] = t1[1];
    indices[6 * idx + 2] = t1[2];
    
    indices[6 * idx + 3] = t2[0];
    indices[6 * idx + 4] = t2[1];
    indices[6 * idx + 5] = t2[2];
}



void cu_backProject(float* data, float4* vertices, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(w, h, block);

    g_backProject <<< grid, block >>> (data, vertices, w, h, fx, fy, cx, cy);
    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());   
}

void cu_calculateNormals(float* data, float4* normals, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(w, h, block);

    g_calculateNormals <<< grid, block >>> (data, normals, w, h, fx, fy, cx, cy);
    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());   
}

void cu_backProjectAndCalcNormals(float* data, float4* vertices, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(w, h, block);

    g_backProjectAndCalcNormals <<< grid, block >>> (data, vertices, w, h, fx, fy, cx, cy);
    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());   
}

void cu_generateVertexIndices(float* data, unsigned int* indices, const int w, const int h)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(w, h, block);

    g_generateVertexIndicesTriangles <<< grid, block >>> (data, indices, w, h);
    cudaCheckLastCall();
    cudaSafeCall(cudaDeviceSynchronize());   
}