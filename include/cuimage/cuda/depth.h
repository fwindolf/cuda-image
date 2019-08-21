#include "arithmetic.h"
#include "devptr.h"
#include "type.h"

namespace cuimage
{
inline __device__ bool d_is_valid_depth(const float d)
{
    return (isvalid(d) && d > 0.f);
}

inline __device__ float d_diff(const float d, const float dn)
{
    if (d_is_valid_depth(d) && d_is_valid_depth(dn))
        return d - dn;

    return 0.f;
}

inline __device__ float2 d_gradient_forward(
    const DevPtr<float>& depth, const int px, const int py)
{
    const float d = depth(px, py);

    const int width = depth.width;
    const int height = depth.height;

    float2 grad = make_float2(0, 0);

    if (px < width - 1)
        grad.x = d_diff(depth(px + 1, py), d);

    if (py < height - 1)
        grad.y = d_diff(depth(px, py + 1), d);

    return grad;
}

inline __device__ float2 d_gradient_backward(
    const DevPtr<float>& depth, const int px, const int py)
{
    const float d = depth(px, py);

    float2 grad = make_float2(0, 0);

    if (px > 0)
        grad.x = d_diff(d, depth(px - 1, py));

    if (py > 0)
        grad.y = d_diff(d, depth(px, py - 1));

    return grad;
}

inline __device__ float2 d_gradient_symmetric(
    const DevPtr<float>& depth, const int px, const int py)
{
    const float d = depth(px, py);

    const int width = depth.width;
    const int height = depth.height;

    float2 grad = make_float2(0, 0);

    if (px == 0)
        grad.x = d_diff(depth(px + 1, py), d);
    else if (px == width - 1)
        grad.x = d_diff(d, depth(px - 1, py));
    else
        grad.x = .5f * d_diff(depth(px + 1, py), depth(px - 1, py));

    if (py == 0)
        grad.y = d_diff(depth(px, py + 1), d);
    if (py == height - 1)
        grad.y = d_diff(d, depth(px, py - 1));
    else
        grad.y = .5f * d_diff(depth(px, py + 1), depth(px, py - 1));

    return grad;
}

inline __device__ float3 d_normals(const float d, const float dx,
    const float dy, const int px, const int py, const float fx, const float fy,
    const float cx, const float cy)
{
    if (!d_is_valid_depth(d))
        return make_float3(0.f, 0.f, 0.f);

    const float nx = fx * dx;
    const float ny = fy * dy;
    const float nz = -d - (px - cx) * dx - (py - cy) * dy;

    float3 n = make_float3(nx, ny, nz);
    float dz = max(1e-12, norm(n));
    return n / make<float3>(dz);
}

inline __device__ float4 d_normals(const float3 d, const int px, const int py,
    const float fx, const float fy, const float cx, const float cy)
{
    const float3 n = d_normals(d.x, d.y, d.z, px, py, fx, fy, cx, cy);
    return make_float4(n.x, n.y, n.z, 1.f);
}

inline __device__ float4 d_backproject(const float d, const int px,
    const int py, const float fx, const float fy, const float cx,
    const float cy)
{
    const float x = ((float)px - cx) / fx;
    const float y = ((float)py - cy) / fy;

    return make_float4(x * d, y * d, d, 1.f);
}

inline __device__ float3 d_project(const float4 v, const float fx,
    const float fy, const float cx, const float cy)
{
    const float px = fx * (v.x / v.z) + cx;
    const float py = fy * (v.y / v.z) + cy;
    return make_float3(px, py, 1.f);
}

} // cuimage