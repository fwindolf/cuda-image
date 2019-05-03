#include "image/cuda.h"

void cu_backProject(float* data, float4* vertices, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy);

void cu_calculateNormals(float* data, float4* normals, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy);

void cu_backProjectAndCalcNormals(float* data, float4* vertices, const int w, const int h,
    const float fx, const float fy, const float cx, const float cy);


void cu_generateVertexIndices(float* data, unsigned int* indices, const int w, const int h);