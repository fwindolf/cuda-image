/**
 * @file   file_cu.h
 * @brief  Cuda functions needed for file operations
 * @author Florian Windolf
 */
#pragma once

#include "image/cuda/type.h"

namespace image
{
    /**
     * Upload a chunk of data from host to device
     */
    void* cu_Upload(const void* h_data, const size_t sizeBytes);

} // image
