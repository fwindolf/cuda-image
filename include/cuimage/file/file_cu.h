/**
 * @file   file_cu.h
 * @brief  Cuda functions needed for file operations
 * @author Florian Windolf
 */
#pragma once

#include "cuimage/cuda/type.h"

namespace cuimage
{
    /**
     * Upload a chunk of data from host to device
     */
    void* cu_Upload(const void* h_data, const size_t sizeBytes);

} // image
