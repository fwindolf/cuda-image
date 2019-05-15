/**
 * @file   types.h
 * @brief  Defines different types for visualization
 * @author Florian Windolf
 */
#pragma once

namespace cuimage
{

enum VisType
{
    NONE,
    DEPTH_TYPE,
    DEPTH_WITH_GRADIENTS_TYPE,
    NORMALS_TYPE,
    MASK_TYPE,
    COLOR_TYPE_GREY,
    COLOR_TYPE_GREY_F,
    COLOR_TYPE_RGB,
    COLOR_TYPE_RGB_F,
    COLOR_TYPE_RGBA,
    COLOR_TYPE_RGBA_F
};

} // image
