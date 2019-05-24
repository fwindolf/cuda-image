# Cuda Image

Cuda wrapper for images with basic functionality (math ops, reductions, conversion, visualization).
It comes without any dependency to OpenCV.

## Install

### Build All

This step will build cuda-image as library.
To build the examples, set the `BUILD_EXAMPLES` option.

Set CMAKE_INSTALL_PREFIX to `..` to install the libraries into the source folder. 
Additionally, you can further further customize by setting `BUILD_EXAMPLES`, `BUILD_TESTS` and `BUILD_SHARED`. 

```
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_BUILD_TYPE=Release
make -j8 install
```

## Usage

See the example application for how a standard case of how to use the library.
All image data manipulation is done via kernels (which themselfes are not very optimized most of the time though).

### Reading from file

You can pass the constructor of an image a path to either `.png` or `.exr` files, and it will automatically try to create
an image for the provided type. For best performance, the user should handle the conversion between types and load images
with the suitable type.

### Typing
Every image is strongly typed via the template type. All kernels are defined and insantiated for the 
common cuda types `float|int|uchar` with 0-4 components. 0- and 1- channel types can be treated similarily, and 
easily re-cast via the `reinterpretAs<type>()` functionality.
Each image has an associated width and height, the channel parameter is set on creation from file, but not otherwise used.

### Kernel usage
The `Image` class can be automatically casted into a `DevPtr` struct of the same type. It can be provieded to kernels and allows for data access via the `()` operator. Together with the utility functions from the `/cuda` headers, a cuda kernel can look like this:
```c++
__global__ void foo(DevPtr<float> data)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= output.width || pos.y >= output.height)
        return;
        
    float d = data(pos.x, pos.y);
    // ...
    data(pos.x, pos.y) = d;
}
```
Note that DevicePtrs can be passed by reference on host side, but on device a copy is neccessary (else there will be illegal memory access errors).

### Visualization

The image can be visualized in a named window using the `show(...)` functionality.
You can either create a typed window and update the content with every call of `show()`, or directly show an image by specifying the type.

Types are:
- COLOR_TYPE_<GREY, RGB, RGBA> for uchar images
- COLOR_TYPE_<GREY, RGB, RGBA>_F for float images
- DEPTH_TYPE for float images, showing the depth map as a 3D illuminated surface

Internally, every time a new window is created, it will also create a OpenGL texture. Show copies the data to the bound texture, the show call blocks for that duration.

To quickly debug images, you can also use the `visualize()` function. It provides only a minimal level of synchronization with the OpenGL thread, leaving the cuda context or bound texture broken some time. The call will also issue a warning informing you of that.

## Functionality

- [x] Reading from file
- [ ] Saving to file
- [x] Direct data access, up- and downloads
- [x] Usage in cuda kernels via DevPtr
- [x] Visualization (using Pangolin, with different types)
- [x] Tranformation
    - [x] Set Value
    - [x] Replace
    - [x] Threshold
    - [x] Absolute Value
- [x] Reductions 
    - [x] Min
    - [x] Max
    - [x] Mean
    - [x] Median (for 1-channel, pretty slow though)
    - [x] Norm1
    - [x] Norm2
    - [x] Valid pixels (not nan)
    - [x] NonZero pixels 
- [x] Color transformations (Gray <-> Color)
- [x] Casting (Reinterpreting, Copy to new type)
- [x] Resizing using only valid pixels (Linear, Linear with Mask)
- [x] Masking
- [x] Standard math operations with operator overloads (+, -, *, /)


