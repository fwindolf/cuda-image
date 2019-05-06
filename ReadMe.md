# Cuda Image

Cuda wrapper for images with basic functionality (math ops, reductions, conversion, visualization)

## Install

### Init and build Pangolin
Pangolin is used to visualize the images.

```
git submodule init

cd third_party/Pangolin 
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. & make -j8
make install
cd ../../..
```

### Build All

This step will build the example application as well as the image static library.

```
mkdir build && cd build
cmake .. && make -j8
```

## Usage

See the example application for how a standard case of how to use the library.
All image data manipulation is done via kernels (which themselfes are not very optimized most of the time though).

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

The image can be visualized in a named window using the `show(...)` method. 
On the first call, an opengl texture will be created. The data will be copied to that texture, thus subsequent changes to the image data will not reflect in the shown image.

## Functionality

- [x] Reading from file
- [x] Casting to DevPtr
- [x] Visualization (using Pangolin, with different types)
- [x] Tranformation (Setting, Replacing, Thresholding)
- [x] Reductions (Min, Max, Mean, Norm1, Norm2)
- [x] Color transformations (Gray <-> Color)
- [x] Casting (Reinterpreting, Copy to new type)
- [x] Resizing using only valid pixels (Linear, Linear gwith Mask)
- [x] Masking
- [x] Simple math operations with operator overloads (+, -, *, /)
- [ ] Normalization


