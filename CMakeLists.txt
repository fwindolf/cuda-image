# Copyright (c) 2019 Florian Windolf
CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

PROJECT(CudaImage LANGUAGES CXX CUDA)

# executable and library output paths
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# ------------------------------------------------------------------------
# Build Tests
SET(BUILD_TESTS 1)

# ------------------------------------------------------------------------
# CMake configuration
IF(NOT CMAKE_BUILD_TYPE)
	# CMAKE_BUILD_TYPE: RelWithDebInfo, Release or Debug
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

# ------------------------------------------------------------------------
# Dependencies

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})

# OpenCV
FIND_PACKAGE(OpenCV 3.4 REQUIRED core highgui imgproc)

# Pangolin
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Pangolin/include)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Pangolin/lib)

# Eigen
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Eigen/include)

# LodePng
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/lodepng/include)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)
ADD_LIBRARY(lodepng STATIC ${PROJECT_SOURCE_DIR}/third_party/lodepng/src/lodepng.cpp)

# TinyExr
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/tinyexr)

# GTest
IF(BUILD_TESTS)
    FIND_PACKAGE(GTest)
    INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
ENDIF()

# ------------------------------------------------------------------------
# C++ flags

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

ADD_DEFINITIONS(
    -DSOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
)

# Cuda flags
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# ------------------------------------------------------------------------
# Modules

LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

ADD_SUBDIRECTORY("libimage")
INCLUDE_DIRECTORIES("libimage/include")

# ------------------------------------------------------------------------
# Application

ADD_SUBDIRECTORY("example")
INCLUDE_DIRECTORIES("example/include")

ADD_SUBDIRECTORY("tests")
INCLUDE_DIRECTORIES("tests/include")
