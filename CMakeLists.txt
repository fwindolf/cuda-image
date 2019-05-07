CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)
PROJECT("cuimage")

# ------------------------------------------------------------------------
# Options


OPTION(BUILD_EXAMPLES   "Build examples"    ON)
OPTION(BUILD_TESTS      "Build tests"       ON)
OPTION(BUILD_SHARED     "Build shared lib"  OFF)
OPTION(HAVE_OPENCV      "Build with opencv" OFF)

MESSAGE("Building CuImage with tests=${BUILD_TESTS}, examples=${BUILD_EXAMPLES}, shared=${BUILD_SHARED}, with opencv=${HAVE_OPENCV}")

# ------------------------------------------------------------------------
# CMake configuration

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

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
IF(HAVE_OPENCV)   
    FIND_PACKAGE(OpenCV 3.4 REQUIRED core highgui imgproc)
ENDIF()

# Pangolin
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Pangolin/include)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Pangolin/lib)

# Eigen
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Eigen/include)

# LodePng
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/lodepng/include)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

IF (BUILD_SHARED)
    ADD_LIBRARY(lodepng SHARED 
        ${PROJECT_SOURCE_DIR}/third_party/lodepng/src/lodepng.cpp
    )
ELSE()
    ADD_LIBRARY(lodepng STATIC
        ${PROJECT_SOURCE_DIR}/third_party/lodepng/src/lodepng.cpp
    )
ENDIF()

# TinyExr
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/tinyexr)

# GTest
IF(BUILD_TESTS)
    FIND_PACKAGE(GTest)
    INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
ENDIF()

# ------------------------------------------------------------------------
# C++ flags

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

IF(BUILD_SHARED)
    # Add compile option -fPIC
    SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENDIF()

ADD_DEFINITIONS(
    -DSOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
)

IF(HAVE_OPENCV)
    ADD_DEFINITIONS(
        -DHAVE_OPENCV=\"1\"
    )
ENDIF()


# ------------------------------------------------------------------------
# Cuda flags

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ftemplate-backtrace-limit=0; --cudart;  static; -O4; --expt-relaxed-constexpr; -fmad=true; --expt-extended-lambda;--relocatable-device-code=false ;
-Xptxas -dlcm=ca; -Xptxas --opt-level=4 )


# ------------------------------------------------------------------------
# Uninstall target
# https://gist.github.com/royvandam/3033428

add_custom_target(uninstall
    "${CMAKE_COMMAND}" -P  "${CMAKE_SOURCE_DIR}/uninstall.cmake"
)


# ------------------------------------------------------------------------
# Library

ADD_SUBDIRECTORY("src")

# ------------------------------------------------------------------------
# Examples and tests

IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY("examples")
ENDIF()

IF(BUILD_TESTS)
    ADD_SUBDIRECTORY("test")
ENDIF()
