CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)
PROJECT("cuimage")

# ------------------------------------------------------------------------
# Options


OPTION(BUILD_EXAMPLES   "Build examples"    OFF)
OPTION(BUILD_TESTS      "Build tests"       OFF)
OPTION(BUILD_SHARED     "Build shared lib"  OFF)

MESSAGE("Building CuImage with tests=${BUILD_TESTS}, examples=${BUILD_EXAMPLES}, shared=${BUILD_SHARED}")

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

# Pangolin
IF(PANGOLIN_DIR)
    MESSAGE("Pangolin DIR is set: Including ${PANGOLIN_DIR}/include and linking ${PANGOLIN_DIR}/lib")
    INCLUDE_DIRECTORIES(${PANGOLIN_DIR}/include)
    LINK_DIRECTORIES(${PANGOLIN_DIR}/lib)
ELSE()
    FIND_PACKAGE(Pangolin REQUIRED)
ENDIF()

# Eigen
IF(NOT EIGEN3_INCLUDE_DIR OR "${EIGEN3_INCLUDE_DIR}" STREQUAL "")
    SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
ENDIF()
IF(NOT EIGEN3_INCLUDE_DIR)
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

#TinyExr
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/tinyexr)

# OpenCV (w/o cuda support)
FIND_PACKAGE(OpenCV 3.4 REQUIRED core highgui imgproc)

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

# ------------------------------------------------------------------------
# Cuda flags

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ftemplate-backtrace-limit=0; --cudart;  static; --expt-relaxed-constexpr; -fmad=true; --expt-extended-lambda;--relocatable-device-code=false ;
-Xptxas -dlcm=ca; -Xptxas --opt-level=4 )


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O0 -g -G)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -g -G)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")


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
