SET(LIB_NAME cuimage)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

IF(NOT HAVE_OPENCV)
    INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/lodepng/include)
ENDIF()

# ------------------------------------------------------------------------
# cuimage library

SET(LIB_INCLUDE "${CMAKE_SOURCE_DIR}/include/${LIB_NAME}")

FILE(GLOB INCLUDES 
    "${LIB_INCLUDE}/*.h" 
    "${LIB_INCLUDE}/**/*.h"
)

FILE(GLOB SOURCES  
    "*.cpp"
    "**/*.cpp"
    "**/**/*.cpp"
)

FILE(GLOB CUDA_SOURCES
    "**/*.cu"
    "**/**/*.cu"
)

CUDA_ADD_LIBRARY(${LIB_NAME} STATIC 
    ${INCLUDES} 
    ${SOURCES}    
    ${CUDA_SOURCES}
)

SET(LIB_DEPENDENCIES
    lodepng
    pangolin
    ${OpenCV_LIBS}
    glut
    GLU
    GL
    GLEW
)

TARGET_LINK_LIBRARIES(${LIB_NAME}
    ${LIB_DEPENDENCIES}
)
