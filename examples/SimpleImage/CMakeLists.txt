# Simple Image Application
SET(APP "SimpleImage")

SET(MAIN "main.cpp")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

ADD_EXECUTABLE(${APP}
    ${MAIN}
)

TARGET_LINK_LIBRARIES(${APP}
    ${OpenCV_LIBS}
    cudart
    cuimage
)