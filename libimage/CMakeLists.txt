SET(LIB_NAME image)

FILE(GLOB INCLUDES 
    "include/${LIB_NAME}/*.h" 
    "include/${LIB_NAME}/**/*.h"
)

FILE(GLOB SOURCES  
    "src/*.cpp"
    "src/**/*.cpp"
    "src/**/**/*.cpp"
)

FILE(GLOB CUDA_SOURCES
    "src/**/*.cu"
    "src/**/**/*.cu"
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

ADD_LIBRARY(${LIB_NAME} STATIC 
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

# ------------------------------------------------------------------------
# Tests

IF(BUILD_TESTS)
    SET(LIB_TEST ${LIB_NAME}_test)

    SET(TEST_MAIN "test/main.cpp")
    FILE(GLOB TEST_SOURCES "test/test_*.cpp")

    ADD_EXECUTABLE(${LIB_TEST}
        ${TEST_MAIN}
        ${TEST_SOURCES}
    )

    TARGET_LINK_LIBRARIES(${LIB_TEST}
        gtest gtest_main 
        pthread
        ${LIB_DEPENDENCIES}
    )

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

    ADD_TEST(${LIB_TEST} 
        test
    )

ENDIF()