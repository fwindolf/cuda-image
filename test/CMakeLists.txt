SET(APP_TESTS test)

SET(APP_DEPENDENCIES
    ${OpenCV_LIBS}
    cuimage
    gtest
)

# ------------------------------------------------------------------------
# Tests

SET(MAIN "main.cpp")
FILE(GLOB INCLUDES "*.h")
FILE(GLOB SOURCES  "test_*.cpp")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

ADD_EXECUTABLE(${APP_TESTS}
    ${MAIN}
    ${SOURCES}
    ${INCLUDE}
)

TARGET_LINK_LIBRARIES(${APP_TESTS}
    ${APP_DEPENDENCIES}
)

TARGET_COMPILE_OPTIONS(${APP_TESTS}
	PRIVATE -std=c++11
)