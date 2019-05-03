SET(APP_EXAMPLE example)

SET(APP_DEPENDENCIES
    image
)

# ------------------------------------------------------------------------
# Example application

SET(MAIN "main.cpp")
FILE(GLOB INCLUDES "include/*.h" "include/**/*.h")
FILE(GLOB SOURCES  "src/*.cpp" "src/**/*.cpp" "src/*.cu" "src/**/*.cu")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

CUDA_ADD_EXECUTABLE(${APP_EXAMPLE}
    ${MAIN}
    ${CUDA_SOURCES}
    ${SOURCES}
    ${INCLUDE}
)

TARGET_LINK_LIBRARIES(${APP_EXAMPLE}
    ${APP_DEPENDENCIES}
)

TARGET_COMPILE_OPTIONS(${APP_EXAMPLE}
	PRIVATE -std=c++11
)