SET(LIB_NAME cuimage)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

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

IF(BUILD_SHARED)
    CUDA_ADD_LIBRARY(${LIB_NAME} SHARED 
        ${INCLUDES} 
        ${SOURCES}    
        ${CUDA_SOURCES}
    )       
ELSE()
    CUDA_ADD_LIBRARY(${LIB_NAME} STATIC 
        ${INCLUDES} 
        ${SOURCES}    
        ${CUDA_SOURCES}
    )
ENDIF()

SET(LIB_DEPENDENCIES
    pangolin
    glut
    GLU
    GL
    GLEW
)

TARGET_LINK_LIBRARIES(${LIB_NAME}
    ${LIB_DEPENDENCIES}
)

# ------------------------------------------------------------------------
# install target

SET(INSTALL_INCLUDE_DIR "include")

INSTALL(DIRECTORY ${LIB_INCLUDE}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}
)

INSTALL(TARGETS ${LIB_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
