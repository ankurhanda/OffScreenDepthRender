# -DCMAKE_BUILD_TYPE=Debug
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project( offscreenrender )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -std==gnu++11"-fext-numeric-literals")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -std=gnu++11"-fext-numeric-literals")


# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
ENDIF()

FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(Pangolin REQUIRED)
FIND_PACKAGE(CVD REQUIRED)

SET(TINYOBJLOADER_SOURCES
    ./src/utils/map_object_label2training_label.h
    ./src/tinyobjloader/tiny_obj_loader.h
    ./src/tinyobjloader/tiny_obj_loader.cc
    ./src/tinyobjloader/obj_writer.h
    ./src/tinyobjloader/obj_writer.cc
)

INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)

INCLUDE_DIRECTORIES(
  ${CVD_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIR}
)

LINK_DIRECTORIES(
  /usr/local/lib
  /usr/local/lib64
)

LINK_LIBRARIES(  
  ${CVD_LIBRARY}
  ${Pangolin_LIBRARY}
  -lGLEW
  -lglut
  -lGLU
  -lGL
  -lboost_system
  -lboost_filesystem
  -lboost_thread
)

IF(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

ADD_EXECUTABLE(render_rgb_depth
		${TINYOBJLOADER_SOURCES}
               ./src/main_opengl_render_rgb_depth.cpp)

