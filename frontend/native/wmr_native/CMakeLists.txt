cmake_minimum_required(VERSION 3.24)
project(wmr_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED NODE_INCLUDE_DIR)
  message(FATAL_ERROR "NODE_INCLUDE_DIR must point to the Node/Electron headers. The build must not use node-gyp.")
endif()

add_library(wmr_native SHARED src/wmr_native.cpp)
target_include_directories(wmr_native PRIVATE "${NODE_INCLUDE_DIR}")
set_target_properties(wmr_native PROPERTIES PREFIX "" SUFFIX ".node")

if(APPLE)
  set_target_properties(wmr_native PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()
