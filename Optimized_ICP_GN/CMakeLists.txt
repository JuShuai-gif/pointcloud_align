cmake_minimum_required(VERSION 3.5)

project(Optimized_ICP_GN)

set(CMAKE_BUILD_TYPE Release)

include_directories(.)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

add_executable(Optimized_ICP_GN main.cpp optimized_icp_gn.cpp)

target_link_libraries(Optimized_ICP_GN ${PCL_LIBRARIES})

