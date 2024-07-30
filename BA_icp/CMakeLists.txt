cmake_minimum_required(VERSION 3.5)

project(ba_icp)

set(CMAKE_BUILD_TYPE Release)

include_directories(.)
# 查找 g2o 包
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
# 包含目录
include_directories(${g2o_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

add_executable(ba_icp ba_icp.cpp)

target_link_libraries(ba_icp ${g2o_LIBRARIES} ${PCL_LIBRARIES})

