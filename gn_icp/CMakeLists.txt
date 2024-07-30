cmake_minimum_required(VERSION 3.5)

project(gn_icp)

set(CMAKE_BUILD_TYPE Release)
# 查找 Sophus 包
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
# 包含目录
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

# 添加可执行文件
add_executable(gn_icp gn_icp.cpp)

# 链接 Sophus 库
target_link_libraries(gn_icp Sophus::Sophus ${PCL_LIBRARIES})


