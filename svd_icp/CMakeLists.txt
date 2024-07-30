

project(svd_icp)

set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

add_executable(svd_icp svd_icp.cpp)

target_link_libraries(svd_icp ${PCL_LIBRARIES})

