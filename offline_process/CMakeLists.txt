cmake_minimum_required(VERSION 2.8.3)
project(offline_process)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -rdynamic")


find_package(catkin REQUIRED COMPONENTS
    roscpp
    tf
    )

catkin_package()
include_directories(${catkin_INCLUDE_DIRS}) 

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

add_executable(offline_process
    src/offline_optimization_node.cpp
    src/factor.cpp
    src/bifactor.cpp
    src/initialize.cpp
    ${PROJECT_SOURCE_DIR}../../vins_estimator/src/utility/utility.cpp
    
    )

target_link_libraries(offline_process ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})