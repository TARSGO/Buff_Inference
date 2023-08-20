cmake_minimum_required(VERSION 3.25)
project(wind_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH ~/git/openvino/build/)
find_package(OpenCV 4 REQUIRED)
find_package(OpenCV 4.5.0)
find_package(InferenceEngine REQUIRED)
# set(CMAKE_PREFIX_PATH /opt/intel/openvino_2021/deployment_tools/ngraph/cmake)
# find_package(ngraph)
find_package(Eigen3 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${InferenceEngine_INCLUDE_DIRS})

add_executable(wind_inference main.cpp detect/inference.cpp detect/inference.h)
target_link_libraries(wind_inference ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} ${Eigen3_LIBRARIES})
