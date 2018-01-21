cmake_minimum_required(VERSION 2.8.12)
project(msckf)

add_compile_options(-std=c++11)

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  nav_msgs
  sensor_msgs
  geometry_msgs
  eigen_conversions
  tf_conversions
  random_numbers
  message_generation
  nodelet
  image_transport
  cv_bridge
  message_filters
  std_srvs
  camera_model
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)

##################
## ROS messages ##
##################
add_message_files(
  FILES
  FeatureMeasurement.msg
  CameraMeasurement.msg
  TrackingInfo.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
    eigen_conversions tf_conversions random_numbers message_runtime
    image_transport cv_bridge message_filters std_srvs
  DEPENDS Boost EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
)

# Msckf 
add_executable(msckf
  src/main.cpp
  src/msckf/image_processor.cpp
  src/msckf/msckf_vio.cpp
  src/msckf/utils.cpp
  src/visensor7251/device.cpp
  src/visensor7251/snTool.cpp
)
add_dependencies(msckf
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(msckf
  ${catkin_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  cyusb
)

