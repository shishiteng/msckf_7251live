/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_UTILS_H
#define MSCKF_UTILS_H

#include <ros/ros.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <Eigen/Geometry>

namespace msckf {
/*
 * @brief utilities for msckf
 */
namespace utils {
Eigen::Isometry3d getTransformEigen(const ros::NodeHandle &nh,
                                    const std::string &field);

cv::Mat getTransformCV(const ros::NodeHandle &nh,
                       const std::string &field);

cv::Mat getVec16Transform(const ros::NodeHandle &nh,
                          const std::string &field);

cv::Mat getKalibrStyleTransform(const ros::NodeHandle &nh,
                                const std::string &field);
}
}
#endif
