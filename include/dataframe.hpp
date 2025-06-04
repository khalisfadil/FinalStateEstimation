#pragma once

#include <Eigen/Dense>

#include <problem/costterm/imusupercostterm.hpp>
#include <point.hpp>
#include <pose.hpp>

namespace  stateestimate{

    struct DataFrame {
        double timestamp;
        std::vector<Point3D> pointcloud;
        std::vector<finalicp::IMUData> imu_data_vec;
        std::vector<PoseData> pose_data_vec;
    };

}  // namespace stateestimate