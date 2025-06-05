```cpp
#pragma once

#include <Eigen/Dense>

#include <problem/costterm/imusupercostterm.hpp>
#include <point.hpp>
#include <pose.hpp>

namespace stateestimate {

struct DataFrame {
  // Configurable default capacities for vector preallocation
  static constexpr size_t DEFAULT_POINTCLOUD_CAPACITY = 50000;  // Typical Lidar scan size
  static constexpr size_t DEFAULT_IMU_CAPACITY = 15;           // ~10-15 IMU samples per Lidar frame
  static constexpr size_t DEFAULT_POSE_CAPACITY = 5;           // Small, as often unused

  double timestamp;
  std::vector<Point3D> pointcloud;
  std::vector<finalicp::IMUData> imu_data_vec;
  std::vector<PoseData> pose_data_vec;

  // Default constructor reserves space for vectors
  DataFrame() {
    pointcloud.reserve(DEFAULT_POINTCLOUD_CAPACITY);
    imu_data_vec.reserve(DEFAULT_IMU_CAPACITY);
    pose_data_vec.reserve(DEFAULT_POSE_CAPACITY);
  }

  // Constructor with custom timestamp
  explicit DataFrame(double ts) : timestamp(ts) {
    pointcloud.reserve(DEFAULT_POINTCLOUD_CAPACITY);
    imu_data_vec.reserve(DEFAULT_IMU_CAPACITY);
    pose_data_vec.reserve(DEFAULT_POSE_CAPACITY);
  }

  // Constructor with custom capacities for flexibility
  DataFrame(double ts, size_t pointcloud_capacity, size_t imu_capacity, size_t pose_capacity)
      : timestamp(ts) {
    pointcloud.reserve(pointcloud_capacity);
    imu_data_vec.reserve(imu_capacity);
    pose_data_vec.reserve(pose_capacity);
  }
};

}  // namespace stateestimate
```