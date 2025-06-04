#pragma once

#include <Eigen/Dense>

namespace stateestimate {

    struct PoseData {
        double timestamp = 0;
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    };

}  // namespace stateestimate