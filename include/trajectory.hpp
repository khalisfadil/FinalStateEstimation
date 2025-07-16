#pragma once

#include <slam.hpp> // Assumed to contain common SLAM type definitions
#include <problem/costterm/imusupercostterm.hpp> // Assumed for IMU factor graph integration

#include <point.hpp> // Definition for Point3D
#include <vector>
#include <Eigen/Dense>

namespace stateestimate {

    // Type alias for a vector of 4x4 matrices, using Eigen's aligned allocator for memory safety
    using ArrayMatrix4d = std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;
    using ArrayPoses = ArrayMatrix4d;

    /**
     * @brief A data container for a single segment of a sensor's trajectory.
     * @details This struct holds all essential information for a single time interval in a
     * LiDAR-inertial odometry or SLAM system. It includes the start and end poses,
     * raw sensor data (LiDAR, IMU), and estimated state variables (velocity, biases)
     * along with their uncertainties (covariance matrices).
     */
    struct TrajectoryFrame {
        
        /// @brief Default constructor.
        TrajectoryFrame() = default;

        /**
         * @brief Constructs a TrajectoryFrame with initial sensor data.
         * @param begin_time The start timestamp of the frame.
         * @param end_time The end timestamp of the frame.
         * @param points_vec The vector of LiDAR points captured during this frame.
         * @param imu_vec The vector of IMU measurements from this frame.
         */
        TrajectoryFrame(double begin_time, double end_time, std::vector<Point3D> points_vec, std::vector<finalicp::IMUData> imu_vec)
            : begin_timestamp(begin_time), end_timestamp(end_time), points(points_vec), imu_data_vec(imu_vec) {}

        // --- Pose Accessors and Interpolation ---

        /**
         * @brief Assembles and returns the 4x4 pose matrix at the beginning of the frame.
         * @return Eigen::Matrix4d The 4x4 transformation matrix (T_world_robot).
         */
        Eigen::Matrix4d getBeginPose() const {
            Eigen::Matrix4d begin_pose = Eigen::Matrix4d::Identity();
            begin_pose.block<3, 3>(0, 0) = begin_R;
            begin_pose.block<3, 1>(0, 3) = begin_t;
            return begin_pose;
        }

        /**
         * @brief Assembles and returns the 4x4 pose matrix at the end of the frame.
         * @return Eigen::Matrix4d The 4x4 transformation matrix (T_world_robot).
         */
        Eigen::Matrix4d getEndPose() const {
            Eigen::Matrix4d end_pose = Eigen::Matrix4d::Identity();
            end_pose.block<3, 3>(0, 0) = end_R;
            end_pose.block<3, 1>(0, 3) = end_t;
            return end_pose;
        }
        
        /**
         * @brief Returns the pose at the middle of the frame.
         * @details If an explicit mid-pose has been set (e.g., after an optimization step),
         * it returns that. Otherwise, it interpolates the pose at the midpoint time.
         * @return Eigen::Matrix4d The 4x4 transformation matrix.
         */
        Eigen::Matrix4d getMidPose() const {
            if (mid_pose_init) {
                return mid_pose_;
            } else {
                return getInterpolatedPose(getEvalTime());
            }
        }

        /**
         * @brief Interpolates the pose at any given timestamp within the frame's interval.
         * @param timestamp The absolute time for which to calculate the pose.
         * @return Eigen::Matrix4d The interpolated 4x4 transformation matrix.
         */
        Eigen::Matrix4d getInterpolatedPose(const double timestamp) const {
            // Ensure the time interval is valid to prevent division by zero
            if (std::abs(end_timestamp - begin_timestamp) < 1e-9) {
                return getBeginPose();
            }

            // Calculate interpolation factor 'alpha'
            const double alpha = (timestamp - begin_timestamp) / (end_timestamp - begin_timestamp);

            // Interpolate rotation using Spherical Linear Interpolation (SLERP)
            Eigen::Quaterniond q_begin(begin_R);
            Eigen::Quaterniond q_end(end_R);
            Eigen::Quaterniond q_interp = q_begin.slerp(alpha, q_end).normalized();

            // Interpolate translation using simple Linear Interpolation (LERP)
            Eigen::Vector3d t_interp = (1.0 - alpha) * begin_t + alpha * end_t;
            
            // Assemble the final pose matrix
            Eigen::Matrix4d interpolated_pose = Eigen::Matrix4d::Identity();
            interpolated_pose.block<3, 3>(0, 0) = q_interp.toRotationMatrix();
            interpolated_pose.block<3, 1>(0, 3) = t_interp;
            return interpolated_pose;
        }
        
        /**
         * @brief Sets an explicit evaluation time, otherwise defaults to the midpoint.
         * @param eval_timestamp The absolute timestamp for evaluation.
         */
        void setEvalTime(double eval_timestamp) {
            eval_timestamp_ = eval_timestamp;
            eval_time_init = true;
        }

        /**
         * @brief Gets the evaluation time for this frame.
         * @return The explicitly set evaluation time or the midpoint time if not set.
         */
        double getEvalTime() const {
            if (eval_time_init) {
                return eval_timestamp_;
            } else {
                return (begin_timestamp + end_timestamp) / 2.0;
            }
        }

        /**
         * @brief Sets an explicit pose for the midpoint of the frame.
         * @param mid_pose The 4x4 transformation matrix to set.
         */
        void setMidPose(const Eigen::Matrix4d& mid_pose) {
            mid_pose_ = mid_pose;
            mid_pose_init = true;
        }

        // --- Timestamps & Initialization Flags ---
        double begin_timestamp = 0.0;     ///< Start timestamp of the frame.
        double end_timestamp = 1.0;       ///< End timestamp of the frame.
        bool eval_time_init = false;      ///< Flag indicating if a custom evaluation time has been set.
        bool mid_pose_init = false;       ///< Flag indicating if an explicit mid-pose has been set.

        // --- Frame Boundary Poses ---
        Eigen::Matrix3d begin_R = Eigen::Matrix3d::Identity(); ///< Rotation at the beginning of the frame.
        Eigen::Vector3d begin_t = Eigen::Vector3d::Zero();     ///< Translation at the beginning of the frame.
        Eigen::Matrix3d end_R = Eigen::Matrix3d::Identity();   ///< Rotation at the end of the frame.
        Eigen::Vector3d end_t = Eigen::Vector3d::Zero();       ///< Translation at the end of the frame.

        // --- End-of-Frame Covariances (Uncertainty) ---
        Eigen::Matrix<double, 6, 6> end_T_rm_cov = Eigen::Matrix<double, 6, 6>::Identity();         ///< Covariance of the end pose.
        Eigen::Matrix<double, 6, 6> end_w_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity();     ///< Covariance of the end velocity.
        Eigen::Matrix<double, 6, 6> end_dw_mr_inr_cov = Eigen::Matrix<double, 6, 6>::Identity();    ///< Covariance of the end acceleration.
        Eigen::Matrix<double, 18, 18> end_state_cov = Eigen::Matrix<double, 18, 18>::Identity();    ///< Covariance of the full end-state [Pose, Vel, Bias].

        // --- Mid-Frame State Variables (Estimated at evaluation time) ---
        Eigen::Matrix<double, 6, 1> mid_w = Eigen::Matrix<double, 6, 1>::Zero();    ///< Spatial Velocity (linear and angular) at mid-frame.
        Eigen::Matrix<double, 6, 1> mid_dw = Eigen::Matrix<double, 6, 1>::Zero();   ///< Spatial Acceleration (linear and angular) at mid-frame.
        Eigen::Matrix<double, 6, 1> mid_b = Eigen::Matrix<double, 6, 1>::Zero();    ///< IMU biases (accelerometer and gyroscope) at mid-frame.
        Eigen::Matrix<double, 18, 18> mid_state_cov = Eigen::Matrix<double, 18, 18>::Identity();  ///< Covariance of the mid-frame state.
        Eigen::Matrix4d mid_T_mi = Eigen::Matrix4d::Identity();  ///< Transformation from IMU to Map frame at mid-frame.

        // --- Raw Sensor Data ---
        std::vector<Point3D> points;                           ///< LiDAR point cloud captured during this frame.
        std::vector<finalicp::IMUData> imu_data_vec;           ///< IMU measurements recorded during this frame.

    private:
        Eigen::Matrix4d mid_pose_ = Eigen::Matrix4d::Identity(); ///< Storage for an explicitly set mid-frame pose.
        double eval_timestamp_ = 0.5;                            ///< Storage for the custom evaluation timestamp.
    };

    /// @brief A trajectory is defined as a vector of TrajectoryFrames.
    using Trajectory = std::vector<TrajectoryFrame>;

}  // namespace stateestimate