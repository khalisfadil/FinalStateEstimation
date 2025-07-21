#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <problem/costterm/imusupercostterm.hpp>
#include <problem/costterm/p2psupercostterm.hpp>
#include <solver/gausnewtonsolver.hpp>
#include <odometry.hpp>
#include <slam.hpp>
#include <evaluable/vspace/vspaceinterpolator.hpp>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>
// #include <glog/logging.h>
#include <common/stopwatch.hpp>

namespace stateestimate {

    class lidarinertialodom : public Odometry {
        public:
            using Matrix18d = Eigen::Matrix<double, 18, 18>;

            enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

            struct Neighborhood {
                Eigen::Vector3d center = Eigen::Vector3d::Zero();
                Eigen::Vector3d normal = Eigen::Vector3d::Zero();
                Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
                double a2D = 1.0;  // Planarity coefficient
            };

            struct Options : public Odometry::Options {
                // ----------------------------------------------------------------------------------
                // Sensor and Vehicle Configuration
                // ----------------------------------------------------------------------------------
                
                /// Fixed transformation from the robot's base frame to the sensor's frame (e.g., LiDAR). This is the extrinsic calibration.
                Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();

                // ----------------------------------------------------------------------------------
                // Continuous-Time Trajectory Model Parameters
                // ----------------------------------------------------------------------------------

                /// Diagonal elements of the continuous-time motion model's process noise covariance matrix ($Q_c$). Controls the uncertainty of motion.
                Eigen::Matrix<double, 6, 1> qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
                /// Diagonal elements of the maneuverability matrix ($A_d$) for the Singer motion model. Influences how quickly the acceleration can change.
                Eigen::Matrix<double, 6, 1> ad_diag = Eigen::Matrix<double, 6, 1>::Ones();
                /// Number of additional states (knots) to add between the start and end of a scan for the continuous-time trajectory.
                int num_extra_states = 0;

                // ----------------------------------------------------------------------------------
                // Point-to-Plane (P2P) ICP Parameters
                // ----------------------------------------------------------------------------------

                /// Exponent applied to the planarity score to weight point-to-plane correspondences. Higher values give more weight to very planar surfaces.
                double power_planarity = 2.0;
                /// Maximum distance for a point-to-plane correspondence to be considered a valid match.
                double p2p_max_dist = 0.5;
                /// Type of robust loss function (e.g., L2, Cauchy) to use for point-to-plane errors to reduce the impact of outliers.
                LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
                /// The scale parameter (sigma) for the chosen robust loss function for P2P errors.
                double p2p_loss_sigma = 0.1;

                // ----------------------------------------------------------------------------------
                // Radial Velocity Parameters (for Doppler LiDAR)
                // ----------------------------------------------------------------------------------

                /// Whether to use radial velocity measurements from the sensor (if available) as constraints in the optimization.
                bool use_rv = false;
                /// Whether to merge point-to-plane and radial velocity factors into a single, more efficient cost term.
                bool merge_p2p_rv = false;
                /// Maximum radial velocity error (in m/s) to be considered a valid measurement.
                double rv_max_error = 2.0;
                /// Type of robust loss function to use for radial velocity errors.
                LOSS_FUNC rv_loss_func = LOSS_FUNC::CAUCHY;
                /// Inverse covariance (information) value for radial velocity measurements.
                double rv_cov_inv = 0.1;
                /// The scale parameter (sigma) for the chosen robust loss function for radial velocity errors.
                double rv_loss_threshold = 0.05;

                // ----------------------------------------------------------------------------------
                // Optimization and Solver Parameters
                // ----------------------------------------------------------------------------------

                /// Enable/disable verbose logging from the optimization solver.
                bool verbose = false;
                /// Maximum number of iterations for the optimization solver in each step.
                int max_iterations = 5;
                /// Threshold to switch from parallel to sequential processing for small workloads to avoid overhead.
                int sequential_threshold = 500;
                /// Number of threads to use for parallelizable tasks.
                unsigned int num_threads = 4;
                /// Number of frames to wait before adding a frame's points to the map, allowing its pose to converge first.
                int delay_adding_points = 4;
                /// Whether to re-interpolate the entire trajectory using the final optimized state values for higher accuracy output.
                bool use_final_state_value = false;
                /// If true, the ICP loop can terminate early if the state change is below a threshold.
                bool break_icp_early = true;
                /// Whether to use a line search algorithm within the Gauss-Newton solver to find a better step size.
                bool use_line_search = false;

                // ----------------------------------------------------------------------------------
                // IMU Parameters
                // ----------------------------------------------------------------------------------

                /// Magnitude of the gravity vector. A positive value suggests a North-East-Down (NED) or similar z-down coordinate system.
                double gravity = 9.8042; 
                /// Whether to use IMU data to constrain the motion model.
                bool use_imu = true;
                /// Whether to use the accelerometer part of the IMU data (in addition to the gyroscope).
                bool use_accel = true;
                /// Diagonal elements of the measurement noise covariance for the accelerometer ($R_{acc}$).
                Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
                /// Diagonal elements of the measurement noise covariance for the gyroscope ($R_{gyro}$).
                Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
                /// Measurement noise for an external pose source (if available).
                Eigen::Matrix<double, 6, 1> r_pose = Eigen::Matrix<double, 6, 1>::Zero();
                /// Initial uncertainty (covariance, $P_0$) for the accelerometer bias.
                Eigen::Matrix<double, 3, 1> p0_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
                /// Prior covariance ($P_k$) on the accelerometer bias for frames after initialization.
                double pk_bias_accel = 0.0001;
                /// Process noise (covariance, $Q$) for the accelerometer bias random walk model (how much it can drift over time).
                Eigen::Matrix<double, 3, 1> q_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
                /// Initial uncertainty (covariance, $P_0$) for the gyroscope bias.
                double p0_bias_gyro = 0.0001;
                /// Prior covariance ($P_k$) on the gyroscope bias for frames after initialization.
                double pk_bias_gyro = 0.0001;
                /// Process noise (covariance, $Q$) for the gyroscope bias random walk model.
                double q_bias_gyro = 0.0001;
                /// Type of robust loss function for accelerometer errors.
                std::string acc_loss_func = "CAUCHY";
                /// Scale parameter (sigma) for the accelerometer robust loss function.
                double acc_loss_sigma = 1.0;
                /// Type of robust loss function for gyroscope errors.
                std::string gyro_loss_func = "L2";
                /// Scale parameter (sigma) for the gyroscope robust loss function.
                double gyro_loss_sigma = 1.0;

                // ----------------------------------------------------------------------------------
                // IMU-Map Transformation (T_mi) Parameters
                // ----------------------------------------------------------------------------------

                /// If true, the IMU-to-Map extrinsic ($T_{mi}$) is only optimized at the beginning and then held fixed.
                bool T_mi_init_only = true;
                /// If true, use a ground truth value for the IMU-to-Map extrinsic ($T_{mi}$) instead of estimating it.
                bool use_T_mi_gt = false;
                /// The ground truth IMU-to-Map extrinsic, represented as a 6D vector (translation + rotation).
                Eigen::Matrix<double, 6, 1> xi_ig = Eigen::Matrix<double, 6, 1>::Ones();
                /// Initial covariance for the $T_{mi}$ estimation.
                Eigen::Matrix<double, 6, 1> T_mi_init_cov = Eigen::Matrix<double, 6, 1>::Ones();
                /// Process noise for the IMU-to-Map extrinsic ($T_{mi}$) if it's continuously estimated.
                Eigen::Matrix<double, 6, 1> qg_diag = Eigen::Matrix<double, 6, 1>::Ones();
                /// Prior covariance on $T_{mi}$ for frames after initialization.
                Eigen::Matrix<double, 6, 1> T_mi_prior_cov = Eigen::Matrix<double, 6, 1>::Ones();
                /// Whether to apply the $T_{mi}$ prior after the initial frames.
                bool use_T_mi_prior_after_init = false;
                /// Whether to apply a prior on the IMU biases after the initial frames.
                bool use_bias_prior_after_init = false;

                // ----------------------------------------------------------------------------------
                // Initial State Priors (for the very first frame)
                // ----------------------------------------------------------------------------------
                
                /// Initial uncertainty (covariance, $P_0$) for the robot's pose.
                Eigen::Matrix<double, 6, 1> p0_pose = Eigen::Matrix<double, 6, 1>::Ones();
                /// Initial uncertainty (covariance, $P_0$) for the robot's velocity.
                Eigen::Matrix<double, 6, 1> p0_vel = Eigen::Matrix<double, 6, 1>::Ones();
                /// Initial uncertainty (covariance, $P_0$) for the robot's acceleration.
                Eigen::Matrix<double, 6, 1> p0_accel = Eigen::Matrix<double, 6, 1>::Ones();
                
                // ----------------------------------------------------------------------------------
                // Map Management
                // ----------------------------------------------------------------------------------

                /// Whether to remove voxels from the map after a certain number of frames have passed (their 'lifetime').
                bool filter_lifetimes = false;
            };

            static Options parse_json_options(const std::string& json_path);

            lidarinertialodom(const std::string& json_path);
            ~lidarinertialodom();

            Trajectory trajectory() override;
            RegistrationSummary registerFrame(const DataFrame& frame) override;

        private:
            inline double AngularDistance(const Eigen::Matrix3d& rota, const Eigen::Matrix3d& rotb);
            void sub_sample_frame(std::vector<Point3D>& frame, double size_voxel, int sequential_threshold);
            void sub_sample_frame_outlier_removal(std::vector<Point3D>& frame, double size_voxel,  int sequential_threshold);
            void grid_sampling(const std::vector<Point3D>& frame, std::vector<Point3D>& keypoints, double size_voxel_subsampling, int sequential_threshold);
            void initializeTimestamp(int index_frame, const DataFrame& const_frame);
            Neighborhood compute_neighborhood_distribution(const ArrayVector3d& points, int sequential_threshold);
            Eigen::Matrix<double, 6, 1> initialize_gravity(const std::vector<finalicp::IMUData>& imu_data_vec);
            void initializeMotion(int index_frame);
            std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D>& const_frame);
            void updateMap(int index_frame, int update_frame);
            bool icp(int index_frame, std::vector<Point3D>& keypoints, const std::vector<finalicp::IMUData>& imu_data_vec, const std::vector<PoseData>& pose_data_vec);
            void build_voxel_map(const std::vector<Point3D>& frame, double size_voxel, 
                                                tsl::robin_map<Voxel, Point3D, VoxelHash>& voxel_map, int sequential_threshold);

        private:
            Options options_;
            finalicp::se3::SE3StateVar::Ptr T_sr_var_ = nullptr;  // robot to sensor transformation as a slam variable

            struct TrajectoryVar {
                TrajectoryVar() = default;
                TrajectoryVar(const finalicp::traj::Time& t, const finalicp::se3::SE3StateVar::Ptr& T,
                            const finalicp::vspace::VSpaceStateVar<6>::Ptr& w, const finalicp::vspace::VSpaceStateVar<6>::Ptr& dw,
                            const finalicp::vspace::VSpaceStateVar<6>::Ptr& b, const finalicp::se3::SE3StateVar::Ptr& T_m_i)
                    : time(t), T_rm(T), w_mr_inr(w), dw_mr_inr(dw), imu_biases(b), T_mi(T_m_i) {}
                finalicp::traj::Time time;
                finalicp::se3::SE3StateVar::Ptr T_rm;
                finalicp::vspace::VSpaceStateVar<6>::Ptr w_mr_inr;
                finalicp::vspace::VSpaceStateVar<6>::Ptr dw_mr_inr;
                finalicp::vspace::VSpaceStateVar<6>::Ptr imu_biases;
                finalicp::se3::SE3StateVar::Ptr T_mi;
            };

            std::vector<TrajectoryVar> trajectory_vars_;
            size_t to_marginalize_ = 0;
            std::map<double, std::pair<Matrix18d, Matrix18d>> interp_mats_;
            finalicp::SlidingWindowFilter::Ptr sliding_window_filter_;

            SLAM_REGISTER_ODOMETRY("SLAM_LIDAR_INERTIAL_ODOM", lidarinertialodom);
    };

} // namespace stateestimate
