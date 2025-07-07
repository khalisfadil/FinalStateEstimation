#pragma once

#include <nlohmann/json.hpp>

#include <fstream>
#include <set>

#include <slam.hpp>
#include <evaluable/vspace/vspaceinterpolator.hpp>
#include <problem/costterm/imusupercostterm.hpp>
#include <problem/costterm/p2psupercostterm.hpp>
#include <solver/gausnewtonsolver.hpp>
#include <odometry.hpp>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>

#include <glog/logging.h>

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

                // vehicle Pose
                Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();

                // trajectory
                Eigen::Matrix<double, 6, 1> qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
                Eigen::Matrix<double, 6, 1> ad_diag = Eigen::Matrix<double, 6, 1>::Ones();
                int num_extra_states = 0;

                // p2p
                double power_planarity = 2.0;
                double p2p_max_dist = 0.5;
                LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
                double p2p_loss_sigma = 0.1;

                // radial velocity
                bool use_rv = false;
                bool merge_p2p_rv = false;
                double rv_max_error = 2.0;
                LOSS_FUNC rv_loss_func = LOSS_FUNC::CAUCHY;
                double rv_cov_inv = 0.1;
                double rv_loss_threshold = 0.05;

                // optimization
                bool verbose = false;
                int max_iterations = 5;
                int sequential_threshold = 100;
                unsigned int num_threads = 4;

                int delay_adding_points = 4;
                bool use_final_state_value = false;

                // IMU
                double gravity = 9.8042; // NED frame coordinate. z down
                Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
                Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
                Eigen::Matrix<double, 6, 1> r_pose = Eigen::Matrix<double, 6, 1>::Zero();
                Eigen::Matrix<double, 3, 1> p0_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
                double pk_bias_accel = 0.0001;
                Eigen::Matrix<double, 3, 1> q_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
                double p0_bias_gyro = 0.0001;
                double pk_bias_gyro = 0.0001;
                double q_bias_gyro = 0.0001;
                bool use_imu = true;
                bool T_mi_init_only = true;
                bool use_T_mi_gt = false;

                // T_mi:
                Eigen::Matrix<double, 6, 1> qg_diag = Eigen::Matrix<double, 6, 1>::Ones();
                Eigen::Matrix<double, 6, 1> p0_pose = Eigen::Matrix<double, 6, 1>::Ones();
                Eigen::Matrix<double, 6, 1> p0_vel = Eigen::Matrix<double, 6, 1>::Ones();
                Eigen::Matrix<double, 6, 1> p0_accel = Eigen::Matrix<double, 6, 1>::Ones();
                Eigen::Matrix<double, 6, 1> T_mi_init_cov = Eigen::Matrix<double, 6, 1>::Ones();
                Eigen::Matrix<double, 6, 1> T_mi_prior_cov = Eigen::Matrix<double, 6, 1>::Ones();
                Eigen::Matrix<double, 6, 1> xi_ig = Eigen::Matrix<double, 6, 1>::Ones();
                bool use_T_mi_prior_after_init = false;
                bool use_bias_prior_after_init = false;
                std::string acc_loss_func = "CAUCHY";
                double acc_loss_sigma = 1.0;
                std::string gyro_loss_func = "L2";
                double gyro_loss_sigma = 1.0;

                bool filter_lifetimes = false;
                bool break_icp_early = true;
                bool use_line_search = false;
                bool use_accel = true;
            };

            lidarinertialodom(const std::string& json_path);

            ~lidarinertialodom();

            Options parse_metadata(const nlohmann::json& json_data);

            Trajectory trajectory() override;

            RegistrationSummary registerFrame(const DataFrame &frame) override;

        private:

            inline double AngularDistance(const Eigen::Matrix3d &rota, const Eigen::Matrix3d &rotb);

            void sub_sample_frame(std::vector<Point3D>& frame, double size_voxel, int num_threads, int sequential_threshold);

            void grid_sampling(const std::vector<Point3D>& frame, std::vector<Point3D>& keypoints, double size_voxel_subsampling, int num_threads, int sequential_threshold);

            void initializeTimestamp(int index_frame, const DataFrame &const_frame);

            Neighborhood compute_neighborhood_distribution(const ArrayVector3d& points, int num_threads, int sequential_threshold);

            Eigen::Matrix<double, 6, 1> initialize_gravity(const std::vector<finalicp::IMUData> &imu_data_vec);

            void initializeMotion(int index_frame);

            std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);

            void updateMap(int index_frame, int update_frame);

            bool icp(int index_frame, std::vector<Point3D> &keypoints, const std::vector<finalicp::IMUData> &imu_data_vec, const std::vector<PoseData> &pose_data_vec);

        private:

            Options options_;

            finalicp::se3::SE3StateVar::Ptr T_sr_var_ = nullptr;  // robot to sensor transformation as a slam variable

            struct TrajectoryVar {
                TrajectoryVar(const finalicp::traj::Time &t, const finalicp::se3::SE3StateVar::Ptr &T,
                            const finalicp::vspace::VSpaceStateVar<6>::Ptr &w, const finalicp::vspace::VSpaceStateVar<6>::Ptr &dw,
                            const finalicp::vspace::VSpaceStateVar<6>::Ptr &b, const finalicp::se3::SE3StateVar::Ptr &T_m_i)
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


} // namespace lidarinertialodom