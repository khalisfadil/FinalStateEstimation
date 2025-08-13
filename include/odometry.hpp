#pragma once

#include <problem/costterm/imusupercostterm.hpp>
#include <dataframe.hpp>
#include <map.hpp>
#include <voxel.hpp>
#include <pose.hpp>
#include <trajectory.hpp>
#include <nlohmann/json.hpp> // Added for JSON parsing
#include <fstream> // Added for file operations
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

namespace stateestimate {

    class Odometry {
    public:
        using Ptr = std::shared_ptr<Odometry>;
        using ConstPtr = std::shared_ptr<const Odometry>;

        ArrayPoses T_i_r_gt_poses;
        
        struct Options {
            using Ptr = std::shared_ptr<Options>;
            using ConstPtr = std::shared_ptr<const Options>;

            virtual ~Options() = default;

            // frame preprocessing
            int init_num_frames = 20;  // The number of frames defining the initialization of the map
            double init_voxel_size = 0.2;
            double voxel_size = 0.5;
            double init_sample_voxel_size = 1.0;
            double sample_voxel_size = 1.5;

            // map
            double size_voxel_map = 1.0;       // Max Voxel: -32767 to 32767 then 32km map for SIZE_VOXEL_MAP = 1m
            double min_distance_points = 0.1;  // The minimal distance between points in the map
            int max_num_points_in_voxel = 20;  // The maximum number of points in a voxel
            double max_distance = 100.0;       // The threshold on the voxel size to remove points from the map
            int min_number_neighbors = 20;     // The minimum number of neighbors to be considered in the map
            int max_number_neighbors = 20;
            int voxel_lifetime = 10;

            // common icp options
            int num_iters_icp = 10;                      // The Maximum number of ICP iterations performed
            double threshold_orientation_norm = 0.0001;  // Threshold on rotation (deg) for ICP's stopping criterion
            double threshold_translation_norm = 0.001;   // Threshold on translation (m) for ICP's stopping criterion
            int min_number_keypoints = 100;

            // parallel processing
            int sequential_threshold_odom = 1000;
            unsigned int num_threads_odom = 4;

            bool debug_print = false;  // Whether to output debug information to std::cout
            std::string debug_path = "../report/traj/";
        };

        // Factory method for JSON-based initialization
        static Odometry::Ptr Get(const std::string& odometry, const std::string& json_path) {
            auto it = name2JsonCtor().find(odometry);
            if (it == name2JsonCtor().end()) {
                throw std::runtime_error("Unknown odometry type: " + odometry);
            }
            return it->second(json_path);
        }

        Odometry(const Options& options) : options_(options) { 
            map_.initialize(options_.voxel_lifetime, options_.sequential_threshold_odom, options_.num_threads_odom); }

        virtual ~Odometry() = default;

        // trajectory
        virtual Trajectory trajectory() = 0;

        // map
        size_t size() const { return map_.size(); }
        ArrayVector3d map() const { return map_.pointcloud(); }

        // The Output of a registration, including metrics
        struct RegistrationSummary {
            std::vector<Point3D> keypoints;                      // Last Keypoints selected
            std::vector<Point3D> corrected_points;               // Sampled points expressed in the initial frame
            std::vector<Point3D> all_corrected_points;           // Initial points expressed in the initial frame
            Eigen::Matrix3d Rs2m = Eigen::Matrix3d::Identity();  // The rotation between the initial frame and the new frame
            Eigen::Vector3d ts2m = Eigen::Vector3d::Zero();      // The translation between the initial frame and the new frame
            bool success = true;                                 // Whether the registration was a success
        };

        // Registers a new Frame to the Map with an initial estimate
        virtual RegistrationSummary registerFrame(const DataFrame& frame) = 0;

        // It's virtual, but not pure virtual (= 0), so derived classes are not
        // required to implement it. The empty {} means it does nothing by default.
        virtual void initT(const Eigen::Matrix4d& T) = 0;
        virtual void saveResults(const std::string& timestamp) = 0;

    protected:
        Trajectory trajectory_; // std::vector<TrajectoryFrame>; in trajectory.hpp
        Map map_;

    private:
        const Options options_;

        using JsonCtorFunc = std::function<Ptr(const std::string&)>;
        using Name2JsonCtor = std::unordered_map<std::string, JsonCtorFunc>;

        static Name2JsonCtor& name2JsonCtor() {
            static Name2JsonCtor name2jsonctor;
            return name2jsonctor;
        }

        template <typename T>
        friend class OdometryRegister;
    };

    template <typename T>
    struct OdometryRegister {
        OdometryRegister() {
            bool success = Odometry::name2JsonCtor().try_emplace(
                T::odometry_name_,
                Odometry::JsonCtorFunc([](const std::string& json_path) {
                    return std::make_shared<T>(json_path);
                })
            ).second;
            if (!success) {
                throw std::runtime_error{"OdometryRegister failed - duplicated name for JSON constructor"};
            }
        }
    };

    #define SLAM_REGISTER_ODOMETRY(NAME, TYPE)                      \
    public:                                                         \
        inline static constexpr auto odometry_name_ = NAME;         \
    private:                                                        \
        inline static stateestimate::OdometryRegister<TYPE> odometry_reg_;

} // namespace stateestimate

#include <odometry/lidarinertialodometry.hpp> // without this, the chef never shows up to work
