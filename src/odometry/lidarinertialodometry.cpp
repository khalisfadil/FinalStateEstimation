#include <odometry/lidarinertialodometry.hpp>

#include <iomanip>
#include <random>

#include <slam.hpp>

namespace  stateestimate{

    // ########################################################################
    // AngularDistance
    // ########################################################################

    inline double lidarinertialodom::AngularDistance(const Eigen::Matrix3d &rota, const Eigen::Matrix3d &rotb) {
        double d = 0.5 * ((rota * rotb.transpose()).trace() - 1);
        return std::acos(std::max(std::min(d, 1.0), -1.0)) * 180.0 / M_PI;
    }

    // ########################################################################
    // sub_sample_frame
    // ########################################################################
    void lidarinertialodom::sub_sample_frame(std::vector<Point3D>& frame, double size_voxel, int num_threads, int sequential_threshold) {

        // Define a voxel map using tsl::robin_map (a fast hash map) to store one point per voxel
        // Voxel: Key representing voxel coordinates
        // Point3D: Value representing the 3D point
        // VoxelHash: Hash function for voxel keys
        using VoxelMap = tsl::robin_map<Voxel, Point3D, VoxelHash>;

        // Check if the input frame size is below the sequential threshold
        if (frame.size() < static_cast<size_t>(sequential_threshold)) {
            // Step 1: Sequential processing for small inputs (faster for small datasets)
            // Create a voxel map to store one point per voxel
            VoxelMap voxel_map;
            // Reserve space to reduce reallocation (estimate: ~1/4 of input size)
            voxel_map.reserve(frame.size() / 4);

            // Step 2: Iterate through all points in the frame
            for (const auto& point : frame) {
                // Compute the voxel coordinates for the current point based on size_voxel
                Voxel voxel = Voxel::Coordinates(point.pt, size_voxel);
                // Insert the point into the voxel map if the voxel is not already occupied
                // try_emplace ensures only the first point for a voxel is stored
                voxel_map.try_emplace(voxel, point);
            }

            // Step 3: Rebuild the frame with downsampled points
            // Clear the input frame
            frame.clear();
            // Reserve space for the downsampled points (size of voxel map)
            frame.reserve(voxel_map.size());
            // Copy the points from the voxel map to the frame
            for (const auto& pair : voxel_map) {
                frame.push_back(pair.second); // pair.second is the Point3D
            }
            // Shrink the frame to fit the actual size (optimize memory)
            frame.shrink_to_fit();
            // Return after sequential processing
            return;
        }

        // Step 4: Parallel processing for large inputs (frame size >= sequential_threshold)
        // Define a structure to hold a voxel and its corresponding point
        struct VoxelPoint {
            Voxel voxel;    // Voxel coordinates
            Point3D point;  // Corresponding 3D point
        };

        // Step 5: Set up TBB thread limit for parallel processing
        // Limit the number of threads to num_threads
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

        // Step 6: Create a concurrent vector to store voxel-point pairs
        // tbb::concurrent_vector is thread-safe for parallel insertions
        tbb::concurrent_vector<VoxelPoint> voxel_points;
        // Reserve space to reduce reallocation (estimate: ~1/2 of input size)
        voxel_points.reserve(frame.size() / 2);

        // Step 7: Parallel computation of voxel coordinates
        // Use TBB's parallel_for to process the frame in parallel
        // blocked_range divides the range [0, frame.size()) into chunks
        // sequential_threshold is used as the grain size for load balancing
        tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size(), sequential_threshold),
            [&](const tbb::blocked_range<size_t>& range) {
                // Iterate over the assigned range of indices
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    // Get the 3D coordinates of the current point
                    const auto& pt = frame[i].pt;
                    // Compute the voxel coordinates for the point
                    Voxel voxel = Voxel::Coordinates(pt, size_voxel);
                    // Store the voxel and point in the concurrent vector
                    voxel_points.push_back({voxel, frame[i]});
                }
            });

        // Step 8: Sequentially build the voxel map from the voxel-point pairs
        // Create a voxel map to store one point per voxel
        VoxelMap voxel_map;
        // Reserve space based on the size of the concurrent vector
        voxel_map.reserve(voxel_points.size());
        // Iterate through all voxel-point pairs
        for (const auto& vp : voxel_points) {
            // Insert the point into the voxel map if the voxel is not already occupied
            voxel_map.try_emplace(vp.voxel, vp.point);
        }

        // Step 9: Rebuild the frame with downsampled points
        // Clear the input frame
        frame.clear();
        // Reserve space for the downsampled points
        frame.reserve(voxel_map.size());
        // Copy the points from the voxel map to the frame
        for (const auto& pair : voxel_map) {
            frame.push_back(pair.second); // pair.second is the Point3D
        }
        // Shrink the frame to fit the actual size (optimize memory)
        frame.shrink_to_fit();
    }

    // ########################################################################
    // grid_sampling
    // ########################################################################

    void lidarinertialodom::grid_sampling(const std::vector<Point3D>& frame, std::vector<Point3D>& keypoints, 
                                     double size_voxel_subsampling, int num_threads, int sequential_threshold) {

        // Step 1: Clear the output keypoints vector to ensure it starts empty
        // This prevents appending new points to any existing data
        keypoints.clear();

        // Step 2: Create a temporary vector to hold a copy of the input frame
        // frame_sub is used to avoid modifying the input frame (which is const)
        std::vector<Point3D> frame_sub;

        // Step 3: Resize frame_sub to match the size of the input frame
        // This pre-allocates memory for efficiency
        frame_sub.resize(frame.size());

        // Step 4: Copy all points from the input frame to frame_sub
        // A simple loop is used to perform the copy
        for (int i = 0; i < (int)frame_sub.size(); i++) {
            frame_sub[i] = frame[i];
        }

        // Step 5: Apply voxel grid subsampling to frame_sub
        // Calls sub_sample_frame to reduce the number of points by keeping one point per voxel
        // Modifies frame_sub in-place, using the provided voxel size, thread count, and threshold
        sub_sample_frame(frame_sub, size_voxel_subsampling, num_threads, sequential_threshold);

        // Step 6: Reserve space in keypoints to avoid reallocations
        // Uses the size of the downsampled frame_sub to estimate the required capacity
        keypoints.reserve(frame_sub.size());

        // Step 7: Copy the downsampled points from frame_sub to keypoints
        // Uses std::transform with a lambda to copy each point (identity transformation)
        // std::back_inserter appends points to keypoints
        std::transform(frame_sub.begin(), frame_sub.end(), std::back_inserter(keypoints), 
                    [](const auto c) { return c; });

        // Step 8: Optimize memory usage of keypoints
        // shrink_to_fit reduces the capacity to match the actual size of keypoints
        keypoints.shrink_to_fit();
    }

    // ########################################################################
    // compute_neighborhood_distribution
    // ########################################################################

    lidarinertialodom::Neighborhood lidarinertialodom::compute_neighborhood_distribution(const ArrayVector3d& points, int num_threads, int sequential_threshold) {

        Neighborhood neighborhood;

        // Handle empty or single-point cases
        if (points.empty()) {
            return neighborhood; // Default: zero center/normal, identity covariance, a2D=1.0
        }
        if (points.size() == 1) {
            neighborhood.center = points[0];
            neighborhood.covariance = Eigen::Matrix3d::Zero();
            neighborhood.normal = Eigen::Vector3d::UnitZ(); // Arbitrary default
            neighborhood.a2D = 0.0; // Non-planar for single point
            return neighborhood;
        }

        // Set TBB thread limit once for the entire function
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

        // Single-pass computation for barycenter and covariance
        Eigen::Vector3d barycenter = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        const size_t point_count = points.size();

        if (point_count < static_cast<size_t>(sequential_threshold)) {
            // Sequential for small inputs
            for (const auto& point : points) {
                barycenter += point;
                covariance += point * point.transpose(); // Accumulate outer product
            }
        } else {
            // Parallel for large inputs
            struct Accumulator {
                Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                Eigen::Matrix3d outer_sum = Eigen::Matrix3d::Zero();
            };
            Accumulator result = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, point_count, sequential_threshold),
                Accumulator(),
                [&](const tbb::blocked_range<size_t>& range, Accumulator acc) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        acc.sum += points[i];
                        acc.outer_sum += points[i] * points[i].transpose();
                    }
                    return acc;
                },
                [](const Accumulator& a, const Accumulator& b) {
                    Accumulator result;
                    result.sum = a.sum + b.sum;
                    result.outer_sum = a.outer_sum + b.outer_sum;
                    return result;
                }
            );
            barycenter = result.sum;
            covariance = result.outer_sum;
        }

        // Normalize barycenter and compute covariance
        barycenter /= static_cast<double>(point_count);
        covariance /= static_cast<double>(point_count);
        covariance -= barycenter * barycenter.transpose(); // Finalize covariance

        // Verify covariance symmetry (debug mode only)
        // assert((covariance - covariance.transpose()).norm() < 1e-10 && "Covariance matrix is not symmetric");

        neighborhood.center = barycenter;
        neighborhood.covariance = covariance;

        // Compute eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance);

        // Normal is eigenvector corresponding to smallest eigenvalue
        neighborhood.normal = es.eigenvectors().col(0).normalized();

        // Compute planarity coefficient (a2D): measures planarity (0 for planar, ~1 for isotropic)
        double sigma_1 = std::sqrt(std::max(0.0, es.eigenvalues()[2])); // Largest
        double sigma_2 = std::sqrt(std::max(0.0, es.eigenvalues()[1])); // Middle
        double sigma_3 = std::sqrt(std::max(0.0, es.eigenvalues()[0])); // Smallest
        constexpr double epsilon = 1e-6; // Avoid division by near-zero
        neighborhood.a2D = sigma_1 > epsilon ? (sigma_2 - sigma_3) / sigma_1 : 0.0; // planatery coefficient

        // Check for NaN
        // if (!std::isfinite(neighborhood.a2D)) {
        //     throw std::runtime_error("Planarity coefficient is NaN");
        // }
        if (neighborhood.a2D != neighborhood.a2D) {
            // LOG(ERROR) << "FOUND NAN!!!";
            throw std::runtime_error("error");
        }

        return neighborhood;
    }

    // ########################################################################
    // parse_json_options
    // ########################################################################

    // Correctly define parse_json_options as a static member function
    lidarinertialodom::Options lidarinertialodom::parse_json_options(const std::string& json_path) {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            // LOG(ERROR) << "Failed to open JSON file: " << json_path;
            throw std::runtime_error("Failed to open JSON file: " + json_path);
        }

        nlohmann::json json_data;
        try {
            file >> json_data;
        } catch (const nlohmann::json::parse_error& e) {
            // LOG(ERROR) << "JSON parse error in " << json_path << ": " << e.what();
            throw std::runtime_error("JSON parse error in " + json_path + ": " + e.what());
        }

        lidarinertialodom::Options parsed_options;

        if (!json_data.is_object()) {
            // LOG(ERROR) << "JSON data must be an object";
            throw std::runtime_error("JSON data must be an object");
        }

        try {
            // Parse odometry_options object
            if (!json_data.contains("odometry_options") || !json_data["odometry_options"].is_object()) {
                // LOG(ERROR) << "Missing or invalid 'odometry_options' object";
                throw std::runtime_error("Missing or invalid 'odometry_options' object");
            }
            const auto& odometry_options = json_data["odometry_options"];
            
            // Base Odometry::Options
            if (odometry_options.contains("init_num_frames")) parsed_options.init_num_frames = odometry_options["init_num_frames"].get<int>();
            if (odometry_options.contains("init_voxel_size")) parsed_options.init_voxel_size = odometry_options["init_voxel_size"].get<double>();
            if (odometry_options.contains("voxel_size")) parsed_options.voxel_size = odometry_options["voxel_size"].get<double>();
            if (odometry_options.contains("init_sample_voxel_size")) parsed_options.init_sample_voxel_size = odometry_options["init_sample_voxel_size"].get<double>();
            if (odometry_options.contains("sample_voxel_size")) parsed_options.sample_voxel_size = odometry_options["sample_voxel_size"].get<double>();
            if (odometry_options.contains("size_voxel_map")) parsed_options.size_voxel_map = odometry_options["size_voxel_map"].get<double>();
            if (odometry_options.contains("min_distance_points")) parsed_options.min_distance_points = odometry_options["min_distance_points"].get<double>();
            if (odometry_options.contains("max_num_points_in_voxel")) parsed_options.max_num_points_in_voxel = odometry_options["max_num_points_in_voxel"].get<int>();
            if (odometry_options.contains("max_distance")) parsed_options.max_distance = odometry_options["max_distance"].get<double>();
            if (odometry_options.contains("min_number_neighbors")) parsed_options.min_number_neighbors = odometry_options["min_number_neighbors"].get<int>();
            if (odometry_options.contains("max_number_neighbors")) parsed_options.max_number_neighbors = odometry_options["max_number_neighbors"].get<int>();
            if (odometry_options.contains("voxel_lifetime")) parsed_options.voxel_lifetime = odometry_options["voxel_lifetime"].get<int>();
            if (odometry_options.contains("num_iters_icp")) parsed_options.num_iters_icp = odometry_options["num_iters_icp"].get<int>();
            if (odometry_options.contains("threshold_orientation_norm")) parsed_options.threshold_orientation_norm = odometry_options["threshold_orientation_norm"].get<double>();
            if (odometry_options.contains("threshold_translation_norm")) parsed_options.threshold_translation_norm = odometry_options["threshold_translation_norm"].get<double>();
            if (odometry_options.contains("min_number_keypoints")) parsed_options.min_number_keypoints = odometry_options["min_number_keypoints"].get<int>();
            if (odometry_options.contains("debug_print")) parsed_options.debug_print = odometry_options["debug_print"].get<bool>();
            if (odometry_options.contains("debug_path")) parsed_options.debug_path = odometry_options["debug_path"].get<std::string>();

            // lidarinertialodom::Options
            if (odometry_options.contains("T_sr") && odometry_options["T_sr"].is_array() && odometry_options["T_sr"].size() == 16) {
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        parsed_options.T_sr(i, j) = odometry_options["T_sr"][i * 4 + j].get<double>();
                    }
                }
            }
            if (odometry_options.contains("qc_diag") && odometry_options["qc_diag"].is_array() && odometry_options["qc_diag"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.qc_diag(i) = odometry_options["qc_diag"][i].get<double>();
                }
            }
            if (odometry_options.contains("ad_diag") && odometry_options["ad_diag"].is_array() && odometry_options["ad_diag"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.ad_diag(i) = odometry_options["ad_diag"][i].get<double>();
                }
            }
            if (odometry_options.contains("num_extra_states")) parsed_options.num_extra_states = odometry_options["num_extra_states"].get<int>();
            if (odometry_options.contains("power_planarity")) parsed_options.power_planarity = odometry_options["power_planarity"].get<double>();
            if (odometry_options.contains("p2p_max_dist")) parsed_options.p2p_max_dist = odometry_options["p2p_max_dist"].get<double>();
            if (odometry_options.contains("p2p_loss_func")) {
                std::string loss_func = odometry_options["p2p_loss_func"].get<std::string>();
                if (loss_func == "L2") parsed_options.p2p_loss_func = lidarinertialodom::LOSS_FUNC::L2;
                else if (loss_func == "DCS") parsed_options.p2p_loss_func = lidarinertialodom::LOSS_FUNC::DCS;
                else if (loss_func == "CAUCHY") parsed_options.p2p_loss_func = lidarinertialodom::LOSS_FUNC::CAUCHY;
                else if (loss_func == "GM") parsed_options.p2p_loss_func = lidarinertialodom::LOSS_FUNC::GM;
                else {
                    // LOG(ERROR) << "Invalid p2p_loss_func: " << loss_func;
                    throw std::runtime_error("Invalid p2p_loss_func: " + loss_func);
                }
            }
            if (odometry_options.contains("p2p_loss_sigma")) parsed_options.p2p_loss_sigma = odometry_options["p2p_loss_sigma"].get<double>();
            if (odometry_options.contains("use_rv")) parsed_options.use_rv = odometry_options["use_rv"].get<bool>();
            if (odometry_options.contains("merge_p2p_rv")) parsed_options.merge_p2p_rv = odometry_options["merge_p2p_rv"].get<bool>();
            if (odometry_options.contains("rv_max_error")) parsed_options.rv_max_error = odometry_options["rv_max_error"].get<double>();
            if (odometry_options.contains("rv_loss_func")) {
                std::string loss_func = odometry_options["rv_loss_func"].get<std::string>();
                if (loss_func == "L2") parsed_options.rv_loss_func = lidarinertialodom::LOSS_FUNC::L2;
                else if (loss_func == "DCS") parsed_options.rv_loss_func = lidarinertialodom::LOSS_FUNC::DCS;
                else if (loss_func == "CAUCHY") parsed_options.rv_loss_func = lidarinertialodom::LOSS_FUNC::CAUCHY;
                else if (loss_func == "GM") parsed_options.rv_loss_func = lidarinertialodom::LOSS_FUNC::GM;
                else {
                    // LOG(ERROR) << "Invalid rv_loss_func: " << loss_func;
                    throw std::runtime_error("Invalid rv_loss_func: " + loss_func);
                }
            }
            if (odometry_options.contains("rv_cov_inv")) parsed_options.rv_cov_inv = odometry_options["rv_cov_inv"].get<double>();
            if (odometry_options.contains("rv_loss_threshold")) parsed_options.rv_loss_threshold = odometry_options["rv_loss_threshold"].get<double>();
            if (odometry_options.contains("verbose")) parsed_options.verbose = odometry_options["verbose"].get<bool>();
            if (odometry_options.contains("max_iterations")) parsed_options.max_iterations = odometry_options["max_iterations"].get<int>();
            if (odometry_options.contains("sequential_threshold")) parsed_options.sequential_threshold = odometry_options["sequential_threshold"].get<int>();
            if (odometry_options.contains("num_threads")) parsed_options.num_threads = odometry_options["num_threads"].get<unsigned int>();
            if (odometry_options.contains("delay_adding_points")) parsed_options.delay_adding_points = odometry_options["delay_adding_points"].get<int>();
            if (odometry_options.contains("use_final_state_value")) parsed_options.use_final_state_value = odometry_options["use_final_state_value"].get<bool>();
            if (odometry_options.contains("gravity")) parsed_options.gravity = odometry_options["gravity"].get<double>();
            if (odometry_options.contains("r_imu_acc") && odometry_options["r_imu_acc"].is_array() && odometry_options["r_imu_acc"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    parsed_options.r_imu_acc(i) = odometry_options["r_imu_acc"][i].get<double>();
                }
            }
            if (odometry_options.contains("r_imu_ang") && odometry_options["r_imu_ang"].is_array() && odometry_options["r_imu_ang"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    parsed_options.r_imu_ang(i) = odometry_options["r_imu_ang"][i].get<double>();
                }
            }
            if (odometry_options.contains("r_pose") && odometry_options["r_pose"].is_array() && odometry_options["r_pose"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.r_pose(i) = odometry_options["r_pose"][i].get<double>();
                }
            }
            if (odometry_options.contains("p0_bias_accel") && odometry_options["p0_bias_accel"].is_array() && odometry_options["p0_bias_accel"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    parsed_options.p0_bias_accel(i) = odometry_options["p0_bias_accel"][i].get<double>();
                }
            }
            if (odometry_options.contains("pk_bias_accel")) parsed_options.pk_bias_accel = odometry_options["pk_bias_accel"].get<double>();
            if (odometry_options.contains("q_bias_accel") && odometry_options["q_bias_accel"].is_array() && odometry_options["q_bias_accel"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    parsed_options.q_bias_accel(i) = odometry_options["q_bias_accel"][i].get<double>();
                }
            }
            if (odometry_options.contains("p0_bias_gyro")) parsed_options.p0_bias_gyro = odometry_options["p0_bias_gyro"].get<double>();
            if (odometry_options.contains("pk_bias_gyro")) parsed_options.pk_bias_gyro = odometry_options["pk_bias_gyro"].get<double>();
            if (odometry_options.contains("q_bias_gyro")) parsed_options.q_bias_gyro = odometry_options["q_bias_gyro"].get<double>();
            if (odometry_options.contains("use_imu")) parsed_options.use_imu = odometry_options["use_imu"].get<bool>();
            if (odometry_options.contains("T_mi_init_only")) parsed_options.T_mi_init_only = odometry_options["T_mi_init_only"].get<bool>();
            if (odometry_options.contains("use_T_mi_gt")) parsed_options.use_T_mi_gt = odometry_options["use_T_mi_gt"].get<bool>();
            if (odometry_options.contains("qg_diag") && odometry_options["qg_diag"].is_array() && odometry_options["qg_diag"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.qg_diag(i) = odometry_options["qg_diag"][i].get<double>();
                }
            }
            if (odometry_options.contains("p0_pose") && odometry_options["p0_pose"].is_array() && odometry_options["p0_pose"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.p0_pose(i) = odometry_options["p0_pose"][i].get<double>();
                }
            }
            if (odometry_options.contains("p0_vel") && odometry_options["p0_vel"].is_array() && odometry_options["p0_vel"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.p0_vel(i) = odometry_options["p0_vel"][i].get<double>();
                }
            }
            if (odometry_options.contains("p0_accel") && odometry_options["p0_accel"].is_array() && odometry_options["p0_accel"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.p0_accel(i) = odometry_options["p0_accel"][i].get<double>();
                }
            }
            if (odometry_options.contains("T_mi_init_cov") && odometry_options["T_mi_init_cov"].is_array() && odometry_options["T_mi_init_cov"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.T_mi_init_cov(i) = odometry_options["T_mi_init_cov"][i].get<double>();
                }
            }
            if (odometry_options.contains("T_mi_prior_cov") && odometry_options["T_mi_prior_cov"].is_array() && odometry_options["T_mi_prior_cov"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.T_mi_prior_cov(i) = odometry_options["T_mi_prior_cov"][i].get<double>();
                }
            }
            if (odometry_options.contains("xi_ig") && odometry_options["xi_ig"].is_array() && odometry_options["xi_ig"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.xi_ig(i) = odometry_options["xi_ig"][i].get<double>();
                }
            }
            if (odometry_options.contains("use_T_mi_prior_after_init")) parsed_options.use_T_mi_prior_after_init = odometry_options["use_T_mi_prior_after_init"].get<bool>();
            if (odometry_options.contains("use_bias_prior_after_init")) parsed_options.use_bias_prior_after_init = odometry_options["use_bias_prior_after_init"].get<bool>();
            if (odometry_options.contains("acc_loss_func")) parsed_options.acc_loss_func = odometry_options["acc_loss_func"].get<std::string>();
            if (odometry_options.contains("acc_loss_sigma")) parsed_options.acc_loss_sigma = odometry_options["acc_loss_sigma"].get<double>();
            if (odometry_options.contains("gyro_loss_func")) parsed_options.gyro_loss_func = odometry_options["gyro_loss_func"].get<std::string>();
            if (odometry_options.contains("gyro_loss_sigma")) parsed_options.gyro_loss_sigma = odometry_options["gyro_loss_sigma"].get<double>();
            if (odometry_options.contains("filter_lifetimes")) parsed_options.filter_lifetimes = odometry_options["filter_lifetimes"].get<bool>();
            if (odometry_options.contains("break_icp_early")) parsed_options.break_icp_early = odometry_options["break_icp_early"].get<bool>();
            if (odometry_options.contains("use_line_search")) parsed_options.use_line_search = odometry_options["use_line_search"].get<bool>();
            if (odometry_options.contains("use_accel")) parsed_options.use_accel = odometry_options["use_accel"].get<bool>();
        } catch (const nlohmann::json::exception& e) {
            // LOG(ERROR) << "JSON parsing error in metadata: " << e.what();
            throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));
        }
        return parsed_options;
    }

    // ########################################################################
    // lidarinertialodom constructor
    // ########################################################################

    lidarinertialodom::lidarinertialodom(const std::string& json_path)
        : Odometry(parse_json_options(json_path)), options_(parse_json_options(json_path)) {
        T_sr_var_ = finalicp::se3::SE3StateVar::MakeShared(math::se3::Transformation(options_.T_sr));
        T_sr_var_->locked() = true;
        sliding_window_filter_ = finalicp::SlidingWindowFilter::MakeShared(options_.num_threads);
    }

    // ########################################################################
    // ~lidarinertialodom deconstructor
    // ########################################################################

    lidarinertialodom::~lidarinertialodom() {
        using namespace finalicp::traj;

        // Skip if trajectory is empty
        if (trajectory_.empty() || trajectory_vars_.empty()) {
            return;
        }

        // Build filename with UTC timestamp
        auto now = std::chrono::system_clock::now();
        auto utc_time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&utc_time), "%Y%m%d_%H%M%S");
        std::string timestamp = ss.str();
        std::string filename = options_.debug_path + "/trajectory_" + timestamp + ".txt";

        // Open file with error handling
        std::ofstream trajectory_file(filename, std::ios::out);
        if (!trajectory_file.is_open()) {
            // LOG(ERROR) << "Failed to open trajectory file: " << filename;
            return; // Avoid further operations if file cannot be opened
        }
        // LOG(INFO) << "Building full trajectory." << std::endl;

        // Build full trajectory
        auto full_trajectory =  finalicp::traj::const_acc::Interface::MakeShared(options_.qc_diag);
        for (const auto& var : trajectory_vars_) {
            full_trajectory->add(var.time, var.T_rm, var.w_mr_inr, var.dw_mr_inr);
        }
        
        // LOG(INFO) << "Dumping trajectory." << std::endl;

        // Buffer output in stringstream
        std::stringstream buffer;
        buffer << std::fixed << std::setprecision(12);

        double begin_time = trajectory_.front().begin_timestamp;
        double end_time = trajectory_.back().end_timestamp;
        constexpr double dt = 0.01; // Hardcoded as in original

        for (double time = begin_time; time <= end_time; time += dt) {
            Time traj_time(time);
            const auto T_rm = full_trajectory->getPoseInterpolator(traj_time)->value().matrix();
            const auto w_mr_inr = full_trajectory->getVelocityInterpolator(traj_time)->value();

            buffer << 0.0 << " " << traj_time.nanosecs() << " "
                << T_rm(0, 0) << " " << T_rm(0, 1) << " " << T_rm(0, 2) << " " << T_rm(0, 3) << " "
                << T_rm(1, 0) << " " << T_rm(1, 1) << " " << T_rm(1, 2) << " " << T_rm(1, 3) << " "
                << T_rm(2, 0) << " " << T_rm(2, 1) << " " << T_rm(2, 2) << " " << T_rm(2, 3) << " "
                << T_rm(3, 0) << " " << T_rm(3, 1) << " " << T_rm(3, 2) << " " << T_rm(3, 3) << " "
                << w_mr_inr(0) << " " << w_mr_inr(1) << " " << w_mr_inr(2) << " "
                << w_mr_inr(3) << " " << w_mr_inr(4) << " " << w_mr_inr(5) << "\n";
        }

        // Write buffered output to file
        trajectory_file << buffer.str();
        trajectory_file.close();

        // LOG(INFO) << "Dumping trajectory. - DONE" << std::endl;
    }

    // ########################################################################
    // ~lidarinertialodom deconstructor
    // ########################################################################

    Trajectory lidarinertialodom::trajectory() {

        if (!options_.use_final_state_value) {
            return trajectory_;
        }

        // LOG(INFO) << "Building full trajectory." << std::endl;

        // Build full trajectory
        auto full_trajectory = finalicp::traj::const_acc::Interface::MakeShared(options_.qc_diag);
        for (const auto& var : trajectory_vars_) {
            full_trajectory->add(var.time, var.T_rm, var.w_mr_inr, var.dw_mr_inr);
        }

        // LOG(INFO) << "Updating trajectory." << std::endl;

        using namespace finalicp::se3;
        using namespace finalicp::traj;

        // Cache T_sr inverse
        const Eigen::Matrix4d T_sr_inv = options_.T_sr.inverse();

        // Sequential update
        for (auto& frame : trajectory_) {
            // Begin pose
            Time begin_slam_time(frame.begin_timestamp);
            const auto begin_T_mr = full_trajectory->getPoseInterpolator(begin_slam_time)->value().inverse().matrix();
            const auto begin_T_ms = begin_T_mr * T_sr_inv;
            frame.begin_R = begin_T_ms.block<3, 3>(0, 0);
            frame.begin_t = begin_T_ms.block<3, 1>(0, 3);

            // Mid pose
            Time mid_slam_time(static_cast<double>(frame.getEvalTime()));
            const auto mid_T_mr = full_trajectory->getPoseInterpolator(begin_slam_time)->value().inverse().matrix();
            const auto mid_T_ms = mid_T_mr * T_sr_inv;
            frame.setMidPose(mid_T_ms);

            // End pose
            Time end_slam_time(frame.end_timestamp);
            const auto end_T_mr = full_trajectory->getPoseInterpolator(begin_slam_time)->value().inverse().matrix();
            const auto end_T_ms = end_T_mr * T_sr_inv;
            frame.end_R = end_T_ms.block<3, 3>(0, 0);
            frame.end_t = end_T_ms.block<3, 1>(0, 3);
        }

        return trajectory_;
    }

    // ########################################################################
    // registerFrame 
    // ########################################################################

    /*
    The lidarinertialodom::registerFrame function processes a DataFrame for LiDAR-inertial odometry, 
    returning a RegistrationSummary with transformed points and pose data. 
    It validates the non-empty point cloud, adds a new frame to trajectory_, 
    and initializes timestamps and motion. For the first frame, 
    it sets initial states (pose, velocity, biases) and aligns gravity using IMU data, 
    while for subsequent frames, it downsamples the point cloud, 
    performs ICP registration with IMU and pose data, and updates the map with a delay. 
    All points are corrected using interpolated poses (sequential or parallel processing), 
    and the summary includes corrected points, keypoints, and the final pose (end_R, end_t). 
    Debug timers track performance if enabled, ensuring robust and efficient frame registration.*/

    auto lidarinertialodom::registerFrame(const DataFrame &const_frame) -> RegistrationSummary {
        RegistrationSummary summary;

        // Initialize timers for performance debugging if enabled
        std::vector<std::pair<std::string, std::unique_ptr<finalicp::Stopwatch<>>>> timer;
        if (options_.debug_print) {
            timer.emplace_back("initialization ..................... ", std::make_unique<finalicp::Stopwatch<>>(false));
            timer.emplace_back("icp ................................ ", std::make_unique<finalicp::Stopwatch<>>(false));
            timer.emplace_back("updateMap .......................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        }

        // Step 1: Validate input point cloud
        // Check if the input point cloud is empty; return failure if so

        // Step 2: Add new frame to trajectory
        // Create a new entry in the trajectory vector for the current frame
        int index_frame = trajectory_.size();
        trajectory_.emplace_back();

        // Step 3: Initialize frame metadata
        // Set up timestamp and motion data for the new frame
        initializeTimestamp(index_frame, const_frame);                                  //####!!! 1 tbb included
        initializeMotion(index_frame);                                                  //####!!! 2

        // Step 4: Process input point cloud
        // Convert and prepare the point cloud for registration
        if (!timer.empty()) timer[0].second->start();
        auto frame = initializeFrame(index_frame, const_frame.pointcloud);              //####!!! 3 tbb included
        if (!timer.empty()) timer[0].second->stop();

        // Step 5: Process frame based on frame index
        // Handle first frame initialization or subsequent frame registration
        std::vector<Point3D> keypoints;
        if (index_frame > 0) {
            // Determine voxel size for downsampling based on frame index
            double sample_voxel_size = index_frame < options_.init_num_frames
                ? options_.init_sample_voxel_size
                : options_.sample_voxel_size;

            // Step 5a: Downsample point cloud
            // Reduce point cloud density using grid sampling for efficiency
            if (!timer.empty()) timer[1].second->start();
            grid_sampling(frame, keypoints, sample_voxel_size, options_.num_threads, options_.sequential_threshold);   //####!!! 4
            if (!timer.empty()) timer[1].second->stop();

            // Step 5b: Perform Iterative Closest Point (ICP) registration
            // Align current frame with previous frames using IMU and pose data
            const auto& imu_data_vec = const_frame.imu_data_vec;
            const auto& pose_data_vec = const_frame.pose_data_vec;
            if (!timer.empty()) timer[1].second->start();
            summary.success = icp(index_frame, keypoints, imu_data_vec, pose_data_vec); //####!!! 5
            if (!timer.empty()) timer[1].second->stop();
            summary.keypoints = keypoints;
            if (!summary.success) {return summary;}
        } else {
            // Step 5c: Initialize first frame
            // Set up initial state and transformations for the trajectory start
            using namespace finalicp;
            using namespace finalicp::se3;
            using namespace finalicp::vspace;
            using namespace finalicp::traj;

            if (!timer.empty()) timer[0].second->start();

            // Define initial transformations and velocities
            math::se3::Transformation T_rm;
            math::se3::Transformation T_mi;
            math::se3::Transformation T_sr(options_.T_sr);
            Eigen::Matrix<double, 6, 1> w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 1> dw_mr_inr = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();

            // Initialize trajectory variables for the beginning of the frame
            const double begin_time = trajectory_[index_frame].begin_timestamp;
            Time begin_slam_time(begin_time);
            auto begin_T_rm_var = SE3StateVar::MakeShared(T_rm);
            auto begin_w_mr_inr_var = VSpaceStateVar<6>::MakeShared(w_mr_inr);
            auto begin_dw_mr_inr_var = VSpaceStateVar<6>::MakeShared(dw_mr_inr);
            auto begin_imu_biases = VSpaceStateVar<6>::MakeShared(b_zero);
            auto begin_T_mi_var = SE3StateVar::MakeShared(T_mi);
            trajectory_vars_.emplace_back(begin_slam_time, std::move(begin_T_rm_var), std::move(begin_w_mr_inr_var),
                                        std::move(begin_dw_mr_inr_var), std::move(begin_imu_biases), std::move(begin_T_mi_var));

            // Initialize trajectory variables for the end of the frame
            const double end_time = trajectory_[index_frame].end_timestamp;
            Time end_slam_time(end_time);
            auto end_T_rm_var = SE3StateVar::MakeShared(T_rm);
            auto end_w_mr_inr_var = VSpaceStateVar<6>::MakeShared(w_mr_inr);
            auto end_dw_mr_inr_var = VSpaceStateVar<6>::MakeShared(dw_mr_inr);
            auto end_imu_biases = VSpaceStateVar<6>::MakeShared(b_zero);
            auto end_T_mi_var = SE3StateVar::MakeShared(T_mi);
            trajectory_vars_.emplace_back(end_slam_time, std::move(end_T_rm_var), std::move(end_w_mr_inr_var),
                                        std::move(end_dw_mr_inr_var), std::move(end_imu_biases), std::move(end_T_mi_var));

            // Step 5d: Align gravity using IMU data
            // Ensure proper orientation by aligning with gravity vector

            Eigen::Matrix<double, 6, 1> xi_mi = initialize_gravity(const_frame.imu_data_vec);   //####!!! 6
            begin_T_mi_var->update(xi_mi);
            end_T_mi_var->update(xi_mi);

            to_marginalize_ = 1;

            // Step 5e: Initialize covariance matrices
            // Set initial uncertainties for pose, velocity, and acceleration
            Eigen::Matrix<double, 6, 6> P0_pose = options_.p0_pose.asDiagonal();
            Eigen::Matrix<double, 6, 6> P0_vel = options_.p0_vel.asDiagonal();
            Eigen::Matrix<double, 6, 6> P0_accel = options_.p0_accel.asDiagonal();
            trajectory_[index_frame].end_T_rm_cov = P0_pose;
            trajectory_[index_frame].end_w_mr_inr_cov = P0_vel;
            trajectory_[index_frame].end_dw_mr_inr_cov = P0_accel;

            summary.success = true;
            if (!timer.empty()) timer[0].second->stop();
        }

        // Step 6: Store processed points
        // Save the processed point cloud to the trajectory
        trajectory_[index_frame].points = std::move(frame);

        // Step 7: Update the map
        // Incorporate points into the global map, with optional delay
        if (!timer.empty()) timer[2].second->start();
        if (index_frame == 0) {
            updateMap(index_frame, index_frame);                                        //####!!! 7
        } else if ((index_frame - options_.delay_adding_points) > 0) {
            updateMap(index_frame, index_frame - options_.delay_adding_points);
        }
        if (!timer.empty()) timer[2].second->stop();

        // Step 8: Correct point cloud positions
        // Apply transformations to correct point positions based on trajectory
        // Validate trajectory poses for correction
        const auto& traj = trajectory_[index_frame];

        #if false
            std::vector<Point3D> all_corrected_points = const_frame.pointcloud;
            if (all_corrected_points.size() < options_.sequential_threshold) {
                // Step 8a: Sequential point correction
                // Correct points using linear interpolation for small datasets
                auto q_begin = Eigen::Quaterniond(traj.begin_R);
                auto q_end = Eigen::Quaterniond(traj.end_R);
                const Eigen::Vector3d t_begin = traj.begin_t;
                const Eigen::Vector3d t_end = traj.end_t;
                for (auto& point : all_corrected_points) {
                    const double alpha = point.alpha_timestamp;
                    const Eigen::Matrix3d R = q_begin.slerp(alpha, q_end).normalized().toRotationMatrix();
                    const Eigen::Vector3d t = (1.0 - alpha) * t_begin + alpha * t_end;
                    point.pt = R * point.raw_pt + t;
                }
            } else {
                // Step 8b: Parallel point correction
                // Use TBB for parallel processing of large point clouds
                tbb::concurrent_vector<Point3D> concurrent_points(const_frame.pointcloud.begin(), const_frame.pointcloud.end());
                tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, concurrent_points.size(), options_.sequential_threshold),
                    [&](const tbb::blocked_range<size_t>& range) {
                        auto q_begin = Eigen::Quaterniond(traj.begin_R);
                        auto q_end = Eigen::Quaterniond(traj.end_R);
                        const Eigen::Vector3d t_begin = traj.begin_t;
                        const Eigen::Vector3d t_end = traj.end_t;
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            auto& point = concurrent_points[i];
                            const double alpha = point.alpha_timestamp;
                            const Eigen::Matrix3d R = q_begin.slerp(alpha, q_end).normalized().toRotationMatrix();
                            const Eigen::Vector3d t = (1.0 - alpha) * t_begin + alpha * t_end;
                            point.pt = R * point.raw_pt + t;
                        }
                    });

                // Transfer corrected points back to std::vector
                all_corrected_points.assign(concurrent_points.begin(), concurrent_points.end());
            }
            summary.all_corrected_points = std::move(all_corrected_points);
        #endif

        // Step 9: Prepare output summary
        // Set corrected points, rotation, and translation for output
        summary.corrected_points = summary.keypoints;
        summary.R_ms = traj.end_R;
        summary.t_ms = traj.end_t;

        // Step 10: Output debug timers
        // Print timing information if debug mode is enabled
        // LOG(INFO) << "OUTER LOOP TIMERS" << std::endl;
        // if (options_.debug_print) {
            // for (size_t i = 0; i < timer.size(); i++)
            // LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
        // }

        return summary;
    }

    // ########################################################################
    // initializeTimestamp 
    // ########################################################################

    /*
    The lidarinertialodom::initializeTimestamp function determines the minimum and maximum timestamps from a DataFrame’s point cloud 
    for a specified frame (index_frame) in a LiDAR-inertial odometry system. 
    It validates the non-empty point cloud, computes the timestamp range, 
    and ensures timestamps are finite and ordered (min_timestamp ≤ max_timestamp), 
    throwing errors if invalid. The function assigns these to trajectory_[index_frame]’s begin_timestamp 
    and end_timestamp and sets the evaluation time to const_frame.timestamp, 
    ensuring temporal alignment for odometry tasks.*/

    void lidarinertialodom::initializeTimestamp(int index_frame, const DataFrame &const_frame) {
        // Validate input
        double min_timestamp = std::numeric_limits<double>::max();
        double max_timestamp = std::numeric_limits<double>::min();

        if (const_frame.pointcloud.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential processing
            for (const auto& point : const_frame.pointcloud) {
                min_timestamp = std::min(min_timestamp, point.timestamp);
                max_timestamp = std::max(max_timestamp, point.timestamp);
            }
        } else {
            // Parallel processing with TBB
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            auto result = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, const_frame.pointcloud.size(), options_.sequential_threshold),
                std::pair<double, double>{std::numeric_limits<double>::max(), std::numeric_limits<double>::min()},
                [&](const tbb::blocked_range<size_t>& range, std::pair<double, double> local) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const double ts = const_frame.pointcloud[i].timestamp;
                        local.first = std::min(local.first, ts);
                        local.second = std::max(local.second, ts);
                    }
                    return local;
                },
                [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                    return std::pair<double, double>{std::min(a.first, b.first), std::max(a.second, b.second)};
                }
            );
            min_timestamp = result.first;
            max_timestamp = result.second;
        }

        // Assign to trajectory
        trajectory_[index_frame].begin_timestamp = min_timestamp;
        trajectory_[index_frame].end_timestamp = max_timestamp;
        trajectory_[index_frame].setEvalTime(const_frame.timestamp);
    }

    // ########################################################################
    // initializeMotion 
    // ########################################################################

    /*
    The lidarinertialodom::initializeMotion function sets the start and end poses (rotation begin_R, end_R and translation begin_t, end_t) 
    for a frame (index_frame) in a LiDAR-inertial odometry system’s trajectory_. 
    It validates index_frame and the inverse sensor-to-robot transformation (T_sr). 
    For the first two frames (index_frame ≤ 1), it assigns T_sr’s rotation and translation to both poses. 
    For later frames, it extrapolates the end pose using the relative transformation between 
    the prior two frames’ end poses and sets the begin pose to the previous frame’s end pose, 
    ensuring smooth motion initialization using Eigen for matrix operations.*/

    void lidarinertialodom::initializeMotion(int index_frame) {

        // Cache T_sr inverse
        const Eigen::Matrix4d T_rs = options_.T_sr.inverse();

        if (index_frame <= 1) {
            // Initialize first two frames with T_rs
            trajectory_[index_frame].begin_R = T_rs.block<3, 3>(0, 0);
            trajectory_[index_frame].begin_t = T_rs.block<3, 1>(0, 3);
            trajectory_[index_frame].end_R = T_rs.block<3, 3>(0, 0);
            trajectory_[index_frame].end_t = T_rs.block<3, 1>(0, 3);
        } else {
            // Extrapolate end pose from previous two frames
            const auto& prev = trajectory_[index_frame - 1];
            const auto& prev_prev = trajectory_[index_frame - 2];
            
            // Compute relative transformation
            const Eigen::Matrix3d R_rel = prev.end_R * prev_prev.end_R.inverse();
            const Eigen::Vector3d t_rel = prev.end_R * prev_prev.end_R.inverse() * (prev.end_t - prev_prev.end_t);
            
            // Extrapolate end pose
            trajectory_[index_frame].end_R = R_rel * prev.end_R;
            trajectory_[index_frame].end_t = prev.end_t + t_rel;

            // Set begin pose to previous frame's end pose
            trajectory_[index_frame].begin_R = prev.end_R;
            trajectory_[index_frame].begin_t = prev.end_t;
        }
    }

    // ########################################################################
    // initializeFrame 
    // ########################################################################

    /*
    The lidarinertialodom::initializeFrame function preprocesses a 3D point cloud frame for LiDAR-inertial odometry 
    by validating inputs (index_frame and const_frame), copying the frame, 
    subsampling it with a voxel size (init_voxel_size for early frames, voxel_size for others), 
    shuffling points for unbiased processing, and transforming raw points to world coordinates using interpolated poses from trajectory_[index_frame]. 
    It employs slerp for rotations, linear interpolation for translations based on alpha_timestamp, 
    and processes points sequentially or in parallel (via TBB) based on size. The function returns a transformed, 
    subsampled point cloud, ensuring efficiency and robustness for registration tasks like ICP.*/

    std::vector<Point3D> lidarinertialodom::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {

        // Initialize point cloud
        std::vector<Point3D> frame = const_frame; // Copy necessary due to const input

        // Select voxel size
        const double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;

        // Subsample
        sub_sample_frame(frame, sample_size, options_.num_threads, options_.sequential_threshold);

        // Shuffle points to avoid bias
        std::mt19937_64 g(42); // Fixed seed for reproducibility
        std::shuffle(frame.begin(), frame.end(), g);

        // Validate poses
        const auto& traj = trajectory_[index_frame];

        auto q_begin = Eigen::Quaterniond(traj.begin_R);
        auto q_end = Eigen::Quaterniond(traj.end_R);
        const Eigen::Vector3d t_begin = traj.begin_t;
        const Eigen::Vector3d t_end = traj.end_t;

        if (frame.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential processing
            for (auto& point : frame) {
                const double alpha = point.alpha_timestamp;
                const Eigen::Matrix3d R = q_begin.slerp(alpha, q_end).normalized().toRotationMatrix();
                const Eigen::Vector3d t = (1.0 - alpha) * t_begin + alpha * t_end;
                point.pt = R * point.raw_pt + t;
            }
        } else {
            // Parallel processing with TBB
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size(), options_.sequential_threshold),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        auto& point = frame[i];
                        const double alpha = point.alpha_timestamp;
                        if (alpha < 0.0 || alpha > 1.0 || !std::isfinite(alpha)) {
                            throw std::runtime_error("Invalid alpha_timestamp in initializeFrame for frame " + std::to_string(index_frame));
                        }
                        const Eigen::Matrix3d R = q_begin.slerp(alpha, q_end).normalized().toRotationMatrix();
                        const Eigen::Vector3d t = (1.0 - alpha) * t_begin + alpha * t_end;
                        point.pt = R * point.raw_pt + t;
                    }
                });
        }

        return frame;
    }

    // ########################################################################
    // updateMap 
    // ########################################################################

    /*The lidarinertialodom::updateMap function updates the point cloud map for a specified frame (update_frame) in a LiDAR-inertial odometry system 
    by transforming and integrating points from trajectory_[update_frame].points into a global map, 
    using parameters like voxel size (size_voxel_map), minimum point distance (min_distance_points), 
    and maximum points per voxel (max_num_points_in_voxel) from options_. 
    It validates inputs (index_frame, update_frame, non-empty trajectory_vars_ for update_frame > 1, 
    and finite timestamps), then applies motion correction using SLAM trajectory interpolation to compute poses at unique point timestamps, 
    caching them either sequentially or in parallel (via TBB, based on sequential_threshold = 100). 
    Points are transformed from sensor to map coordinates using these poses and the inverse sensor-to-robot transformation (T_sr), 
    processed sequentially or in parallel, and added to the map with voxel-based filtering. 
    Optionally, it filters point lifetimes, clears the frame to free memory, 
    and removes distant points from the map based on the current frame’s end position (end_t) and a maximum distance (max_distance), 
    ensuring an efficient and accurate map update for odometry.*/

    void lidarinertialodom::updateMap(int index_frame, int update_frame) {

        // Map parameters
        const double kSizeVoxelMap = options_.size_voxel_map;
        const double kMinDistancePoints = options_.min_distance_points;
        const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

        // Update frame
        auto& frame = trajectory_[update_frame].points;
        if (frame.empty()) {
            return; // No points to add
        }

        // Motion correction with SLAM interpolation
        auto update_trajectory = finalicp::traj::singer::Interface::MakeShared(options_.qc_diag, options_.ad_diag);
        const double begin_timestamp = trajectory_[update_frame].begin_timestamp;
        const double end_timestamp = trajectory_[update_frame].end_timestamp;

        const finalicp::traj::Time begin_slam_time(begin_timestamp); // Fixed: Use finalicp::traj::Time
        const finalicp::traj::Time end_slam_time(end_timestamp);     // Fixed: Use finalicp::traj::Time

        // Add trajectory states
        size_t num_states = 0;
        for (size_t i = std::max(static_cast<int>(to_marginalize_) - 1, 0); i < trajectory_vars_.size(); i++) {
            const auto& var = trajectory_vars_.at(i);
            update_trajectory->add(var.time, var.T_rm, var.w_mr_inr, var.dw_mr_inr);
            num_states++;
            if (var.time == end_slam_time) break;
            if (var.time > end_slam_time) {
                throw std::runtime_error("Trajectory variable time exceeds end_slam_time in updateMap for frame " + std::to_string(update_frame));
            }
        }

        // LOG(INFO) << "Adding points to map between (inclusive): " << begin_slam_time.seconds() << " - "
            // << end_slam_time.seconds() << ", with num states: " << num_states << std::endl;

        // Collect unique timestamps
        std::set<double> unique_point_times_set;
        for (const auto& point : frame) {
            unique_point_times_set.insert(point.timestamp);
        }
        std::vector<double> unique_point_times(unique_point_times_set.begin(), unique_point_times_set.end());

        // Cache interpolated poses
        const Eigen::Matrix4d T_rs = options_.T_sr.inverse();

        std::map<double, Eigen::Matrix4d> T_ms_cache_map;
        if (unique_point_times.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential pose interpolation
            for (const auto& ts : unique_point_times) {
                const auto T_rm_intp_eval = update_trajectory->getPoseInterpolator(finalicp::traj::Time(ts));

                const Eigen::Matrix4d T_ms = T_rm_intp_eval->value().inverse().matrix() * T_rs;

                T_ms_cache_map[ts] = T_ms;
            }
        } else {
            // Parallel pose interpolation with TBB
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::concurrent_vector<Eigen::Matrix4d> T_ms_cache(unique_point_times.size());
            tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t jj = range.begin(); jj != range.end(); ++jj) {
                        const auto& ts = unique_point_times[jj];
                        const auto T_rm_intp_eval = update_trajectory->getPoseInterpolator(finalicp::traj::Time(ts));
                        const Eigen::Matrix4d T_ms = T_rm_intp_eval->value().inverse().matrix() * T_rs;
                        T_ms_cache[jj] = T_ms;
                    }
                });

            // Populate cache map
            for (size_t jj = 0; jj < unique_point_times.size(); ++jj) {
                T_ms_cache_map[unique_point_times[jj]] = T_ms_cache[jj];
            }
        }

        // Apply transformations
        if (frame.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential point transformation
            for (auto& point : frame) {
                try {
                    const auto& T_ms = T_ms_cache_map.at(point.timestamp);
                    point.pt = T_ms.block<3, 3>(0, 0) * point.raw_pt + T_ms.block<3, 1>(0, 3);
                } catch (const std::out_of_range&) {
                    throw std::runtime_error("Timestamp not found in cache in updateMap for frame " + std::to_string(update_frame));
                }
            }
        } else {
            // Parallel point transformation with TBB
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size(), options_.sequential_threshold),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        auto& point = frame[i];
                        try {
                            const auto& T_ms = T_ms_cache_map.at(point.timestamp);
                            point.pt = T_ms.block<3, 3>(0, 0) * point.raw_pt + T_ms.block<3, 1>(0, 3);
                        } catch (const std::out_of_range&) {
                            throw std::runtime_error("Timestamp not found in cache in updateMap for frame " + std::to_string(update_frame));
                        }
                    }
                });
        }

        // Update map
        map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);
        if (options_.filter_lifetimes) {
            map_.update_and_filter_lifetimes();
        }

        // Clear frame
        frame.clear();
        frame.shrink_to_fit();

        // Remove distant points
        const double kMaxDistance = options_.max_distance;
        const Eigen::Vector3d location = trajectory_[index_frame].end_t;
        map_.remove(location, kMaxDistance);
    }

    // ########################################################################
    // updateMap 
    // ########################################################################

    /*The lidarinertialodom::initialize_gravity function estimates the initial IMU-to-map transformation (T_mi) 
    for a LiDAR-inertial odometry system using a vector of IMU data (imu_data_vec). 
    It validates non-empty input and finite accelerations, initializes locked state variables (pose T_rm, biases, and velocities), 
    and sets up a noise model and L2 loss function based on options_.r_imu_acc and gravity. 
    It creates cost terms for acceleration errors (sequentially or in parallel via TBB, depending on size) 
    and adds a prior cost for T_mi with covariance from options_.T_mi_init_cov. 
    A Gauss-Newton solver optimizes the problem to refine T_mi, returning its 6D vector representation after ensuring finite results, 
    enabling gravity-aligned initialization for odometry.*/

    Eigen::Matrix<double, 6, 1> lidarinertialodom::initialize_gravity(const std::vector<finalicp::IMUData> &imu_data_vec) {

        // Initialize state variables
        const auto T_rm_init = finalicp::se3::SE3StateVar::MakeShared(math::se3::Transformation());
        math::se3::Transformation T_mi;
        const auto T_mi_var = finalicp::se3::SE3StateVar::MakeShared(T_mi);
        T_rm_init->locked() = true;

        Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> dw_zero = Eigen::Matrix<double, 6, 1>::Zero();
        const auto bias = finalicp::vspace::VSpaceStateVar<6>::MakeShared(b_zero);
        const auto dw_mr_inr = finalicp::vspace::VSpaceStateVar<6>::MakeShared(dw_zero);
        bias->locked() = true;
        dw_mr_inr->locked() = true;

        // Initialize noise model and loss function
        Eigen::Matrix<double, 3, 3> R = options_.r_imu_acc.asDiagonal();
        auto noise_model = finalicp::StaticNoiseModel<3>::MakeShared(R);
        auto loss_func = finalicp::L2LossFunc::MakeShared();

        // Create cost terms
        std::vector<finalicp::BaseCostTerm::ConstPtr> cost_terms;
        cost_terms.reserve(imu_data_vec.size()); // +1 for prior term
        if (imu_data_vec.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential cost term creation with std::vector
           for (const auto &imu_data : imu_data_vec) {
                auto acc_error_func = finalicp::imu::AccelerationError(T_rm_init, dw_mr_inr, bias, T_mi_var, imu_data.lin_acc);
                acc_error_func->setGravity(options_.gravity);
                const auto acc_cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(acc_error_func, noise_model, loss_func);
                cost_terms.emplace_back(acc_cost);
            }
        } else {
            // Parallel cost term creation with tbb::concurrent_vector
            tbb::concurrent_vector<finalicp::BaseCostTerm::ConstPtr> concurrent_cost_terms(imu_data_vec.size());
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, imu_data_vec.size(), options_.sequential_threshold),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const auto& imu_data = imu_data_vec[i];
                        auto acc_error_func = finalicp::imu::AccelerationError(T_rm_init, dw_mr_inr, bias, T_mi_var, imu_data.lin_acc);
                        acc_error_func->setGravity(options_.gravity);
                        const auto acc_cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(acc_error_func, noise_model, loss_func);
                        concurrent_cost_terms[i] = acc_cost;
                    }
                });

            // Transfer to std::vector
            cost_terms.assign(concurrent_cost_terms.begin(), concurrent_cost_terms.end());
        }

        {
            // Add prior cost term for T_mi
            Eigen::Matrix<double, 6, 6> init_T_mi_cov = options_.T_mi_init_cov.asDiagonal();
            init_T_mi_cov(3, 3) = 1.0;
            init_T_mi_cov(4, 4) = 1.0;
            init_T_mi_cov(5, 5) = 1.0;
            math::se3::Transformation T_mi_zero;
            auto T_mi_error = finalicp::se3::se3_error(T_mi_var, T_mi_zero);
            auto prior_noise_model = finalicp::StaticNoiseModel<6>::MakeShared(init_T_mi_cov);
            auto prior_loss_func = finalicp::L2LossFunc::MakeShared();
            cost_terms.emplace_back(finalicp::WeightedLeastSqCostTerm<6>::MakeShared(T_mi_error, prior_noise_model, prior_loss_func));
        }

        // Solve optimization problem
        finalicp::OptimizationProblem problem;
        for (const auto& cost : cost_terms) {
            problem.addCostTerm(cost);
        }
        problem.addStateVariable(T_mi_var);

        finalicp::GaussNewtonSolverNVA::Params params;
        params.verbose = options_.verbose;
        params.max_iterations = static_cast<unsigned int>(options_.max_iterations);
        finalicp::GaussNewtonSolverNVA solver(problem, params);
        solver.optimize();

        // Validate result
        // LOG(INFO) << "Initialization, T_mi:" << std::endl 
            // << T_mi_var->value().matrix() << std::endl
            // << "vec: " << T_mi_var->value().vec() << std::endl;
        
        return T_mi_var->value().vec();
    }

    // ########################################################################
    // icp 
    // ########################################################################

    bool lidarinertialodom::icp(int index_frame, std::vector<Point3D> &keypoints,
                           const std::vector<finalicp::IMUData> &imu_data_vec,
                           const std::vector<PoseData> &pose_data_vec) {

        using namespace finalicp;
        using namespace finalicp::se3;
        using namespace finalicp::traj;
        using namespace finalicp::vspace;
        using namespace finalicp::imu;

        // Step 1: Declare success flag for ICP
        // icp_success indicates if ICP alignment completes successfully (true by default)
        bool icp_success = true;

        // Step 2: Set up timers to measure performance (if debugging is enabled)
        // timer stores pairs of labels (e.g., "Initialization") and Stopwatch objects
        std::vector<std::pair<std::string, std::unique_ptr<finalicp::Stopwatch<>>>> timer;
        if (options_.debug_print) {
            // Add timers for different ICP phases (only if debug_print is true)
            timer.emplace_back("Update Transform ............... ", std::make_unique<finalicp::Stopwatch<>>(false));
            timer.emplace_back("Association .................... ", std::make_unique<finalicp::Stopwatch<>>(false));
            timer.emplace_back("Optimization ................... ", std::make_unique<finalicp::Stopwatch<>>(false));
            timer.emplace_back("Alignment ...................... ", std::make_unique<finalicp::Stopwatch<>>(false));
            timer.emplace_back("Initialization ................. ", std::make_unique<finalicp::Stopwatch<>>(false));
            timer.emplace_back("Marginalization ................ ", std::make_unique<finalicp::Stopwatch<>>(false));
        }

        // Step 3: Start the initialization timer (timer[4] = "Initialization")
        // Measures time taken to set up the SLAM trajectory
        if (!timer.empty()) timer[4].second->start();

        // Step 4: Create a new SLAM_TRAJ using singer::Interface
        // singer::Interface models the robot's trajectory (pose, velocity, etc.) over time
        // options_.qc_diag and ad_diag define noise models for trajectory dynamics
        auto SLAM_TRAJ = finalicp::traj::singer::Interface::MakeShared(options_.qc_diag, options_.ad_diag);

        // Step 5: Initialize containers for state variables and cost terms
        // SLAM_STATE_VAR holds variables to optimize (pose, velocity, etc.)
        // Cost terms define errors for optimization (e.g., point cloud alignment, IMU)
        std::vector<finalicp::StateVarBase::Ptr> SLAM_STATE_VAR;
        std::vector<finalicp::BaseCostTerm::ConstPtr> prior_cost_terms; // Prior constraints
        std::vector<finalicp::BaseCostTerm::ConstPtr> meas_cost_terms; // Point cloud measurements
        std::vector<finalicp::BaseCostTerm::ConstPtr> imu_cost_terms; // IMU measurements
        std::vector<finalicp::BaseCostTerm::ConstPtr> pose_meas_cost_terms; // Pose measurements
        std::vector<finalicp::BaseCostTerm::ConstPtr> imu_prior_cost_terms; // IMU bias priors
        std::vector<finalicp::BaseCostTerm::ConstPtr> T_mi_prior_cost_terms; // T_mi priors

        // Step 6: Track indices for trajectory variables
        // prev_trajectory_var_index points to the last state in trajectory_vars_
        // curr_trajectory_var_index tracks new states added for this frame
        const size_t prev_trajectory_var_index = trajectory_vars_.size() - 1;   //??
        size_t curr_trajectory_var_index = trajectory_vars_.size() - 1;         //??

        // Step 7: Validate inputs and previous state
        // Ensure index_frame is valid and trajectory_vars_ is not empty
        // Logging system
        // LOG(INFO) << "[ICP] prev scan end time: " << trajectory_[index_frame - 1].end_timestamp << std::endl;
        

        // Step 8: Get the previous frame's end timestamp
        // prev_time is the end time of the previous frame (index_frame - 1)
        const double PREV_TIME = trajectory_[index_frame - 1].end_timestamp;

        // Step 9: Verify the previous state’s timestamp matches prev_time
        // trajectory_vars_.back().time should equal prev_time for consistency
        if (trajectory_vars_.back().time != finalicp::traj::Time(PREV_TIME)) {
            throw std::runtime_error("Previous scan end time mismatch in icp for frame " + std::to_string(index_frame));
        }

        // Step 10: Retrieve previous frame’s state variables
        // These describe the robot’s state at the end of the previous frame
        const auto& PREV_VAR = trajectory_vars_.back(); // Last state in trajectory_vars_
        finalicp::traj::Time prev_slam_time = PREV_VAR.time; // Timestamp
        math::se3::Transformation prev_T_rm = PREV_VAR.T_rm->value(); // Map-to-robot pose
        Eigen::Matrix<double, 6, 1> prev_w_mr_inr = PREV_VAR.w_mr_inr->value(); // Velocity
        Eigen::Matrix<double, 6, 1> prev_dw_mr_inr = PREV_VAR.dw_mr_inr->value(); // Acceleration
        Eigen::Matrix<double, 6, 1> prev_imu_biases = PREV_VAR.imu_biases->value(); // IMU biases
        math::se3::Transformation prev_T_mi = PREV_VAR.T_mi->value(); // IMU-to-map transformation

        // Step 11: Validate previous state values
        // Ensure all state values are finite (not NaN or infinite)

        // Step 12: Get pointers to previous state variables
        // These are shared pointers to state objects for SLAM optimization
        const auto prev_T_rm_var = PREV_VAR.T_rm; // Pose variable
        const auto prev_w_mr_inr_var = PREV_VAR.w_mr_inr; // Velocity variable
        const auto prev_dw_mr_inr_var = PREV_VAR.dw_mr_inr; // Acceleration variable
        const auto prev_imu_biases_var = PREV_VAR.imu_biases; // IMU biases variable
        auto prev_T_mi_var = PREV_VAR.T_mi; // T_mi variable (non-const for updates)

        // Step 13: Prepare ground truth IMU-to-map transformation (T_mi_gt)
        // xi_ig (assumed 6D vector) defines the ground truth T_mi (rotation + translation)
        // Zero translation to focus on rotation (common for IMU alignment)
        bool use_T_mi_gt = options_.use_T_mi_gt; // Use ground truth T_mi if true
        Eigen::Matrix4d T_mi_gt_mat = math::se3::Transformation(options_.xi_ig).matrix();
        T_mi_gt_mat.block<3, 1>(0, 3).setZero(); // Set translation (x, y, z) to 0
        math::se3::Transformation T_mi_gt(T_mi_gt_mat); // Create transformation object

        // Step 14: Add previous state to SLAM trajectory
        // This anchors the trajectory at the previous frame’s end state
        SLAM_TRAJ->add(prev_slam_time, prev_T_rm_var, prev_w_mr_inr_var, prev_dw_mr_inr_var);

        // Step 15: Add previous state variables to optimization list
        // These variables will be optimized (if not locked) in ICP
        SLAM_STATE_VAR.emplace_back(prev_T_rm_var); // Add pose
        SLAM_STATE_VAR.emplace_back(prev_w_mr_inr_var); // Add velocity
        SLAM_STATE_VAR.emplace_back(prev_dw_mr_inr_var); // Add acceleration

        // Step 16: Handle IMU-related state variables (if IMU is enabled)
        if (options_.use_imu) {
            // Add IMU biases to optimization (biases evolve over time)
            SLAM_STATE_VAR.emplace_back(prev_imu_biases_var);

            // Decide how to handle T_mi (IMU-to-map transformation)
            if (use_T_mi_gt) {
                // Use ground truth T_mi: set T_mi to T_mi_gt and lock it
                prev_T_mi_var->update(T_mi_gt.vec()); // Update to ground truth (rotation only)
                // LOG(INFO) << "prev_T_mi_var->value()" << std::endl;
                // LOG(INFO) << prev_T_mi_var->value() << std::endl;
                prev_T_mi_var->locked() = true; // Lock to prevent optimization
            } else {
                // Use estimated T_mi: decide if it should be optimized
                if (!options_.T_mi_init_only || index_frame == 1) {
                    // Optimize T_mi if not init-only or this is the first frame
                    // T_mi_init_only=true means optimize T_mi only at index_frame=1
                    SLAM_STATE_VAR.emplace_back(prev_T_mi_var); // Add for optimization
                }
                // If T_mi_init_only=true and index_frame>1, T_mi stays as is (not optimized)
            }
        }

        ///################################################################################
        // LOG(INFO) << "[ICP] curr scan end time: " << trajectory_[index_frame].end_timestamp << std::endl;
        // LOG(INFO) << "[ICP] total num new states: " << 1 << std::endl;

        // Step 17: Get the current frame’s end timestamp
        // curr_time tells us when this frame ends
        const double CURR_TIME = trajectory_[index_frame].end_timestamp;

        // Step 18: Calculate the number of new states to add
        // num_extra_states is how many extra points (knots) to add between PREV_TIME and curr_time
        // +1 includes the mandatory end state at curr_time
        const int NUM_STATES = options_.num_extra_states + 1;

        // Step 19: Create timestamps (knot times) for new states
        // knot_times lists when each new state occurs, from PREV_TIME to curr_time
        const double TIME_DIFF = (CURR_TIME - PREV_TIME) / static_cast<double>(NUM_STATES);
        std::vector<double> KNOT_TIMES;
        KNOT_TIMES.reserve(NUM_STATES);
        for (int i = 0; i < options_.num_extra_states; ++i) {
            KNOT_TIMES.emplace_back(PREV_TIME + (double)(i + 1) * TIME_DIFF);
        }
        KNOT_TIMES.emplace_back(CURR_TIME);

        // Step 20: Estimate the next pose (T_next) for the current frame
        // T_next predicts the robot’s position at curr_time based on past frames
        Eigen::Matrix4d T_NEXT_MAT = Eigen::Matrix4d::Identity();
        if (index_frame > 2) {
            // Use the last two frames to predict motion (rotation and translation)
            const auto& prev = trajectory_[index_frame - 1];
            const auto& prev_prev = trajectory_[index_frame - 2];

            // Calculate relative motion between frames
            const Eigen::Matrix3d R_rel = prev.end_R * prev_prev.end_R.inverse();
            const Eigen::Vector3d t_rel = prev.end_t - prev_prev.end_t;
            // Predict next pose by applying relative motion
            T_NEXT_MAT.block<3, 3>(0, 0) = R_rel * prev.end_R;
            T_NEXT_MAT.block<3, 1>(0, 3) = prev.end_t + R_rel * t_rel;
        } else {
            // For early frames, use trajectory interpolation
            T_NEXT_MAT = SLAM_TRAJ->getPoseInterpolator(finalicp::traj::Time(KNOT_TIMES.back()))->value().inverse().matrix();
        }
        const math::se3::Transformation T_NEXT(Eigen::Matrix4d(T_NEXT_MAT.inverse()));

        // Step 21: Prepare default values for new states
        // w_next and dw_next are initial velocity and acceleration (set to zero)
        // const Eigen::Matrix<double, 6, 1> w_next = Eigen::Matrix<double, 6, 1>::Zero();
        // const Eigen::Matrix<double, 6, 1> dw_next = Eigen::Matrix<double, 6, 1>::Zero();

        // Step 22: Add new states for the current frame sequentially
        // Each state includes pose, velocity, acceleration, etc., at a knot time
        for (size_t i = 0; i < KNOT_TIMES.size(); ++i) {
            // Get timestamp for this state
            double knot_time = KNOT_TIMES[i];
            finalicp::traj::Time knot_slam_time(knot_time);

            // Predict pose using previous velocity for intermediate states, T_next for end state
            const Eigen::Matrix<double, 6, 1> xi_mr_inr_odo = (knot_slam_time - prev_slam_time).seconds() * prev_w_mr_inr;
            const auto knot_T_rm = math::se3::Transformation(xi_mr_inr_odo) * prev_T_rm;
            //if (i == num_states - 1) knot_T_rm = T_next; // Use predicted pose for end state

            // Create state variables
            const auto T_rm_var = finalicp::se3::SE3StateVar::MakeShared(knot_T_rm); // New pose
            const auto w_mr_inr_var = finalicp::vspace::VSpaceStateVar<6>::MakeShared(prev_w_mr_inr); // Copy velocity
            const auto dw_mr_inr_var = finalicp::vspace::VSpaceStateVar<6>::MakeShared(prev_dw_mr_inr); // Copy acceleration
            const auto imu_biases_var = finalicp::vspace::VSpaceStateVar<6>::MakeShared(prev_imu_biases); // Copy biases

            // LOG(INFO) << "init: w_mr_inr_var->value() " << w_mr_inr_var->value().transpose() << std::endl;
            // LOG(INFO) << "init: dw_mr_inr_var->value() " << dw_mr_inr_var->value().transpose() << std::endl;

            // Add state to trajectory
            SLAM_TRAJ->add(knot_slam_time, T_rm_var, w_mr_inr_var, dw_mr_inr_var);

            // Add state variables to optimization list
            SLAM_STATE_VAR.emplace_back(T_rm_var); // Add pose
            SLAM_STATE_VAR.emplace_back(w_mr_inr_var); // Add velocity
            SLAM_STATE_VAR.emplace_back(dw_mr_inr_var); // Add acceleration
            if (options_.use_imu) {
                SLAM_STATE_VAR.emplace_back(imu_biases_var); // Add IMU biases
            }

            const auto T_mi_var = finalicp::se3::SE3StateVar::MakeShared(use_T_mi_gt ? math::se3::Transformation() : prev_T_mi);

            if (use_T_mi_gt) {
                T_mi_var->locked() = true; // Lock T_mi if ground truth or init-only
                trajectory_vars_.emplace_back(knot_slam_time, T_rm_var, w_mr_inr_var, dw_mr_inr_var, imu_biases_var, T_mi_var);
            } else {
                 if (options_.use_imu) {
                    if (options_.T_mi_init_only) {
                        T_mi_var->locked() = true;
                    } else {
                        SLAM_STATE_VAR.emplace_back(T_mi_var); // Optimize T_mi
                    }
                }
                trajectory_vars_.emplace_back(knot_slam_time, T_rm_var, w_mr_inr_var, dw_mr_inr_var, imu_biases_var, T_mi_var);
                
            }
            // Update index for next state
            curr_trajectory_var_index++;
        }

        ///################################################################################

        // Step 24: Add prior cost terms for the initial frame (index_frame == 1)
        // Priors set initial guesses for pose, velocity, and acceleration to guide optimization
        if (index_frame == 1) {
            // Get the previous frame’s state variables
            const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

            // Define initial pose (T_rm, identity), velocity (w_mr_inr, zero), and acceleration (dw_mr_inr, zero)
            math::se3::Transformation T_rm; // Identity transformation (no initial offset)
            Eigen::Matrix<double, 6, 1> w_mr_inr = Eigen::Matrix<double, 6, 1>::Zero(); // Zero initial velocity
            Eigen::Matrix<double, 6, 1> dw_mr_inr = Eigen::Matrix<double, 6, 1>::Zero(); // Zero initial acceleration

            // Set covariance matrices for priors using options_ (uncertainty in initial guesses)
            Eigen::Matrix<double, 6, 6> P0_pose = Eigen::Matrix<double, 6, 6>::Identity();
            P0_pose.diagonal() = options_.p0_pose; // Pose covariance
            Eigen::Matrix<double, 6, 6> P0_vel = Eigen::Matrix<double, 6, 6>::Identity();
            P0_vel.diagonal() = options_.p0_vel; // Velocity covariance
            Eigen::Matrix<double, 6, 6> P0_accel = Eigen::Matrix<double, 6, 6>::Identity();
            P0_accel.diagonal() = options_.p0_accel; // Acceleration covariance

            // Add prior cost terms to constrain initial state
            SLAM_TRAJ->addPosePrior(PREV_VAR.time, T_rm, P0_pose); // Constrain initial pose
            SLAM_TRAJ->addVelocityPrior(PREV_VAR.time, w_mr_inr, P0_vel); // Constrain initial velocity
            SLAM_TRAJ->addAccelerationPrior(PREV_VAR.time, dw_mr_inr, P0_accel); // Constrain initial acceleration

            if (PREV_VAR.time != Time(trajectory_.at(0).end_timestamp)) throw std::runtime_error{"inconsistent timestamp"};
        }

        // Step 25: Add IMU-related prior cost terms (if IMU is enabled)
        if (options_.use_imu) {

            // For the initial frame, add a prior for IMU biases
            if (index_frame == 1) {

                // Get the previous frame’s state variables
                const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

                // Define zero biases as the prior guess (assume no initial bias)
                Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();

                // Set covariance for initial IMU bias prior
                Eigen::Matrix<double, 6, 6> init_bias_cov = Eigen::Matrix<double, 6, 6>::Identity();
                init_bias_cov.block<3, 3>(0, 0).diagonal() = options_.p0_bias_accel; // Accelerometer bias covariance
                init_bias_cov.block<3, 3>(3, 3).diagonal() = Eigen::Vector3d::Constant(options_.p0_bias_gyro); // Gyroscope bias covariance

                // Create cost term to constrain initial IMU biases
                auto bias_error = finalicp::vspace::vspace_error<6>(PREV_VAR.imu_biases, b_zero);
                auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(init_bias_cov);
                auto loss_func = finalicp::L2LossFunc::MakeShared();
                const auto bias_prior_factor = finalicp::WeightedLeastSqCostTerm<6>::MakeShared(bias_error, noise_model, loss_func);
                imu_prior_cost_terms.emplace_back(bias_prior_factor);

            } else if (options_.use_bias_prior_after_init) {  // For subsequent frames, add IMU bias prior if enabled
                // Get the previous frame’s state variables
                const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

                // Define zero biases as the prior guess
                Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();

                // Set covariance for ongoing IMU bias prior
                Eigen::Matrix<double, 6, 6> bias_cov = Eigen::Matrix<double, 6, 6>::Identity();
                bias_cov.block<3, 3>(0, 0).diagonal() = Eigen::Vector3d::Constant(options_.pk_bias_accel); // Accelerometer bias covariance
                bias_cov.block<3, 3>(3, 3).diagonal() = Eigen::Vector3d::Constant(options_.pk_bias_gyro); // Gyroscope bias covariance

                // Create cost term to constrain IMU biases
                auto bias_error = finalicp::vspace::vspace_error<6>(PREV_VAR.imu_biases, b_zero);
                auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(bias_cov);
                auto loss_func = finalicp::L2LossFunc::MakeShared();
                const auto bias_prior_factor = finalicp::WeightedLeastSqCostTerm<6>::MakeShared(bias_error, noise_model, loss_func);
                imu_prior_cost_terms.emplace_back(bias_prior_factor);
            }

            // For the initial frame, add a prior for T_mi if not using ground truth
            if (index_frame == 1 && !use_T_mi_gt) {
                // Get the previous frame’s state variables
                const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

                // Set covariance for initial T_mi prior
                Eigen::Matrix<double, 6, 6> init_T_mi_cov = Eigen::Matrix<double, 6, 6>::Identity();
                init_T_mi_cov.diagonal() = options_.T_mi_init_cov;

                // Use current T_mi as the prior guess
                math::se3::Transformation T_mi = PREV_VAR.T_mi->value();

                // LOG(INFO) << "T_mi(0)" << std::endl << T_mi.matrix() << std::endl;

                // Create cost term to constrain initial T_mi
                auto T_mi_error = finalicp::se3::se3_error(PREV_VAR.T_mi, T_mi);
                auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(init_T_mi_cov);
                auto loss_func = finalicp::L2LossFunc::MakeShared();
                const auto T_mi_prior_factor = finalicp::WeightedLeastSqCostTerm<6>::MakeShared(T_mi_error, noise_model, loss_func);
                T_mi_prior_cost_terms.emplace_back(T_mi_prior_factor);
            }

            // For subsequent frames, add T_mi prior if enabled and not using ground truth
            if (!options_.T_mi_init_only && !use_T_mi_gt && options_.use_T_mi_prior_after_init) {
                // Get the previous frame’s state variables
                const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

                // Set covariance for ongoing T_mi prior
                Eigen::Matrix<double, 6, 6> T_mi_cov = Eigen::Matrix<double, 6, 6>::Identity();
                T_mi_cov.diagonal() = options_.T_mi_prior_cov;

                // Use identity T_mi as the prior guess (no offset)
                math::se3::Transformation T_mi;

                // Create cost term to constrain T_mi
                auto T_mi_error = finalicp::se3::se3_error(PREV_VAR.T_mi, T_mi);
                auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(T_mi_cov);
                auto loss_func = finalicp::L2LossFunc::MakeShared();
                const auto T_mi_prior_factor = finalicp::WeightedLeastSqCostTerm<6>::MakeShared(T_mi_error, noise_model, loss_func);
                T_mi_prior_cost_terms.emplace_back(T_mi_prior_factor);
            }
        }

        // Step 26: Stop the initialization timer
        // Marks the end of the initialization phase (already handled in Step 18, included for completeness)
        if (!timer.empty()) timer[4].second->stop();

        ///################################################################################

        // Step 28: Update sliding window variables
        // Add state variables to the sliding window filter for optimization
        {
            // Step 27: Start the marginalization timer
            // Timer[5] measures the time taken to update the sliding window filter
            if (!timer.empty()) timer[5].second->start();

            // For the initial frame, include the previous frame’s state variables
            if (index_frame == 1) {
                // Get the previous frame’s state variables
                const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

                // Add pose, velocity, and acceleration to the sliding window
                sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.T_rm, PREV_VAR.w_mr_inr, PREV_VAR.dw_mr_inr});

                // If IMU is enabled, add IMU biases and optionally T_mi
                if (options_.use_imu) {
                    sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.imu_biases});
                    if (!use_T_mi_gt) {
                        sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.T_mi});
                    }
                }
            }

            // Add state variables for new states in the current frame
            for (size_t i = prev_trajectory_var_index + 1; i <= curr_trajectory_var_index; ++i) {
                // Get the current state’s variables
                const auto& VAR = trajectory_vars_.at(i);

                // Add pose, velocity, and acceleration to the sliding window
                sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{VAR.T_rm, VAR.w_mr_inr, VAR.dw_mr_inr});

                // If IMU is enabled, add IMU biases and optionally T_mi
                if (options_.use_imu) {
                    sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{VAR.imu_biases});
                    if (!options_.T_mi_init_only && !use_T_mi_gt) {
                        sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{VAR.T_mi});
                    }
                }
            }
        }

        // Step 29: Marginalize old state variables to keep the sliding window manageable
        // Remove states older than delay_adding_points frames ago
        if (index_frame > options_.delay_adding_points && options_.delay_adding_points >= 0) {

            // Collect state variables to marginalize (from to_marginalize_ up to marg_time)
            std::vector<finalicp::StateVarBase::Ptr> marg_vars;
            size_t num_states = 0;
            const double begin_marg_time = trajectory_vars_.at(to_marginalize_).time.seconds();
            double end_marg_time = trajectory_vars_.at(to_marginalize_).time.seconds();

            // Define the marginalization time based on delay_adding_points
            const double marg_time = trajectory_.at(index_frame - options_.delay_adding_points - 1).end_timestamp;
            finalicp::traj::Time marg_slam_time(marg_time);

            for (size_t i = to_marginalize_; i <= curr_trajectory_var_index; ++i) {
                const auto& VAR = trajectory_vars_.at(i);
                if (VAR.time <= marg_slam_time) {
                    // Update end marginalization time
                    end_marg_time = VAR.time.seconds();

                    // Add state variables to marginalize
                    marg_vars.emplace_back(VAR.T_rm);
                    marg_vars.emplace_back(VAR.w_mr_inr);
                    marg_vars.emplace_back(VAR.dw_mr_inr);
                    if (options_.use_imu) {
                        marg_vars.emplace_back(VAR.imu_biases);
                        if (!VAR.T_mi->locked()) {
                            marg_vars.emplace_back(VAR.T_mi);
                        }
                    }
                    num_states++;
                } else {
                    // Update to_marginalize_ to the first non-marginalized state
                    to_marginalize_ = i;
                    break;
                }
            }

            // Marginalize the collected variables if any
            if (!marg_vars.empty()) {
                sliding_window_filter_->marginalizeVariable(marg_vars);
                // LOG(INFO) << "Marginalizing time (inclusive): " << begin_marg_time << " - " << end_marg_time << ", with num states: " << num_states << std::endl;
            }

            // Step 30: Stop the marginalization timer
            if (!timer.empty()) timer[5].second->stop();
        }

        

        ///################################################################################
        // Step 31: Restart the initialization timer for query point evaluation
        // Timer[4] measures the time taken to process query points and IMU cost terms
        if (!timer.empty()) timer[4].second->start();

        // Step 32: Collect unique timestamps from keypoints for query point evaluation
        // unique_point_times lists distinct timestamps to query the SLAM trajectory
        std::set<double> unique_point_times_set;
        for (const auto& keypoint : keypoints) {
            unique_point_times_set.insert(keypoint.timestamp);
        }
        std::vector<double> unique_point_times(unique_point_times_set.begin(), unique_point_times_set.end());

        // Configure IMU cost term options
        finalicp::IMUSuperCostTerm::Options imu_options;
        imu_options.num_threads = options_.num_threads; // Thread count (sequential here, but set for compatibility)
        imu_options.acc_loss_sigma = options_.acc_loss_sigma; // Accelerometer loss parameter
        imu_options.use_accel = options_.use_accel; // Whether to use acceleration data

        // Map string loss functions to enums
        if (options_.acc_loss_func == "L2") imu_options.acc_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::L2;
        else if (options_.acc_loss_func == "DCS") imu_options.acc_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::DCS;
        else if (options_.acc_loss_func == "CAUCHY") imu_options.acc_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::CAUCHY;
        else if (options_.acc_loss_func == "GM") imu_options.acc_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::GM;
        else throw std::runtime_error("Invalid acc_loss_func: " + options_.acc_loss_func);
        imu_options.gyro_loss_sigma = options_.gyro_loss_sigma; // Gyroscope loss parameter
        if (options_.gyro_loss_func == "L2") imu_options.gyro_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::L2;
        else if (options_.gyro_loss_func == "DCS") imu_options.gyro_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::DCS;
        else if (options_.gyro_loss_func == "CAUCHY") imu_options.gyro_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::CAUCHY;
        else if (options_.gyro_loss_func == "GM") imu_options.gyro_loss_func = finalicp::IMUSuperCostTerm::LOSS_FUNC::GM;
        else throw std::runtime_error("Invalid gyro_loss_func: " + options_.gyro_loss_func);
        imu_options.gravity(2, 0) = options_.gravity; // Gravity vector (z-axis)
        imu_options.r_imu_acc = options_.r_imu_acc; // Accelerometer noise
        imu_options.r_imu_ang = options_.r_imu_ang; // Gyroscope noise

        const auto imu_super_cost_term = IMUSuperCostTerm::MakeShared(SLAM_TRAJ, prev_slam_time, finalicp::traj::Time(KNOT_TIMES.back()), trajectory_vars_[prev_trajectory_var_index].imu_biases,
                trajectory_vars_[prev_trajectory_var_index + 1].imu_biases, trajectory_vars_[prev_trajectory_var_index].T_mi,
                trajectory_vars_[prev_trajectory_var_index + 1].T_mi, imu_options);

        // Step 33: Add IMU cost terms (if IMU is enabled)
        // IMU cost terms constrain the trajectory using accelerometer and gyroscope measurements
        if (options_.use_imu) {

            #if false
                imu_super_cost_term->set(imu_data_vec);
                imu_super_cost_term->init();
            #else

                // Create individual IMU cost terms for each measurement
                imu_cost_terms.reserve(imu_data_vec.size()); // Reserve 
                Eigen::Matrix<double, 3, 3> R_acc = Eigen::Matrix<double, 3, 3>::Identity();
                R_acc.diagonal() = options_.r_imu_acc;
                Eigen::Matrix<double, 3, 3> R_ang = Eigen::Matrix<double, 3, 3>::Identity();
                R_ang.diagonal() = options_.r_imu_ang;
                const auto acc_noise_model = finalicp::StaticNoiseModel<3>::MakeShared(R_acc);
                const auto gyro_noise_model = finalicp::StaticNoiseModel<3>::MakeShared(R_ang);
                const auto acc_loss_func = finalicp::CauchyLossFunc::MakeShared(1.0);
                const auto gyro_loss_func = finalicp::L2LossFunc::MakeShared();

                for (const auto& imu_data : imu_data_vec) {
                    // Find the knot interval containing the IMU timestamp
                    size_t i = prev_trajectory_var_index;
                    for (; i < trajectory_vars_.size() - 1; i++) {
                        if (imu_data.timestamp >= trajectory_vars_[i].time.seconds() && imu_data.timestamp < trajectory_vars_[i + 1].time.seconds()) {
                            break;
                        }
                    }
                    if (imu_data.timestamp < trajectory_vars_[i].time.seconds() || imu_data.timestamp >= trajectory_vars_[i + 1].time.seconds()) {
                        throw std::runtime_error("IMU timestamp not within knot times: " + std::to_string(imu_data.timestamp));
                    }

                    // Interpolate IMU biases between knots
                    const auto bias_intp_eval = finalicp::vspace::VSpaceInterpolator<6>::MakeShared(
                        finalicp::traj::Time(imu_data.timestamp), trajectory_vars_[i].imu_biases, trajectory_vars_[i].time,
                        trajectory_vars_[i + 1].imu_biases, trajectory_vars_[i + 1].time
                    );

                    // Interpolate pose, velocity, and acceleration
                    const auto T_rm_intp_eval = SLAM_TRAJ->getPoseInterpolator(finalicp::traj::Time(imu_data.timestamp));
                    const auto w_mr_inr_intp_eval = SLAM_TRAJ->getVelocityInterpolator(finalicp::traj::Time(imu_data.timestamp));
                    const auto dw_mr_inr_intp_eval = SLAM_TRAJ->getAccelerationInterpolator(finalicp::traj::Time(imu_data.timestamp));

                    // Create acceleration error term
                    const auto acc_error_func = [&]() -> finalicp::imu::AccelerationErrorEvaluator::Ptr {
                        if (options_.T_mi_init_only) {
                            // Use fixed T_mi for initial frame
                            return finalicp::imu::AccelerationError(T_rm_intp_eval, dw_mr_inr_intp_eval, bias_intp_eval, trajectory_vars_[i].T_mi, imu_data.lin_acc);
                        } else {
                            // Interpolate T_mi between knots
                            const auto T_mi_intp_eval = finalicp::se3::PoseInterpolator::MakeShared(
                                finalicp::traj::Time(imu_data.timestamp), trajectory_vars_[i].T_mi, trajectory_vars_[i].time,
                                trajectory_vars_[i + 1].T_mi, trajectory_vars_[i + 1].time
                            );
                            return finalicp::imu::AccelerationError(T_rm_intp_eval, dw_mr_inr_intp_eval, bias_intp_eval, T_mi_intp_eval, imu_data.lin_acc);
                        }
                    }();

                    // Set gravity and timestamp for acceleration error
                    acc_error_func->setGravity(options_.gravity);
                    acc_error_func->setTime(finalicp::traj::Time(imu_data.timestamp));

                    // Create gyroscope error term
                    const auto gyro_error_func = finalicp::imu::GyroError(w_mr_inr_intp_eval, bias_intp_eval, imu_data.ang_vel);
                    gyro_error_func->setTime(finalicp::traj::Time(imu_data.timestamp));

                    // Add acceleration cost term (if enabled)
                    if (options_.use_accel) {
                        const auto acc_cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(acc_error_func, acc_noise_model, acc_loss_func);
                        imu_cost_terms.emplace_back(acc_cost);
                    }

                    // Add gyroscope cost term
                    const auto gyro_cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(gyro_error_func, gyro_noise_model, gyro_loss_func);
                    imu_cost_terms.emplace_back(gyro_cost);
                }
            #endif

            // Step 34: Add prior cost terms for IMU biases
            // Constrain changes in IMU biases between consecutive states
            // Get IMU prior cost terms
            {
                // Set covariance for IMU bias prior
                Eigen::Matrix<double, 6, 6> bias_cov = Eigen::Matrix<double, 6, 6>::Identity();
                bias_cov.block<3, 3>(0, 0).diagonal() =  options_.q_bias_accel; // Accelerometer bias covariance
                bias_cov.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * options_.q_bias_gyro; // Gyroscope bias covariance

                // Create noise model and loss function for bias prior
                auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(bias_cov);
                auto loss_func = finalicp::L2LossFunc::MakeShared();

                // Add prior for each pair of consecutive states
                size_t i = prev_trajectory_var_index;
                for (; i < trajectory_vars_.size() - 1; i++) {
                    // Create error term: difference between consecutive biases
                    const auto nbk = finalicp::vspace::NegationEvaluator<6>::MakeShared(trajectory_vars_[i + 1].imu_biases);
                    auto bias_error = finalicp::vspace::AdditionEvaluator<6>::MakeShared(trajectory_vars_[i].imu_biases, nbk);

                    // Create and add prior cost term
                    const auto bias_prior_factor = finalicp::WeightedLeastSqCostTerm<6>::MakeShared(bias_error, noise_model, loss_func);
                    imu_prior_cost_terms.emplace_back(bias_prior_factor);
                }
            }

            // Step 35: Add prior cost terms for T_mi (if not init-only and not using ground truth)
            // Constrain changes in T_mi between consecutive states
            // Get T_mi prior cost terms
            if (!options_.T_mi_init_only && !use_T_mi_gt) {
                // Define identity T_mi as the prior guess (no relative change)
                const auto T_mi = math::se3::Transformation();

                // Set covariance for T_mi prior
                Eigen::Matrix<double, 6, 6> T_mi_cov = Eigen::Matrix<double, 6, 6>::Identity();
                T_mi_cov.diagonal() = options_.qg_diag;

                // Create noise model and loss function for T_mi prior
                auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(T_mi_cov);
                auto loss_func = finalicp::L2LossFunc::MakeShared();

                // Add prior for each pair of consecutive states
                size_t i = prev_trajectory_var_index;
                for (; i < trajectory_vars_.size() - 1; i++) {
                    // Create error term: relative transformation between consecutive T_mi
                    auto T_mi_error = finalicp::se3::se3_error(finalicp::se3::compose_rinv(trajectory_vars_[i + 1].T_mi, trajectory_vars_[i].T_mi), T_mi);

                    // Create and add prior cost term
                    const auto T_mi_prior_factor = finalicp::WeightedLeastSqCostTerm<6>::MakeShared(T_mi_error, noise_model, loss_func);
                    T_mi_prior_cost_terms.emplace_back(T_mi_prior_factor);
                }
            }
        }

        ///################################################################################

        // Step 35: Configure voxel visitation settings
        // Determine how many neighboring voxels to visit along each axis
        const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1; // More neighbors for early frames
        const int kMinNumNeighbors = options_.min_number_neighbors; // Minimum neighbors for point-to-plane alignment

        auto &current_estimate = trajectory_.at(index_frame);

        // Step 36: Cache interpolation matrices for unique keypoint timestamps
        // interp_mats_ stores matrices (omega, lambda) for efficient pose interpolation
        timer[0].second->start(); // Start update transform timer
        interp_mats_.clear(); // Clear previous interpolation matrices
        const double time1 = prev_slam_time.seconds(); // Start time of the trajectory segment
        const double time2 = KNOT_TIMES.back(); // End time of the trajectory segment
        const double T = time2 - time1; // Time duration

        const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones(); // Unit vector for covariance
        const Eigen::Matrix<double, 18, 18> Qinv_T = SLAM_TRAJ->getQinvPublic(T, ones); // Inverse covariance matrix
        const Eigen::Matrix<double, 18, 18> Tran_T = SLAM_TRAJ->getTranPublic(T); // Transition matrix

        // Compute interpolation matrices
        if (unique_point_times.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential: Process timestamps one by one for small sizes
            for (size_t i = 0; i < unique_point_times.size(); ++i) {
                const double time = unique_point_times[i];
                const double tau = time - time1; // Time offset from start
                const double kappa = time2 - time; // Time offset from end
                const Matrix18d Q_tau = SLAM_TRAJ->getQPublic(tau, ones); // Covariance at tau
                const Matrix18d Tran_kappa = SLAM_TRAJ->getTranPublic(kappa); // Transition at kappa
                const Matrix18d Tran_tau = SLAM_TRAJ->getTranPublic(tau); // Transition at tau
                const Matrix18d omega = Q_tau * Tran_kappa.transpose() * Qinv_T; // Interpolation matrix
                const Matrix18d lambda = Tran_tau - omega * Tran_T; // Interpolation matrix
                interp_mats_.emplace(time, std::make_pair(omega, lambda)); // Cache matrices
            }
        } else {
            // Parallel: Process timestamps concurrently with TBB
            tbb::concurrent_hash_map<double, std::pair<Matrix18d, Matrix18d>> temp_interp_mats;
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const double time = unique_point_times[i];
                        const double tau = time - time1; // Time offset from start
                        const double kappa = time2 - time; // Time offset from end
                        const Matrix18d Q_tau = SLAM_TRAJ->getQPublic(tau, ones); // Covariance at tau
                        const Matrix18d Tran_kappa = SLAM_TRAJ->getTranPublic(kappa); // Transition at kappa
                        const Matrix18d Tran_tau = SLAM_TRAJ->getTranPublic(tau); // Transition at tau
                        const Matrix18d omega = Q_tau * Tran_kappa.transpose() * Qinv_T; // Interpolation matrix
                        const Matrix18d lambda = Tran_tau - omega * Tran_T; // Interpolation matrix
                        tbb::concurrent_hash_map<double, std::pair<Matrix18d, Matrix18d>>::accessor acc;
                        temp_interp_mats.insert(acc, time);
                        acc->second = std::make_pair(omega, lambda); // Thread-safe insertion
                    }
                });

            // Transfer from concurrent_hash_map to interp_mats_ sequentially
            for (const auto& entry : temp_interp_mats) {
                interp_mats_.emplace(entry.first, entry.second);
            }
        }
        timer[0].second->stop(); // Stop update transform timer

        // Step 37: Transform keypoints to the map frame using interpolated poses
        // Lambda function to map raw keypoints to the map frame
        auto transform_keypoints = [&]() {
            // Get state variables at the start and end knots
            const auto knot1 = SLAM_TRAJ->get(prev_slam_time);
            const auto knot2 = SLAM_TRAJ->get(finalicp::traj::Time(KNOT_TIMES.back()));
            const auto T1 = knot1->pose()->value(); // Start pose
            const auto w1 = knot1->velocity()->value(); // Start velocity
            const auto dw1 = knot1->acceleration()->value(); // Start acceleration
            const auto T2 = knot2->pose()->value(); // End pose
            const auto w2 = knot2->velocity()->value(); // End velocity
            const auto dw2 = knot2->acceleration()->value(); // End acceleration

            // Compute relative transformation and Jacobians
            const auto xi_21 = (T2 / T1).vec(); // Relative pose vector
            const math::se3::Transformation T_21(xi_21); // Relative transformation
            const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21); // Inverse Jacobian
            const auto J_21_inv_w2 = J_21_inv * w2; // Transformed velocity
            const auto J_21_inv_curl_dw2 = (-0.5 * math::se3::curlyhat(J_21_inv_w2) * w2 + J_21_inv * dw2); // Transformed acceleration

            // Step 37.1: Cache interpolated poses for unique timestamps
            // Computes and stores pose matrices (T_mr) for each timestamp in unique_point_times
            std::map<double, Eigen::Matrix4d> T_mr_cache_map;

            if (unique_point_times.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // Sequential: Process timestamps one by one for small sizes
                for (size_t jj = 0; jj < unique_point_times.size(); ++jj) {
                    const double ts = unique_point_times[jj];
                    const auto& omega = interp_mats_.at(ts).first;
                    const auto& lambda = interp_mats_.at(ts).second;
                    // Compute interpolated pose vector
                    const Eigen::Matrix<double, 6, 1> xi_i1 =
                        lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
                        omega.block<6, 6>(0, 0) * xi_21 + omega.block<6, 6>(0, 6) * J_21_inv_w2 +
                        omega.block<6, 6>(0, 12) * J_21_inv_curl_dw2;

                    const math::se3::Transformation T_i1(xi_i1); // Interpolated pose relative to T1
                    const math::se3::Transformation T_i0 = T_i1 * T1; // Pose in map frame
                    const Eigen::Matrix4d T_mr = T_i0.inverse().matrix(); // Inverse pose matrix
                    T_mr_cache_map[ts] = T_mr; // Cache pose
                }
            } else {
                // Parallel: Process timestamps concurrently with TBB
                tbb::concurrent_hash_map<double, Eigen::Matrix4d> temp_cache_map;
                tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t jj = range.begin(); jj != range.end(); ++jj) {
                            const double ts = unique_point_times[jj];
                            const auto& omega = interp_mats_.at(ts).first;
                            const auto& lambda = interp_mats_.at(ts).second;
                            // Compute interpolated pose vector
                            const Eigen::Matrix<double, 6, 1> xi_i1 =
                                lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
                                omega.block<6, 6>(0, 0) * xi_21 + omega.block<6, 6>(0, 6) * J_21_inv_w2 +
                                omega.block<6, 6>(0, 12) * J_21_inv_curl_dw2;
                            const math::se3::Transformation T_i1(xi_i1); // Interpolated pose relative to T1
                            const math::se3::Transformation T_i0 = T_i1 * T1; // Pose in map frame
                            const Eigen::Matrix4d T_mr = T_i0.inverse().matrix(); // Inverse pose matrix
                            tbb::concurrent_hash_map<double, Eigen::Matrix4d>::accessor acc;
                            temp_cache_map.insert(acc, ts);
                            acc->second = T_mr; // Thread-safe insertion
                        }
                    });

                // Transfer from concurrent_hash_map to T_mr_cache_map sequentially
                for (const auto& entry : temp_cache_map) {
                    T_mr_cache_map[entry.first] = entry.second;
                }
            }

            // Step 37.2: Transform keypoints to the map frame
            // Applies cached pose matrices (T_mr) to transform raw keypoint coordinates to the map frame
            if (keypoints.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // Sequential: Transform keypoints one by one for small sizes
                for (size_t jj = 0; jj < keypoints.size(); ++jj) {
                    auto& keypoint = keypoints[jj];
                    const Eigen::Matrix4d& T_mr = T_mr_cache_map.at(keypoint.timestamp);
                    keypoint.pt = T_mr.block<3, 3>(0, 0) * keypoint.raw_pt + T_mr.block<3, 1>(0, 3); // Transform raw point
                }
            } else {
                // Parallel: Transform keypoints concurrently with TBB
                tbb::concurrent_vector<Point3D> temp_keypoints(keypoints.size()); // Concurrent storage for transformed keypoints
                // Initialize temp_keypoints sequentially to preserve other keypoint fields
                for (size_t jj = 0; jj < keypoints.size(); ++jj) {
                    temp_keypoints[jj] = keypoints[jj];
                }
                tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t jj = range.begin(); jj != range.end(); ++jj) {
                            auto& keypoint = temp_keypoints[jj];
                            const Eigen::Matrix4d& T_mr = T_mr_cache_map.at(keypoint.timestamp);
                            keypoint.pt = T_mr.block<3, 3>(0, 0) * keypoint.raw_pt + T_mr.block<3, 1>(0, 3); // Transform raw point
                        }
                    });

                // Assign temporary keypoints back to keypoints sequentially
                for (size_t jj = 0; jj < temp_keypoints.size(); ++jj) {
                    keypoints[jj] = temp_keypoints[jj];
                }
            }
        };

        #define USE_P2P_SUPER_COST_TERM true

        // Step 38: Initialize point-to-plane super cost term
        // Constrains the trajectory to align keypoints with the map
        finalicp::P2PSuperCostTerm::Options p2p_options;
        p2p_options.num_threads = options_.num_threads; // Thread count (sequential here, but set for compatibility)
        p2p_options.p2p_loss_sigma = options_.p2p_loss_sigma; // Loss parameter for point-to-plane
        // Map loss function to enum
        switch (options_.p2p_loss_func) {
            case stateestimate::lidarinertialodom::LOSS_FUNC::L2:
                p2p_options.p2p_loss_func = finalicp::P2PSuperCostTerm::LOSS_FUNC::L2;
                break;
            case stateestimate::lidarinertialodom::LOSS_FUNC::DCS:
                p2p_options.p2p_loss_func = finalicp::P2PSuperCostTerm::LOSS_FUNC::DCS;
                break;
            case stateestimate::lidarinertialodom::LOSS_FUNC::CAUCHY:
                p2p_options.p2p_loss_func = finalicp::P2PSuperCostTerm::LOSS_FUNC::CAUCHY;
                break;
            case stateestimate::lidarinertialodom::LOSS_FUNC::GM:
                p2p_options.p2p_loss_func = finalicp::P2PSuperCostTerm::LOSS_FUNC::GM;
                break;
            default:
                p2p_options.p2p_loss_func = finalicp::P2PSuperCostTerm::LOSS_FUNC::L2;
        }
        const auto p2p_super_cost_term = finalicp::P2PSuperCostTerm::MakeShared(SLAM_TRAJ, prev_slam_time, finalicp::traj::Time(KNOT_TIMES.back()), p2p_options);

        // Step 39: Stop the initialization timer
        if (!timer.empty()) timer[4].second->stop();

        ///################################################################################

        // Step 40: Transform keypoints to the robot frame (if using point-to-plane super cost term)
        // Applies the inverse sensor-to-robot transformation (T_rs) to raw keypoint coordinates
        #if USE_P2P_SUPER_COST_TERM
            timer[0].second->start(); // Start update transform timer
            const Eigen::Matrix4d T_rs_mat = options_.T_sr.inverse(); // Inverse sensor-to-robot transformation

            if (keypoints.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // Sequential: Transform keypoints one by one for small sizes
                for (size_t i = 0; i < keypoints.size(); ++i) {
                    auto& keypoint = keypoints[i];
                    keypoint.raw_pt = T_rs_mat.block<3, 3>(0, 0) * keypoint.raw_pt + T_rs_mat.block<3, 1>(0, 3); // Transform raw point
                }
            } else {
                // Parallel: Transform keypoints concurrently with TBB
                tbb::concurrent_vector<Point3D> temp_keypoints(keypoints.size()); // Concurrent storage for transformed keypoints
                // Initialize temp_keypoints sequentially to preserve other keypoint fields
                for (size_t i = 0; i < keypoints.size(); ++i) {
                    temp_keypoints[i] = keypoints[i];
                }
                tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            auto& keypoint = temp_keypoints[i];
                            keypoint.raw_pt = T_rs_mat.block<3, 3>(0, 0) * keypoint.raw_pt + T_rs_mat.block<3, 1>(0, 3); // Transform raw point
                        }
                    });

                // Assign temporary keypoints back to keypoints sequentially
                keypoints.resize(temp_keypoints.size());
                for (size_t i = 0; i < temp_keypoints.size(); ++i) {
                    keypoints[i] = temp_keypoints[i];
                }
            }
            timer[0].second->stop(); // Stop update transform timer
        #endif

        // Step 41: Initialize the current frame’s pose estimate
        // Computes begin and end poses, velocities, and accelerations for the frame
        auto& p2p_matches = p2p_super_cost_term->get(); // Get point-to-plane matches
        p2p_matches.clear(); // Clear previous matches
        int N_matches = 0; // Track number of matches
        p2p_matches.reserve(keypoints.size()); // Preallocate for efficiency

        // Compute begin pose (at frame’s start timestamp)
        finalicp::traj::Time curr_begin_slam_time(static_cast<double>(trajectory_[index_frame].begin_timestamp));
        const Eigen::Matrix4d begin_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_begin_slam_time)->value().inverse().matrix();
        const Eigen::Matrix4d begin_T_ms = begin_T_mr * options_.T_sr.inverse();
        current_estimate.begin_t = begin_T_ms.block<3, 1>(0, 3); // Begin translation
        current_estimate.begin_R = begin_T_ms.block<3, 3>(0, 0); // Begin rotation

        // Compute end pose (at frame’s end timestamp)
        finalicp::traj::Time curr_end_slam_time(static_cast<double>(trajectory_[index_frame].end_timestamp));
        const Eigen::Matrix4d end_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_end_slam_time)->value().inverse().matrix();
        const Eigen::Matrix4d end_T_ms = end_T_mr * options_.T_sr.inverse();
        current_estimate.end_t = end_T_ms.block<3, 1>(0, 3); // End translation
        current_estimate.end_R = end_T_ms.block<3, 3>(0, 0); // End rotation

        // Compute velocities and accelerations
        Eigen::Matrix<double, 6, 1> v_begin = SLAM_TRAJ->getVelocityInterpolator(curr_begin_slam_time)->value();
        Eigen::Matrix<double, 6, 1> v_end = SLAM_TRAJ->getVelocityInterpolator(curr_end_slam_time)->value();
        Eigen::Matrix<double, 6, 1> a_begin = SLAM_TRAJ->getAccelerationInterpolator(curr_begin_slam_time)->value();
        Eigen::Matrix<double, 6, 1> a_end = SLAM_TRAJ->getAccelerationInterpolator(curr_end_slam_time)->value();

        // Step 42: Configure sliding window filter usage
        // Determines whether to use SlidingWindowFilter inside ICP loop
        bool swf_inside_icp = true; // Default for KITTI-raw: false, but true here
        if (index_frame > options_.init_num_frames) {
            swf_inside_icp = true; // Use sliding window filter after initial frames
        }

        ///################################################################################

        // Step 43: Start ICP optimization loop
        // Iterates to refine the trajectory using point-to-plane alignment
        for (int iter = 0; iter < options_.num_iters_icp; iter++) {
            // Initialize optimization problem based on swf_inside_icp
            const auto problem = [&]() -> finalicp::Problem::Ptr {
                if (swf_inside_icp) {
                    // Use SlidingWindowFilter for sliding window optimization
                    return std::make_shared<finalicp::SlidingWindowFilter>(*sliding_window_filter_);
                } else {
                    // Use OptimizationProblem for full state optimization
                    auto problem = finalicp::OptimizationProblem::MakeShared(options_.num_threads);
                    for (const auto& var : SLAM_STATE_VAR) {
                        problem->addStateVariable(var);
                    }
                    return problem;
                }
            }();

            // Add prior cost terms to the problem
            SLAM_TRAJ->addPriorCostTerms(*problem);
            for (const auto& prior_cost_term : prior_cost_terms) {
                problem->addCostTerm(prior_cost_term);
            }

            // Step 44: Clear measurement cost terms and prepare for association
            timer[1].second->start(); // Start association timer
            meas_cost_terms.clear(); // Clear previous measurement cost terms
            p2p_matches.clear(); // Clear previous point-to-plane matches

            #if USE_P2P_SUPER_COST_TERM
                p2p_matches.reserve(keypoints.size()); // Reserve for new matches
            #else
                meas_cost_terms.reserve(keypoints.size()); // Reserve for new cost terms
            #endif

        ///################################################################################

            if (keypoints.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // Sequential: Process keypoints one by one for small sizes
                for (size_t i = 0; i < keypoints.size(); i++) {
                    const auto& keypoint = keypoints[i];
                    const auto& pt_keypoint = keypoint.pt;

                    // Search for neighboring points in the map
                    ArrayVector3d vector_neighbors = map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

                    // Skip if insufficient neighbors
                    if (vector_neighbors.size() < static_cast<size_t>(kMinNumNeighbors)) {
                        continue;
                    }

                    // Compute neighborhood distribution (normals, planarity)
                    auto neighborhood = compute_neighborhood_distribution(vector_neighbors, options_.num_threads, options_.sequential_threshold);

                    // Compute planarity weight and distance to plane
                    const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
                    const double weight = planarity_weight;
                    const double dist_to_plane = std::abs((pt_keypoint - vector_neighbors[0]).transpose() * neighborhood.normal);
                    const bool use_p2p = dist_to_plane < options_.p2p_max_dist;

                    if (use_p2p) {
                        #if USE_P2P_SUPER_COST_TERM
                            // Create point-to-plane match
                            Eigen::Vector3d closest_pt = vector_neighbors[0];
                            Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                            p2p_matches.emplace_back(finalicp::P2PMatch(keypoint.timestamp, closest_pt, closest_normal, keypoint.raw_pt));
                        #else
                            // Create point-to-plane cost term
                            Eigen::Vector3d closest_pt = vector_neighbors[0];
                            Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                            Eigen::Matrix3d W = closest_normal * closest_normal.transpose() + 1e-5 * Eigen::Matrix3d::Identity();
                            const auto noise_model = finalicp::StaticNoiseModel<3>::MakeShared(W, finalicp::NoiseType::INFORMATION);
                            const auto& T_mr_intp_eval = T_mr_intp_eval_map.at(keypoint.timestamp);
                            const auto error_func = finalicp::p2p::p2pError(T_mr_intp_eval, closest_pt, keypoint.raw_pt);
                            error_func->setTime(finalicp::traj::Time(keypoint.timestamp));

                            // Select loss function based on options
                            const auto loss_func = [this]() -> finalicp::BaseLossFunc::Ptr {
                                switch (options_.p2p_loss_func) {
                                    case stateestimate::lidarinertialodom::LOSS_FUNC::L2:
                                        return finalicp::L2LossFunc::MakeShared();
                                    case stateestimate::lidarinertialodom::LOSS_FUNC::DCS:
                                        return finalicp::DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
                                    case stateestimate::lidarinertialodom::LOSS_FUNC::CAUCHY:
                                        return finalicp::CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
                                    case stateestimate::lidarinertialodom::LOSS_FUNC::GM:
                                        return finalicp::GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
                                    default:
                                        return nullptr;
                                }
                            }();

                            const auto cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, loss_func);
                            meas_cost_terms.emplace_back(cost);
                        #endif
                    }
                }
            } else {
                // Parallel: Process keypoints concurrently with TBB
                tbb::concurrent_vector<finalicp::P2PMatch> temp_p2p_matches; // Initialize empty vector
                #if !USE_P2P_SUPER_COST_TERM
                    tbb::concurrent_vector<finalicp::BaseCostTerm::ConstPtr> temp_meas_cost_terms; // Initialize empty vector
                #endif

                tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            const auto& keypoint = keypoints[i];
                            const auto& pt_keypoint = keypoint.pt;

                            // Search for neighboring points in the map
                            ArrayVector3d vector_neighbors = map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

                            // Skip if insufficient neighbors
                            if (vector_neighbors.size() < static_cast<size_t>(kMinNumNeighbors)) {
                                continue;
                            }

                            // Compute neighborhood distribution (normals, planarity)
                            auto neighborhood = compute_neighborhood_distribution(vector_neighbors, options_.num_threads, options_.sequential_threshold);

                            // Compute planarity weight and distance to plane
                            const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
                            const double weight = planarity_weight;
                            const double dist_to_plane = std::abs((pt_keypoint - vector_neighbors[0]).transpose() * neighborhood.normal);
                            const bool use_p2p = dist_to_plane < options_.p2p_max_dist;

                            if (use_p2p) {
                                #if USE_P2P_SUPER_COST_TERM
                                    // Create point-to-plane match
                                    Eigen::Vector3d closest_pt = vector_neighbors[0];
                                    Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                                    temp_p2p_matches.push_back(finalicp::P2PMatch(keypoint.timestamp, closest_pt, closest_normal, keypoint.raw_pt));
                                #else
                                    // Create point-to-plane cost term
                                    Eigen::Vector3d closest_pt = vector_neighbors[0];
                                    Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                                    Eigen::Matrix3d W = closest_normal * closest_normal.transpose() + 1e-5 * Eigen::Matrix3d::Identity();
                                    const auto noise_model = finalicp::StaticNoiseModel<3>::MakeShared(W, finalicp::NoiseType::INFORMATION);
                                    if (T_mr_intp_eval_map.find(keypoint.timestamp) == T_mr_intp_eval_map.end()) {
                                        throw std::runtime_error("Missing T_mr_intp_eval_map entry for timestamp: " + std::to_string(keypoint.timestamp));
                                    }
                                    const auto& T_mr_intp_eval = T_mr_intp_eval_map.at(keypoint.timestamp);
                                    const auto error_func = finalicp::p2p::p2pError(T_mr_intp_eval, closest_pt, keypoint.raw_pt);
                                    error_func->setTime(finalicp::traj::Time(keypoint.timestamp));

                                    // Select loss function based on options
                                    const auto loss_func = [this]() -> finalicp::BaseLossFunc::Ptr {
                                        switch (options_.p2p_loss_func) {
                                            case stateestimate::lidarinertialodom::LOSS_FUNC::L2:
                                                return finalicp::L2LossFunc::MakeShared();
                                            case stateestimate::lidarinertialodom::LOSS_FUNC::DCS:
                                                return finalicp::DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
                                            case stateestimate::lidarinertialodom::LOSS_FUNC::CAUCHY:
                                                return finalicp::CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
                                            case stateestimate::lidarinertialodom::LOSS_FUNC::GM:
                                                return finalicp::GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
                                            default:
                                                return nullptr;
                                        }
                                    }();

                                    const auto cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, loss_func);
                                    temp_meas_cost_terms.push_back(cost);
                                #endif
                            }
                        }
                    });

                // Transfer from concurrent vectors to p2p_matches and meas_cost_terms sequentially
                p2p_matches.assign(temp_p2p_matches.begin(), temp_p2p_matches.end());
                #if !USE_P2P_SUPER_COST_TERM
                    meas_cost_terms.assign(temp_meas_cost_terms.begin(), temp_meas_cost_terms.end());
                #endif
            }

        ///################################################################################

            // Step 45: Add cost terms to the optimization problem
            // Sets the number of matches and adds all cost terms for STEAM optimization
            #if USE_P2P_SUPER_COST_TERM
                N_matches = p2p_matches.size();
            #else
                N_matches = meas_cost_terms.size();
            #endif

            p2p_super_cost_term->initP2PMatches(); // Initialize point-to-plane matches
            for (const auto& cost : meas_cost_terms) {
                problem->addCostTerm(cost); // Add point-to-plane cost terms
            }
            for (const auto& cost : imu_cost_terms) {
                problem->addCostTerm(cost); // Add IMU cost terms
            }
            for (const auto& cost : pose_meas_cost_terms) {
                problem->addCostTerm(cost); // Add pose measurement cost terms
            }
            for (const auto& cost : imu_prior_cost_terms) {
                problem->addCostTerm(cost); // Add IMU bias prior cost terms
            }
            for (const auto& cost : T_mi_prior_cost_terms) {
                problem->addCostTerm(cost); // Add T_mi prior cost terms
            }
            problem->addCostTerm(p2p_super_cost_term); // Add point-to-plane super cost term
            if (options_.use_imu) {
                problem->addCostTerm(imu_super_cost_term); // Add IMU super cost term
            }

            timer[1].second->stop(); // Stop association timer

            // Step 46: Check for sufficient keypoints
            // Ensures enough matches for reliable optimization
            if (N_matches < options_.min_number_keypoints) {
                // LOG(ERROR) << "[ICP]Error : not enough keypoints selected in ct-icp !" << std::endl;
                // LOG(ERROR) << "[ICP]Number_of_residuals : " << N_matches << std::endl;
                icp_success = false;
                break; // Exit the ICP loop if insufficient keypoints
            }

            // Step 47: Solve the optimization problem
            // Uses Gauss-Newton solver to refine the trajectory
            timer[2].second->start(); // Start optimization timer
            finalicp::GaussNewtonSolverNVA::Params params;
            params.verbose = options_.verbose;
            params.max_iterations = static_cast<unsigned int>(options_.max_iterations);
            params.line_search = (iter >= 2 && options_.use_line_search); // Enable line search after 2 iterations if configured
            params.reuse_previous_pattern = !swf_inside_icp; // Disable pattern reuse for sliding window filter
            finalicp::GaussNewtonSolverNVA solver(*problem, params);
            solver.optimize(); // Solve the optimization problem
            timer[2].second->stop(); // Stop optimization timer

            // Step 48: Update the trajectory estimate and check convergence
            // Computes differences in pose, velocity, and acceleration to determine if converged
            timer[3].second->start(); // Start alignment timer
            double diff_trans = 0.0, diff_rot = 0.0, diff_vel = 0.0, diff_acc = 0.0;

            // Update begin pose
            finalicp::traj::Time curr_begin_slam_time(static_cast<double>(trajectory_[index_frame].begin_timestamp));
            const Eigen::Matrix4d begin_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_begin_slam_time)->value().inverse().matrix();
            const Eigen::Matrix4d begin_T_ms = begin_T_mr * options_.T_sr.inverse();
            diff_trans += (current_estimate.begin_t - begin_T_ms.block<3, 1>(0, 3)).norm();
            diff_rot += AngularDistance(current_estimate.begin_R, begin_T_ms.block<3, 3>(0, 0));

            // Update end pose
            finalicp::traj::Time curr_end_slam_time(static_cast<double>(trajectory_[index_frame].end_timestamp));
            const Eigen::Matrix4d end_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_end_slam_time)->value().inverse().matrix();
            const Eigen::Matrix4d end_T_ms = end_T_mr * options_.T_sr.inverse();
            diff_trans += (current_estimate.end_t - end_T_ms.block<3, 1>(0, 3)).norm();
            diff_rot += AngularDistance(current_estimate.end_R, end_T_ms.block<3, 3>(0, 0));

            // Update velocities
            const auto vb = SLAM_TRAJ->getVelocityInterpolator(curr_begin_slam_time)->value();
            const auto ve = SLAM_TRAJ->getVelocityInterpolator(curr_end_slam_time)->value();
            diff_vel += (vb - v_begin).norm();
            diff_vel += (ve - v_end).norm();
            v_begin = vb;
            v_end = ve;

            // Update accelerations
            const auto ab = SLAM_TRAJ->getAccelerationInterpolator(curr_begin_slam_time)->value();
            const auto ae = SLAM_TRAJ->getAccelerationInterpolator(curr_end_slam_time)->value();
            diff_acc += (ab - a_begin).norm();
            diff_acc += (ae - a_end).norm();
            a_begin = ab;
            a_end = ae;

            // Update mid pose
            finalicp::traj::Time curr_mid_slam_time(static_cast<double>(trajectory_[index_frame].getEvalTime()));
            const Eigen::Matrix4d mid_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_mid_slam_time)->value().inverse().matrix();
            const Eigen::Matrix4d mid_T_ms = mid_T_mr * options_.T_sr.inverse();
            current_estimate.setMidPose(mid_T_ms);

            // Update current estimate
            current_estimate.begin_R = begin_T_ms.block<3, 3>(0, 0);
            current_estimate.begin_t = begin_T_ms.block<3, 1>(0, 3);
            current_estimate.end_R = end_T_ms.block<3, 3>(0, 0);
            current_estimate.end_t = end_T_ms.block<3, 1>(0, 3);

            // Update IMU biases (if enabled)
            if (options_.use_imu) {
                size_t i = prev_trajectory_var_index;
                for (; i < trajectory_vars_.size() - 1; i++) {
                    if (curr_mid_slam_time.seconds() >= trajectory_vars_[i].time.seconds() &&
                        curr_mid_slam_time.seconds() < trajectory_vars_[i + 1].time.seconds()) {
                        break;
                    }
                }
                if (curr_mid_slam_time.seconds() < trajectory_vars_[i].time.seconds() ||
                    curr_mid_slam_time.seconds() >= trajectory_vars_[i + 1].time.seconds()) {
                    throw std::runtime_error("Mid time not within knot times in icp");
                }
                current_estimate.mid_b = trajectory_vars_[i].imu_biases->value();
            }

            // LOG(INFO) << "diff_rot: " << diff_rot << " diff_trans: " << diff_trans << " diff_vel: " << diff_vel << " diff_acc: " << diff_acc << std::endl;

            // Check convergence
            if (index_frame > 1 &&
                diff_rot < options_.threshold_orientation_norm &&
                diff_trans < options_.threshold_translation_norm &&
                diff_vel < (options_.threshold_translation_norm * 10.0 + options_.threshold_orientation_norm * 10.0) &&
                diff_acc < (options_.threshold_translation_norm * 100.0 + options_.threshold_orientation_norm * 100.0)) {
                if (options_.debug_print) {
                    // LOG(INFO) << "ICP: Finished with N=" << iter << " ICP iterations" << std::endl;
                }
                if (options_.break_icp_early) {
                    break; // Exit loop if converged and early breaking is enabled
                }
            }

            // Re-transform keypoints for the next iteration
            timer[0].second->start(); // Start update transform timer
            transform_keypoints(); // Updates keypoints.pt using the latest trajectory
            timer[0].second->stop(); // Stop update transform timer

            timer[3].second->stop(); // Stop alignment timer
        } // End ICP optimization loop
        
        ///################################################################################

        // Step 49: Add cost terms to the sliding window filter
        // Includes state priors, point-to-plane, IMU, pose, and T_mi cost terms
        SLAM_TRAJ->addPriorCostTerms(*sliding_window_filter_); // Add state priors (e.g., for initial state x_0)
        for (const auto& prior_cost_term : prior_cost_terms) {
            sliding_window_filter_->addCostTerm(prior_cost_term); // Add prior cost terms
        }
        for (const auto& meas_cost_term : meas_cost_terms) {
            sliding_window_filter_->addCostTerm(meas_cost_term); // Add point-to-plane cost terms
        }
        for (const auto& pose_cost : pose_meas_cost_terms) {
            sliding_window_filter_->addCostTerm(pose_cost); // Add pose measurement cost terms
        }
        for (const auto& imu_cost : imu_cost_terms) {
            sliding_window_filter_->addCostTerm(imu_cost); // Add IMU cost terms
        }
        for (const auto& imu_prior_cost : imu_prior_cost_terms) {
            sliding_window_filter_->addCostTerm(imu_prior_cost); // Add IMU bias prior cost terms
        }
        for (const auto& T_mi_prior_cost : T_mi_prior_cost_terms) {
            sliding_window_filter_->addCostTerm(T_mi_prior_cost); // Add T_mi prior cost terms
        }
        sliding_window_filter_->addCostTerm(p2p_super_cost_term); // Add point-to-plane super cost term
        if (options_.use_imu) {
            sliding_window_filter_->addCostTerm(imu_super_cost_term); // Add IMU super cost term
        }

        // LOG(INFO) << "number of variables: " << sliding_window_filter_->getNumberOfVariables() << std::endl;
        // LOG(INFO) << "number of cost terms: " << sliding_window_filter_->getNumberOfCostTerms() << std::endl;

        // Step 50: Validate and optimize the sliding window filter
        // Checks variable and cost term counts, then solves the optimization problem
        if (sliding_window_filter_->getNumberOfVariables() > 100) {
            throw std::runtime_error("Too many variables in the sliding window filter: " +
                                    std::to_string(sliding_window_filter_->getNumberOfVariables()));
        }
        if (sliding_window_filter_->getNumberOfCostTerms() > 100000) {
            throw std::runtime_error("Too many cost terms in the sliding window filter: " +
                                    std::to_string(sliding_window_filter_->getNumberOfCostTerms()));
        }

        finalicp::GaussNewtonSolverNVA::Params params;
        params.verbose = options_.verbose;
        params.max_iterations = static_cast<unsigned int>(options_.max_iterations);
        finalicp::GaussNewtonSolverNVA solver(*sliding_window_filter_, params);
        if (!swf_inside_icp) {
            solver.optimize(); // Optimize the sliding window filter if not done in ICP loop
        }

        // Step 51: Lock T_mi variables (if applicable)
        // Ensures consistent IMU-to-map transformations for future variables
        if (options_.T_mi_init_only && !use_T_mi_gt) {
            size_t i = prev_trajectory_var_index + 1;
            const auto prev_T_mi_value = prev_T_mi_var->value();
            for (; i < trajectory_vars_.size(); i++) {
                trajectory_vars_[i].T_mi = finalicp::se3::SE3StateVar::MakeShared(prev_T_mi_value);
                trajectory_vars_[i].T_mi->locked() = true; // Lock T_mi to prevent optimization
            }
        }

        // Step 52: Update the current estimate
        // Computes poses, velocities, accelerations, and covariance for begin, mid, and end timestamps
        const Eigen::Matrix4d curr_begin_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_begin_slam_time)->value().inverse().matrix();
        const Eigen::Matrix4d curr_begin_T_ms = curr_begin_T_mr * options_.T_sr.inverse();

        const Eigen::Matrix4d curr_end_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_end_slam_time)->value().inverse().matrix();
        const Eigen::Matrix4d curr_end_T_ms = curr_end_T_mr * options_.T_sr.inverse();

        finalicp::traj::Time curr_mid_slam_time(static_cast<double>(trajectory_[index_frame].getEvalTime()));
        const Eigen::Matrix4d mid_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_mid_slam_time)->value().inverse().matrix();
        const Eigen::Matrix4d mid_T_ms = mid_T_mr * options_.T_sr.inverse();
        current_estimate.setMidPose(mid_T_ms);

        // Update debug fields (for plotting)
        current_estimate.mid_w = SLAM_TRAJ->getVelocityInterpolator(curr_mid_slam_time)->value();
        current_estimate.mid_dw = SLAM_TRAJ->getAccelerationInterpolator(curr_mid_slam_time)->value();
        current_estimate.mid_T_mi = trajectory_vars_[prev_trajectory_var_index].T_mi->value().matrix();
        finalicp::Covariance covariance(solver);
        current_estimate.mid_state_cov = SLAM_TRAJ->getCovariance(covariance, trajectory_vars_[prev_trajectory_var_index].time);

        // Update begin and end poses
        current_estimate.begin_R = curr_begin_T_ms.block<3, 3>(0, 0);
        current_estimate.begin_t = curr_begin_T_ms.block<3, 1>(0, 3);
        current_estimate.end_R = curr_end_T_ms.block<3, 3>(0, 0);
        current_estimate.end_t = curr_end_T_ms.block<3, 1>(0, 3);

        ///################################################################################

        // Step 53: Update IMU biases (if enabled)
        // Interpolates IMU biases at the frame’s midpoint timestamp
        if (options_.use_imu) {
            size_t i = prev_trajectory_var_index;
            for (; i < trajectory_vars_.size() - 1; i++) {
                if (curr_mid_slam_time.seconds() >= trajectory_vars_[i].time.seconds() &&
                    curr_mid_slam_time.seconds() < trajectory_vars_[i + 1].time.seconds()) {
                    break;
                }
            }
            if (i >= trajectory_vars_.size() - 1 ||
                curr_mid_slam_time.seconds() < trajectory_vars_[i].time.seconds() ||
                curr_mid_slam_time.seconds() >= trajectory_vars_[i + 1].time.seconds()) {
                throw std::runtime_error("Mid time not within knot times in icp: " + std::to_string(curr_mid_slam_time.seconds()));
            }

            const auto bias_intp_eval = finalicp::vspace::VSpaceInterpolator<6>::MakeShared(curr_mid_slam_time, trajectory_vars_[i].imu_biases, trajectory_vars_[i].time, trajectory_vars_[i + 1].imu_biases, trajectory_vars_[i + 1].time);
            current_estimate.mid_b = bias_intp_eval->value();
            // LOG(INFO) << "mid_T_mi: " << current_estimate.mid_T_mi << std::endl;
            // LOG(INFO) << "b_begin: " << trajectory_vars_[i].imu_biases->value().transpose() << std::endl;
            // LOG(INFO) << "b_end: " << trajectory_vars_[i + 1].imu_biases->value().transpose() << std::endl;
        }

        // Step 54: Validate final estimate parameters
        // Ensures keypoints, velocities, and accelerations are valid
        // LOG(INFO) << "Number of keypoints used in CT-ICP : " << N_matches << std::endl;
        // LOG(INFO) << "v_begin: " << v_begin.transpose() << std::endl;
        // LOG(INFO) << "v_end: " << v_end.transpose() << std::endl;
        // LOG(INFO) << "a_begin: " << a_begin.transpose() << std::endl;
        // LOG(INFO) << "a_end: " << a_end.transpose() << std::endl;

        if (options_.debug_print) {
            for (size_t i = 0; i < timer.size(); i++) {LOG(INFO) << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;}
            // LOG(INFO) << "Number iterations CT-ICP : " << options_.num_iters_icp << std::endl;
            // LOG(INFO) << "Translation Begin: " << trajectory_[index_frame].begin_t.transpose() << std::endl;
            // LOG(INFO) << "Translation End: " << trajectory_[index_frame].end_t.transpose() << std::endl;
        }

        // Step 55: Return success status
        // Completes the icp function, returning whether the frame was successfully processed
        return icp_success;
    }
}  // namespace stateestimate