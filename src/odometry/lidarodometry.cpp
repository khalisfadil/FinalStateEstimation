#include <odometry/lidarodometry.hpp>

#include <iomanip>
#include <random>

namespace  stateestimate{
    // ########################################################################
    // AngularDistance
    // ########################################################################

    inline double lidarodom::AngularDistance(const Eigen::Matrix3d &rota, const Eigen::Matrix3d &rotb) {
        double d = 0.5 * ((rota * rotb.transpose()).trace() - 1);
        return std::acos(std::max(std::min(d, 1.0), -1.0)) * 180.0 / M_PI;
    }

    // ########################################################################
    // sub_sample_frame
    // ########################################################################

    void lidarodom::sub_sample_frame(std::vector<Point3D>& frame, double size_voxel, int sequential_threshold) {
        if (frame.empty()) return;
        using VoxelMap = tsl::robin_map<Voxel, Point3D, VoxelHash>;
        // Step 1: Build the downsampled voxel map
        VoxelMap voxel_map;
        build_voxel_map(frame, size_voxel, voxel_map, sequential_threshold);

        // Step 2: Rebuild the frame with the downsampled points
        frame.clear();
        frame.reserve(voxel_map.size());
        for (const auto& pair : voxel_map) {
            frame.push_back(pair.second);
        }
     }

    // ########################################################################
    // build_voxel_map
    // ########################################################################

    void lidarodom::build_voxel_map(const std::vector<Point3D>& frame, double size_voxel,
                                            tsl::robin_map<Voxel, Point3D, VoxelHash>& voxel_map,
                                            int sequential_threshold) {
        // Define the map type for clarity
        using LocalVoxelMap = tsl::robin_map<Voxel, Point3D, VoxelHash>;

        // 1. The main body of work for each thread (includes pre-allocation)
        auto reduction_body = [&](const tbb::blocked_range<size_t>& range, LocalVoxelMap local_map) -> LocalVoxelMap {
            // CRITICAL: Pre-allocate to prevent multiple costly re-hashes.
            local_map.reserve(range.size()); 
            for (size_t i = range.begin(); i != range.end(); ++i) {
                const auto& point = frame[i];
                Voxel voxel = Voxel::Coordinates(point.pt, size_voxel);
                local_map.insert({voxel, point});
            }
            return local_map;
        };

        // 2. How to combine two local maps (using insert, as merge is unavailable)
        auto combine_maps = [](LocalVoxelMap left, LocalVoxelMap right) -> LocalVoxelMap {
            // OPTIMIZATION: Always insert the smaller map's elements into the larger one.
            if (left.size() < right.size()) {
                std::swap(left, right);
            }

            // CORRECTED: Use range-insert as .merge() is not available for tsl::robin_map.
            // This will be less efficient than a true merge, but is the correct API call.
            left.insert(right.begin(), right.end());
            
            return left;
        };

        // 3. Perform the parallel reduction using the optimized lambdas
        LocalVoxelMap final_map = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, frame.size(), sequential_threshold),
            LocalVoxelMap(), // Identity element
            reduction_body,
            combine_maps
        );

        // Move the final result into the output parameter
        voxel_map = std::move(final_map);
    }

    // ########################################################################
    // build_voxel_map
    // ########################################################################

    void lidarodom::grid_sampling(const std::vector<Point3D>& frame, std::vector<Point3D>& keypoints,
                                     double size_voxel_subsampling, int sequential_threshold) {
#ifdef DEBUG
        // std::cout << "[GRID_SAMPLING] Starting grid_sampling. Input points: " << frame.size() << std::endl;
#endif
        // Step 1: Clear the output keypoints vector
        keypoints.clear();

        // Step 2: Create a temporary vector and copy in parallel
        std::vector<Point3D> frame_sub(frame.size()); // Pre-allocate to avoid resizing
        tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size(), sequential_threshold),
            [&](const tbb::blocked_range<size_t>& range) {
                std::copy(frame.begin() + range.begin(), frame.begin() + range.end(),
                        frame_sub.begin() + range.begin());
            });

        // Step 3: Apply voxel grid subsampling to frame_sub
        sub_sample_frame(frame_sub, size_voxel_subsampling, sequential_threshold);

        // Step 4: Move downsampled points to keypoints
        keypoints = std::move(frame_sub);

#ifdef DEBUG
        // std::cout << "[GRID_SAMPLING] Finished. Output keypoints: " << keypoints.size() << std::endl;
#endif
    }

    // ########################################################################
    // compute_neighborhood_distribution
    // ########################################################################

    // Assuming ArrayVector3d is std::vector<Eigen::Vector3d>
    // and Neighborhood is a struct with: Eigen::Vector3d center, normal; Eigen::Matrix3d covariance; double a2D;

    lidarodom::Neighborhood lidarodom::compute_neighborhood_distribution(
        const ArrayVector3d& points, int sequential_threshold) {
        
        Neighborhood neighborhood; // Default: center/normal=zero, covariance=identity, a2D=1.0
        const size_t point_count = points.size();

#ifdef DEBUG
        // [DEBUG] Log function entry and number of input points
        // std::cout << "[COMP_NEIGH] Processing neighborhood with " << point_count << " points." << std::endl;
#endif

        // --- Handle Edge Cases ---
        if (point_count < 2) {
#ifdef DEBUG
            // [DEBUG] Log when an edge case is triggered
            // std::cout << "[COMP_NEIGH] Edge case: point_count < 2. Returning default values." << std::endl;
#endif
            if (point_count == 1) {
                neighborhood.center = points[0];
            }
            // For 0 or 1 point, distribution is undefined.
            // Return a stable, default state.
            neighborhood.covariance.setZero();
            neighborhood.normal = Eigen::Vector3d::UnitZ(); // A reasonable default normal
            neighborhood.a2D = 0.0; // Distribution is perfectly linear (a point) or undefined (empty)
            return neighborhood;
        }

        // Limit the number of threads for the parallel section
        // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

        // --- Use a single-pass algorithm to compute sums for mean and covariance ---
        // This is more efficient than a two-pass approach.
        Eigen::Vector3d sum_of_points = Eigen::Vector3d::Zero();
        Eigen::Matrix3d sum_of_outer_products = Eigen::Matrix3d::Zero();

        if (point_count < static_cast<size_t>(sequential_threshold)) {
            // --- Sequential path for small point clouds ---
            for (const auto& point : points) {
                sum_of_points += point;
                sum_of_outer_products += point * point.transpose();
            }
        } else {
            // --- Parallel path for large point clouds using tbb::parallel_reduce ---
            struct Accumulator {
                Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                Eigen::Matrix3d outer = Eigen::Matrix3d::Zero();

                // In-place reduction operator for efficiency
                Accumulator& operator+=(const Accumulator& other) {
                    sum += other.sum;
                    outer += other.outer;
                    return *this;
                }
            };

            Accumulator total = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, point_count, sequential_threshold),
                Accumulator(), // Identity element
                [&](const tbb::blocked_range<size_t>& range, Accumulator acc) -> Accumulator {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        acc.sum += points[i];
                        acc.outer += points[i] * points[i].transpose();
                    }
                    return acc;
                },
                // The final reduction step
                [](Accumulator a, const Accumulator& b) -> Accumulator {
                    return a += b;
                }
            );
            sum_of_points = total.sum;
            sum_of_outer_products = total.outer;
        }

        // --- Finalize Mean and Covariance Calculation ---
        const double inv_point_count = 1.0 / static_cast<double>(point_count);
        neighborhood.center = sum_of_points * inv_point_count;
        
        // Covariance = E[x*x^T] - E[x]*E[x]^T
        neighborhood.covariance = (sum_of_outer_products * inv_point_count) - (neighborhood.center * neighborhood.center.transpose());

#ifdef DEBUG
        // [DEBUG] Check the computed covariance matrix for issues
        if (!neighborhood.covariance.allFinite()) {
            std::cout << "[COMP_NEIGH] CRITICAL: Covariance matrix is NOT finite!" << std::endl;
        }
        // std::cout << "[COMP_NEIGH] Covariance Matrix:\n" << neighborhood.covariance << std::endl;
#endif

        // --- Perform PCA via Eigen Decomposition to find the normal vector and planarity ---
        // The eigenvectors of the covariance matrix are the principal components (axes of variance).
        // The eigenvalues represent the magnitude of variance along those axes.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(neighborhood.covariance);

        // [DEBUG] Check if the Eigen solver was successful
        if (es.info() != Eigen::Success) {
#ifdef DEBUG
            std::cout << "[COMP_NEIGH] CRITICAL: Eigen decomposition failed!" << std::endl;
#endif
            // Handle failure: return a default neighborhood to avoid crashing
            neighborhood.covariance.setZero();
            neighborhood.normal = Eigen::Vector3d::UnitZ();
            neighborhood.a2D = 0.0;
            return neighborhood;
        }

        // The normal of a plane is the direction of least variance, which corresponds
        // to the eigenvector with the smallest eigenvalue. Eigen sorts them in increasing order.
        neighborhood.normal = es.eigenvectors().col(0);

        // --- Calculate planarity coefficient (a2D) ---
        // This metric describes how "flat" the point distribution is.
        const auto& eigenvalues = es.eigenvalues();

#ifdef DEBUG
        // [DEBUG] Print eigenvalues to check for negative or NaN values
        // std::cout << "[COMP_NEIGH] Eigenvalues (lambda_0, lambda_1, lambda_2): " 
        //         << eigenvalues.transpose() << std::endl;
        if (!eigenvalues.allFinite()) {
            std::cout << "[COMP_NEIGH] CRITICAL: Eigenvalues are NOT finite!" << std::endl;
        }
#endif

        // Use std::max to prevent sqrt of small negative numbers from floating point error
        double sigma1 = std::sqrt(std::max(0.0, eigenvalues[2])); // Std dev along largest principal axis
        double sigma2 = std::sqrt(std::max(0.0, eigenvalues[1])); // Std dev along middle principal axis
        double sigma3 = std::sqrt(std::max(0.0, eigenvalues[0])); // Std dev along smallest principal axis (normal direction)

#ifdef DEBUG
        // [DEBUG] Print intermediate sigma values
        // std::cout << "[COMP_NEIGH] Sigmas (s3, s2, s1): " << sigma3 << ", " << sigma2 << ", " << sigma1 << std::endl;
#endif

        // Planarity: 1 for a perfect plane (sigma3=0), 0 for a line (sigma2=sigma3=0) or sphere (sigma1=sigma2=sigma3)
        constexpr double epsilon = 1e-9;
        if (sigma1 > epsilon) {
            neighborhood.a2D = (sigma2 - sigma3) / sigma1;
        } else {
#ifdef DEBUG
            // [DEBUG] Log when the largest std dev is close to zero
            // std::cout << "[COMP_NEIGH] Warning: Largest eigenvalue (sigma1) is near zero. Setting a2D to 0." << std::endl;
#endif
            neighborhood.a2D = 0.0;
        }

#ifdef DEBUG
        // [DEBUG] Print final computed values before the check and return
        // std::cout << "[COMP_NEIGH] Final Normal: " << neighborhood.normal.transpose() << std::endl;
        // std::cout << "[COMP_NEIGH] Final a2D (Planarity): " << neighborhood.a2D << std::endl;
#endif
        
        if (!std::isfinite(neighborhood.a2D)) {
            // This case is rare but indicates a numerical issue.
            throw std::runtime_error("Planarity coefficient is NaN or inf");
        }

        return neighborhood;
    }

    // ########################################################################
    // parse_json_options
    // ########################################################################

    lidarodom::Options lidarodom::parse_json_options(const std::string& json_path) {
        std::ifstream file(json_path);
        if (!file.is_open()) {throw std::runtime_error("Failed to open JSON file: " + json_path);}

        nlohmann::json json_data;
        try {
            file >> json_data;
        } catch (const nlohmann::json::parse_error& e) {throw std::runtime_error("JSON parse error in " + json_path + ": " + e.what());}

        lidarodom::Options parsed_options;

        if (!json_data.is_object()) {throw std::runtime_error("JSON data must be an object");}

        try {
            // Parse odometry_options object
            if (!json_data.contains("odometry_options") || !json_data["odometry_options"].is_object()) {throw std::runtime_error("Missing or invalid 'odometry_options' object");}
            
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
            if (odometry_options.contains("sequential_threshold_odom")) parsed_options.sequential_threshold_odom = odometry_options["sequential_threshold_odom"].get<int>();
            if (odometry_options.contains("num_threads_odom")) parsed_options.num_threads_odom = odometry_options["num_threads_odom"].get<unsigned int>();
            if (odometry_options.contains("debug_print")) parsed_options.debug_print = odometry_options["debug_print"].get<bool>();
            if (odometry_options.contains("debug_path")) parsed_options.debug_path = odometry_options["debug_path"].get<std::string>();

            // lidarodom::Options
            if (odometry_options.contains("Tb2s") && odometry_options["Tb2s"].is_array() && odometry_options["Tb2s"].size() == 16) {
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        parsed_options.Tb2s(i, j) = odometry_options["Tb2s"][i * 4 + j].get<double>();
                    }
                }
            }
            if (odometry_options.contains("qc_diag") && odometry_options["qc_diag"].is_array() && odometry_options["qc_diag"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.qc_diag(i) = odometry_options["qc_diag"][i].get<double>();
                }
            }
            if (odometry_options.contains("num_extra_states")) parsed_options.num_extra_states = odometry_options["num_extra_states"].get<int>();
            if (odometry_options.contains("power_planarity")) parsed_options.power_planarity = odometry_options["power_planarity"].get<double>();
            if (odometry_options.contains("p2p_max_dist")) parsed_options.p2p_max_dist = odometry_options["p2p_max_dist"].get<double>();
            if (odometry_options.contains("p2p_loss_func")) {
                std::string loss_func = odometry_options["p2p_loss_func"].get<std::string>();
                if (loss_func == "L2") parsed_options.p2p_loss_func = lidarodom::LOSS_FUNC::L2;
                else if (loss_func == "DCS") parsed_options.p2p_loss_func = lidarodom::LOSS_FUNC::DCS;
                else if (loss_func == "CAUCHY") parsed_options.p2p_loss_func = lidarodom::LOSS_FUNC::CAUCHY;
                else if (loss_func == "GM") parsed_options.p2p_loss_func = lidarodom::LOSS_FUNC::GM;
                else {throw std::runtime_error("Invalid p2p_loss_func: " + loss_func);}
            }
            if (odometry_options.contains("p2p_loss_sigma")) parsed_options.p2p_loss_sigma = odometry_options["p2p_loss_sigma"].get<double>();
            if (odometry_options.contains("use_rv")) parsed_options.use_rv = odometry_options["use_rv"].get<bool>();
            if (odometry_options.contains("merge_p2p_rv")) parsed_options.merge_p2p_rv = odometry_options["merge_p2p_rv"].get<bool>();
            if (odometry_options.contains("rv_max_error")) parsed_options.rv_max_error = odometry_options["rv_max_error"].get<double>();
            if (odometry_options.contains("rv_loss_func")) {
                std::string loss_func = odometry_options["rv_loss_func"].get<std::string>();
                if (loss_func == "L2") parsed_options.rv_loss_func = lidarodom::LOSS_FUNC::L2;
                else if (loss_func == "DCS") parsed_options.rv_loss_func = lidarodom::LOSS_FUNC::DCS;
                else if (loss_func == "CAUCHY") parsed_options.rv_loss_func = lidarodom::LOSS_FUNC::CAUCHY;
                else if (loss_func == "GM") parsed_options.rv_loss_func = lidarodom::LOSS_FUNC::GM;
                else {throw std::runtime_error("Invalid rv_loss_func: " + loss_func);}
            }
            if (odometry_options.contains("rv_cov_inv")) parsed_options.rv_cov_inv = odometry_options["rv_cov_inv"].get<double>();
            if (odometry_options.contains("rv_loss_threshold")) parsed_options.rv_loss_threshold = odometry_options["rv_loss_threshold"].get<double>();
            
            if (odometry_options.contains("verbose")) parsed_options.verbose = odometry_options["verbose"].get<bool>();
            if (odometry_options.contains("max_iterations")) parsed_options.max_iterations = odometry_options["max_iterations"].get<int>();
            if (odometry_options.contains("sequential_threshold")) parsed_options.sequential_threshold = odometry_options["sequential_threshold"].get<int>();
            if (odometry_options.contains("num_threads")) parsed_options.num_threads = odometry_options["num_threads"].get<unsigned int>();
            if (odometry_options.contains("delay_adding_points")) parsed_options.delay_adding_points = odometry_options["delay_adding_points"].get<int>();
            if (odometry_options.contains("use_final_state_value")) parsed_options.use_final_state_value = odometry_options["use_final_state_value"].get<bool>();
            if (odometry_options.contains("break_icp_early")) parsed_options.break_icp_early = odometry_options["break_icp_early"].get<bool>();
            if (odometry_options.contains("use_line_search")) parsed_options.use_line_search = odometry_options["use_line_search"].get<bool>();
        
            //###
            if (odometry_options.contains("use_elastic_initialization")) parsed_options.use_elastic_initialization = odometry_options["use_elastic_initialization"].get<bool>();
            if (odometry_options.contains("keyframe_translation_threshold_m")) parsed_options.keyframe_translation_threshold_m = odometry_options["keyframe_translation_threshold_m"].get<double>();
            if (odometry_options.contains("keyframe_rotation_threshold_deg")) parsed_options.keyframe_rotation_threshold_deg = odometry_options["keyframe_rotation_threshold_deg"].get<double>();
            if (odometry_options.contains("use_pointtopoint_factors")) parsed_options.use_pointtopoint_factors = odometry_options["use_pointtopoint_factors"].get<bool>();
            //###

            if (odometry_options.contains("gravity")) parsed_options.gravity = odometry_options["gravity"].get<double>();
            if (odometry_options.contains("use_imu")) parsed_options.use_imu = odometry_options["use_imu"].get<bool>();
            if (odometry_options.contains("use_accel")) parsed_options.use_accel = odometry_options["use_accel"].get<bool>();

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
            if (odometry_options.contains("p0_bias_accel") && odometry_options["p0_bias_accel"].is_array() && odometry_options["p0_bias_accel"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    parsed_options.p0_bias_accel(i) = odometry_options["p0_bias_accel"][i].get<double>();
                }
            }
            if (odometry_options.contains("q_bias_accel") && odometry_options["q_bias_accel"].is_array() && odometry_options["q_bias_accel"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    parsed_options.q_bias_accel(i) = odometry_options["q_bias_accel"][i].get<double>();
                }
            }
            if (odometry_options.contains("p0_bias_gyro")) parsed_options.p0_bias_gyro = odometry_options["p0_bias_gyro"].get<double>();
            if (odometry_options.contains("q_bias_gyro")) parsed_options.q_bias_gyro = odometry_options["q_bias_gyro"].get<double>();
            if (odometry_options.contains("acc_loss_func")) parsed_options.acc_loss_func = odometry_options["acc_loss_func"].get<std::string>();
            if (odometry_options.contains("acc_loss_sigma")) parsed_options.acc_loss_sigma = odometry_options["acc_loss_sigma"].get<double>();
            if (odometry_options.contains("gyro_loss_func")) parsed_options.gyro_loss_func = odometry_options["gyro_loss_func"].get<std::string>();
            if (odometry_options.contains("gyro_loss_sigma")) parsed_options.gyro_loss_sigma = odometry_options["gyro_loss_sigma"].get<double>();
            
            if (odometry_options.contains("Ti2m_init_only")) parsed_options.Ti2m_init_only = odometry_options["Ti2m_init_only"].get<bool>();
            if (odometry_options.contains("Ti2m_init_cov") && odometry_options["Ti2m_init_cov"].is_array() && odometry_options["Ti2m_init_cov"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.Ti2m_init_cov(i) = odometry_options["Ti2m_init_cov"][i].get<double>();
                }
            }
            if (odometry_options.contains("qg_diag") && odometry_options["qg_diag"].is_array() && odometry_options["qg_diag"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.qg_diag(i) = odometry_options["qg_diag"][i].get<double>();
                }
            }
            if (odometry_options.contains("Tm2b_init") && odometry_options["Tm2b_init"].is_array() && odometry_options["Tm2b_init"].size() == 16) {
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        parsed_options.Tm2b_init(i, j) = odometry_options["Tm2b_init"][i * 4 + j].get<double>();
                    }
                }
            }
            if (odometry_options.contains("filter_lifetimes")) parsed_options.filter_lifetimes = odometry_options["filter_lifetimes"].get<bool>();
            
        } catch (const nlohmann::json::exception& e) {throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));}

        return parsed_options;
    }

    // ########################################################################
    // lidarodom constructor
    // ########################################################################

    lidarodom::lidarodom(const std::string& json_path)
        : Odometry(parse_json_options(json_path)), options_(parse_json_options(json_path)) {
        Tb2s_var_ = finalicp::se3::SE3StateVar::MakeShared(math::se3::Transformation(options_.Tb2s));
        Tb2s_var_->locked() = true;
        sliding_window_filter_ = finalicp::SlidingWindowFilter::MakeShared(options_.num_threads);
    }

    // ########################################################################
    // ~lidarodom deconstructor
    // ########################################################################
    lidarodom::~lidarodom() {
#ifdef DEBUG
        std::cout << "[DECONSTRUCT] lidarodom object is being destroyed." << std::endl;
#endif
    }

    // ########################################################################
    // saveResults
    // ########################################################################
    void lidarodom::saveResults(const std::string& timestamp) {
        using namespace finalicp::traj;

        // --- Safety Check: Don't do anything if there's no data ---
        if (trajectory_.empty() || trajectory_vars_.empty()) {
            std::cout << "[saveResults] Trajectory is empty, skipping file write." << std::endl;
            return;
        }

        std::cout << "[saveResults] Saving final trajectory and map..." << std::endl;

        // --- Create filenames using the provided timestamp ---
        std::string trajectory_filename = options_.debug_path + "/trajectory_" + timestamp + ".txt";
        std::string pointcloud_filename = options_.debug_path + "/map_" + timestamp + ".txt";

        // --- Wrap file-writing logic in a try-catch block for maximum safety ---
        try {
            // Define trajectory file writing task
            auto write_trajectory = [&]() {
                // ... (The exact same logic as your original destructor) ...
                std::ofstream trajectory_file(trajectory_filename, std::ios::out);
                if (!trajectory_file.is_open()) {
                    std::cerr << "[saveResults] ERROR: Failed to open trajectory file for writing: " << trajectory_filename << std::endl;
                    return;
                }

                auto full_trajectory = finalicp::traj::const_vel::Interface::MakeShared(options_.qc_diag);
                for (const auto& var : trajectory_vars_) {
                    full_trajectory->add(var.time, var.Tm2b, var.wb2m_inr);
                }

                std::stringstream buffer;
                buffer << std::fixed << std::setprecision(12);
                double begin_time = trajectory_.front().begin_timestamp;
                double end_time = trajectory_.back().end_timestamp;
                constexpr double dt = 0.01;
                for (double time = begin_time; time <= end_time; time += dt) {
                    Time traj_time(time);
                    const auto Tm2b = full_trajectory->getPoseInterpolator(traj_time)->value().matrix();
                    const auto wb2m_inr = full_trajectory->getVelocityInterpolator(traj_time)->value();
                    buffer << traj_time.nanosecs() << " "
                        << Tm2b(0, 0) << " " << Tm2b(0, 1) << " " << Tm2b(0, 2) << " " << Tm2b(0, 3) << " "
                        << Tm2b(1, 0) << " " << Tm2b(1, 1) << " " << Tm2b(1, 2) << " " << Tm2b(1, 3) << " "
                        << Tm2b(2, 0) << " " << Tm2b(2, 1) << " " << Tm2b(2, 2) << " " << Tm2b(2, 3) << " "
                        << Tm2b(3, 0) << " " << Tm2b(3, 1) << " " << Tm2b(3, 2) << " " << Tm2b(3, 3) << " "
                        << wb2m_inr(0) << " " << wb2m_inr(1) << " " << wb2m_inr(2) << " "
                        << wb2m_inr(3) << " " << wb2m_inr(4) << " " << wb2m_inr(5) << "\n";
                }
                trajectory_file << buffer.str();
            };

            // Define point cloud file writing task
            auto write_pointcloud = [&]() {
                std::ofstream pointcloud_file(pointcloud_filename, std::ios::out);
                if (!pointcloud_file.is_open()) {
                    std::cerr << "[saveResults] ERROR: Failed to open map file for writing: " << pointcloud_filename << std::endl;
                    return;
                }
                map_.dumpToStream(pointcloud_file);
            };

            // Execute both tasks in parallel
            tbb::parallel_invoke(write_trajectory, write_pointcloud);

            std::cout << "[saveResults] Successfully saved results to " << options_.debug_path << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[saveResults] CRITICAL ERROR while saving results: " << e.what() << std::endl;
        }
    }

    // ########################################################################
    // setInitialPose
    // ########################################################################
    
    void lidarodom::initT(const Eigen::Matrix4d& T) {
        options_.Tm2b_init = T;
    }

    // ########################################################################
    // trajectory
    // ########################################################################

    Trajectory lidarodom::trajectory() {

        if (!options_.use_final_state_value) {
            return trajectory_;
        }

        // Build full trajectory
        auto full_trajectory = finalicp::traj::const_vel::Interface::MakeShared(options_.qc_diag);
        for (const auto& var : trajectory_vars_) {
            full_trajectory->add(var.time, var.Tm2b, var.wb2m_inr);
        }

        using namespace finalicp::se3;
        using namespace finalicp::traj;

        // Cache T_sr inverse
        const Eigen::Matrix4d Ts2b = options_.Tb2s.inverse();

        // Sequential update
        for (auto& frame : trajectory_) {
            // Begin pose
            Time begin_slam_time(frame.begin_timestamp);
            const auto begin_Tb2m = inverse(full_trajectory->getPoseInterpolator(begin_slam_time))->evaluate().matrix();
            const auto begin_Ts2m = begin_Tb2m * Ts2b;
            frame.begin_R = begin_Ts2m.block<3, 3>(0, 0);
            frame.begin_t = begin_Ts2m.block<3, 1>(0, 3);

            // Mid pose
            Time mid_slam_time(static_cast<double>(frame.getEvalTime()));
            const auto mid_Tb2m = inverse(full_trajectory->getPoseInterpolator(mid_slam_time))->evaluate().matrix();
            const auto mid_Ts2m = mid_Tb2m * Ts2b;
            frame.setMidPose(mid_Ts2m);

            // End pose
            Time end_slam_time(frame.end_timestamp);
            const auto end_Tb2m = inverse(full_trajectory->getPoseInterpolator(end_slam_time))->evaluate().matrix();
            const auto end_Ts2m = end_Tb2m * Ts2b;
            frame.end_R = end_Ts2m.block<3, 3>(0, 0);
            frame.end_t = end_Ts2m.block<3, 1>(0, 3);
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

    auto lidarodom::registerFrame(const DataFrame &const_frame) -> RegistrationSummary {
        
        RegistrationSummary summary;

#ifdef DEBUG
        // Initialize timers for performance debugging if enabled
        std::vector<std::pair<std::string, std::unique_ptr<finalicp::Stopwatch<>>>> timer;
        timer.emplace_back("initialization ..................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("gridsampling ....................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("icp ................................ ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("updateMap .......................... ", std::make_unique<finalicp::Stopwatch<>>(false));
#endif
        // Step 1: Add new frame to trajectory
        // Create a new entry in the trajectory vector for the current frame
        int index_frame = trajectory_.size(); // start from 0,1,2,3,4
        trajectory_.emplace_back();
        // Step 2: Validate input point cloud
        // Check if the input point cloud is empty; return failure if so
        if (const_frame.pointcloud.empty()) {
#ifdef DEBUG
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "CRITICAL: Frame " << trajectory_.size() << " REJECTED: Input point cloud is empty." << std::endl;
#endif
            summary.success = false;
            return summary;
        }

#ifdef DEBUG
        // [DEBUG] Announce the start of processing for a new frame
        std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Starting RegisterFrame for index " << index_frame << std::endl;
        std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Input pointcloud size: " << const_frame.pointcloud.size() << std::endl;
#endif

        // Step 3: Initialize frame metadata
        // Set up timestamp and motion data for the new frame
        initializeTimestamp(index_frame, const_frame);                                  //####!!! 1 tbb included // find the min and max timestamp in a single frame

#ifdef DEBUG
        // [DEBUG] Check the timestamps immediately after they are calculated
        // std::cout << std::fixed << std::setprecision(12) 
        //           << "[REG DEBUG | Frame " << index_frame << "]" << "After initializeTimestamp: begin=" << trajectory_[index_frame].begin_timestamp
        //           << ", end=" << trajectory_[index_frame].end_timestamp << std::endl;
#endif

        initializeMotion(index_frame);                                                  //####!!! 2 // estimate the motion based on prev and prev*prev frame

#ifdef DEBUG
        if (index_frame == 0) {
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Frame 0 Initial Pose (R_ms):\n" << trajectory_[index_frame].end_R  << std::endl;
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Frame 0 Initial Pose (t_ms):\n" << trajectory_[index_frame].end_t.transpose() << std::endl;
        }
        // [DEBUG] Check the initial motion prediction
        if (!trajectory_[index_frame].begin_R.allFinite() || !trajectory_[index_frame].end_R.allFinite()) {
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "CRITICAL: Non-finite rotation after initializeMotion!" << std::endl;
        }
#endif

        // Step 4: Process input point cloud
        // Convert and prepare the point cloud for registration
#ifdef DEBUG
        if (!timer.empty()) timer[0].second->start();
#endif
        // this is deskewing process
        auto frame = initializeFrame(index_frame, const_frame.pointcloud);              //####!!! 3 tbb included // correct frame point cloud based on estimated motion

#ifdef DEBUG
        if (!timer.empty()) timer[0].second->stop();
        // std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "After initializeFrame, size is: " << frame.size() << std::endl;
#endif

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
#ifdef DEBUG
            if (!timer.empty()) timer[1].second->start();
#endif
            grid_sampling(frame, keypoints, sample_voxel_size, options_.sequential_threshold);   //####!!! 4 has outlier removal

#ifdef DEBUG
            if (!timer.empty()) timer[1].second->stop();
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "After grid_sampling, keypoints size: " << keypoints.size() << std::endl;
#endif
            // Step 5b: Perform Iterative Closest Point (ICP) registration
            // Align current frame with previous frames using IMU and pose data
            const auto& imu_data_vec = const_frame.imu_data_vec;
#ifdef DEBUG
            if (!timer.empty()) timer[2].second->start();
#endif

            summary.success = icp(index_frame, keypoints, imu_data_vec); //####!!! 5

#ifdef DEBUG
            if (!timer.empty()) timer[2].second->stop();
            // [DEBUG] Report ICP result immediately
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "ICP finished. Success: " << (summary.success ? "true" : "false") << std::endl;
#endif
            summary.keypoints = keypoints;
            if (!summary.success) {
#ifdef DEBUG
                std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "ICP failed for frame " << index_frame << std::endl;
#endif
                return summary;}
        } else { // !!!!!!!!!!! this is responsible for initial frame 0 ######################

#ifdef DEBUG
            // [DEBUG] Announce first frame initialization
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Performing first frame (index 0) initialization." << std::endl;
#endif
            // Step 5c: Initialize first frame
            // Set up initial state and transformations for the trajectory start
            using namespace finalicp;
            using namespace finalicp::se3;
            using namespace finalicp::vspace;
            using namespace finalicp::traj;

#ifdef DEBUG
            if (!timer.empty()) timer[2].second->start();
#endif

            // Define initial transformations and velocities
            math::se3::Transformation Tm2b;
            math::se3::Transformation Ti2m;
            math::se3::Transformation Tb2s(options_.Tb2s);

            Eigen::Matrix<double, 6, 1> wb2m_inr = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();

            // Initialize trajectory variables for the beginning of the frame
            const double begin_time = trajectory_[index_frame].begin_timestamp;
            Time begin_slam_time(begin_time);
            auto begin_Tm2b_var = SE3StateVar::MakeShared(Tm2b);
            auto begin_wb2m_inr_var = VSpaceStateVar<6>::MakeShared(wb2m_inr);
            auto begin_b_var = VSpaceStateVar<6>::MakeShared(b_zero);
            // Initialize Ti2m_var DIRECTLY with the ground truth value
            auto begin_Ti2m_var = SE3StateVar::MakeShared(Ti2m); 
            trajectory_vars_.emplace_back(begin_slam_time, std::move(begin_Tm2b_var), std::move(begin_wb2m_inr_var),
                                        std::move(begin_b_var), std::move(begin_Ti2m_var));

            // Initialize trajectory variables for the end of the frame
            const double end_time = trajectory_[index_frame].end_timestamp;
            Time end_slam_time(end_time);
            auto end_Tm2b_var = SE3StateVar::MakeShared(Tm2b);
            auto end_wb2m_inr_var = VSpaceStateVar<6>::MakeShared(wb2m_inr);
            auto end_b_var = VSpaceStateVar<6>::MakeShared(b_zero);
            // Initialize Ti2m_var DIRECTLY with the ground truth value
            auto end_Ti2m_var = SE3StateVar::MakeShared(Ti2m); 
            trajectory_vars_.emplace_back(end_slam_time, std::move(end_Tm2b_var), std::move(end_wb2m_inr_var),
                                        std::move(end_b_var), std::move(end_Ti2m_var));

#ifdef DEBUG
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Frame 0: Created " << trajectory_vars_.size() << " initial state variables." << std::endl;
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Frame 0 timestamps: begin=" << std::fixed << begin_time << ", end=" << end_time << std::endl;
#endif
            Eigen::Matrix<double, 6, 1> xi_i2m = initialize_gravity(const_frame.imu_data_vec);
            begin_Ti2m_var->update(xi_i2m);
            end_Ti2m_var->update(xi_i2m);

            to_marginalize_ = 1;

            // Step 5e: Initialize covariance matrices
            // Set initial uncertainties for pose, velocity, and acceleration
            trajectory_[index_frame].end_Tm2b_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
            trajectory_[index_frame].end_wb2m_inr_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
            trajectory_[index_frame].end_state_cov = Eigen::Matrix<double, 18, 18>::Identity() * 1e-4;

            summary.success = true;

#ifdef DEBUG
            if (!timer.empty()) timer[2].second->stop();
#endif
        }

        // Step 6: Store processed points
        // Save the processed point cloud to the trajectory
        trajectory_[index_frame].points = std::move(frame);

        const Eigen::Vector3d t = trajectory_[index_frame].end_t;
        const Eigen::Matrix3d r = trajectory_[index_frame].end_R;

        // Step 7: Update the map
        // Incorporate points into the global map, with optional delay
#ifdef DEBUG
        if (!timer.empty()) timer[3].second->start();
#endif

        if (index_frame == 0) {
#ifdef DEBUG
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Updating map for frame 0." << std::endl;
#endif
            updateMap(index_frame, index_frame);                                        //####!!! 7

        } else if ((index_frame - options_.delay_adding_points) > 0) {
#ifdef DEBUG
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Updating map using frame " << index_frame - options_.delay_adding_points << "." << std::endl;
#endif
            updateMap(index_frame, (index_frame - options_.delay_adding_points));
            t_prev_ = t;
            r_prev_ = r;
        }

#ifdef DEBUG
        if (!timer.empty()) timer[3].second->stop();
        std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Map size is now: " << map_.size() << std::endl;
#endif

        // Step 8: Correct point cloud positions
        // Apply transformations to correct point positions based on trajectory
        // Validate trajectory poses for correction
        const auto& traj = trajectory_[index_frame];

        // Step 9: Prepare output summary
        // Set corrected points, rotation, and translation for output
        summary.corrected_points = summary.keypoints;
        summary.Rs2m = traj.end_R;
        summary.ts2m = traj.end_t;

        // Step 10: Output debug timers
        // Print timing information if debug mode is enabled
#ifdef DEBUG
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "OUTER LOOP TIMERS" << std::endl;
            for (size_t i = 0; i < timer.size(); i++) {
                std::cout << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
            }
            std::cout << "[REG DEBUG | Frame " << index_frame << "]" << "Finished RegisterFrame for index " << index_frame << std::endl;
#endif
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

    void lidarodom::initializeTimestamp(int index_frame, const DataFrame &const_frame) {
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
            // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
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
        // Calculate the midpoint of the scan
        const double mid_timestamp = (max_timestamp + min_timestamp) / 2.0;

#ifdef DEBUG
            // [ADDED DEBUG] Print all calculated timestamps before assignment
            std::cout << "[INIT TS DEBUG | Frame " << index_frame << "] " << "Frame " << index_frame << ": min=" << std::fixed << min_timestamp
                    << ", max=" << max_timestamp << ", mid=" << mid_timestamp << std::endl;
#endif

        // Assign to trajectory
        trajectory_[index_frame].begin_timestamp = min_timestamp;
        trajectory_[index_frame].end_timestamp = max_timestamp;
        trajectory_[index_frame].setEvalTime(mid_timestamp);
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

    void lidarodom::initializeMotion(int index_frame) {// the first frame should be initialized with Tmr

#ifdef DEBUG
        // [ADDED DEBUG] Announce entry into the function
        std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Initializing motion for frame " << index_frame << ". ---" << std::endl;
#endif

        if (index_frame == 0) { //MAIN ALLERT as T_sr is identity. T_ms is exactly same as T_mr
            // --- For the very first frame, use the ground truth initial pose ---
            const Eigen::Matrix4d Tb2m = Eigen::Matrix<double, 4, 4>::Identity();

            // 3. Get the transformation from sensor to robot (T_rs).
            const Eigen::Matrix4d Ts2b = options_.Tb2s.inverse();

            // 4. Calculate the initial sensor pose in the map: T_ms = T_mr * T_rs.
            const Eigen::Matrix4d Ts2m = Tb2m * Ts2b;

#ifdef DEBUG
            // [ADDED DEBUG] Print the initial transformations for Frame 0
            std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Frame 0: Using initial pose from options." << std::endl;
            std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Frame 0: Initial Sensor Pose (Ts2m):\n" << Ts2m << std::endl;
            if (!Ts2m.allFinite()) {
                std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "CRITICAL: Initial pose Ts2m is non-finite (NaN or inf)!" << std::endl;
            } else {
                std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Initial pose Ts2m is finite." << std::endl;
            }
#endif
            // 5. Set the trajectory's initial pose.
            trajectory_[index_frame].begin_R = Ts2m.block<3, 3>(0, 0);
            trajectory_[index_frame].begin_t = Ts2m.block<3, 1>(0, 3);
            trajectory_[index_frame].end_R = Ts2m.block<3, 3>(0, 0);
            trajectory_[index_frame].end_t = Ts2m.block<3, 1>(0, 3);

        } else if (index_frame == 1) {
            // For the second frame, its motion starts from the end of the first frame.
            trajectory_[index_frame].begin_R = trajectory_[index_frame - 1].end_R;
            trajectory_[index_frame].begin_t = trajectory_[index_frame - 1].end_t;
            trajectory_[index_frame].end_R = trajectory_[index_frame - 1].end_R;
            trajectory_[index_frame].end_t = trajectory_[index_frame - 1].end_t;

#ifdef DEBUG
        // [ADDED DEBUG] Confirm the pose was copied for Frame 1
        std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Frame 1: Copying pose from end of Frame 0." << std::endl;
        std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Frame 1: Initial Pose (Rotation):\n" << trajectory_[index_frame].begin_R << std::endl;
        std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Frame 1: Initial Pose (Translation): " << trajectory_[index_frame].begin_t.transpose() << std::endl;
        if (!trajectory_[index_frame].begin_R.allFinite() || !trajectory_[index_frame].end_R.allFinite() || !trajectory_[index_frame].begin_t.allFinite() || !trajectory_[index_frame].end_t.allFinite()) {
            std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "CRITICAL: Initial pose Ts2m is non-finite (NaN or inf)!" << std::endl;
        } else {
            std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Initial pose T_ms is finite." << std::endl;
        }
#endif
        
        } else { 
            // For all subsequent frames, extrapolate motion from the previous two.
            const auto& prev = trajectory_[index_frame - 1];
            const auto& prev_prev = trajectory_[index_frame - 2];

            // Compute relative transformation between previous sensor poses
            const Eigen::Matrix3d R_rel = prev.end_R * prev_prev.end_R.transpose();
            const Eigen::Vector3d t_rel = prev.end_t - prev_prev.end_t;

            // Extrapolate the end pose of the current sensor frame
            trajectory_[index_frame].end_R = R_rel * prev.end_R;
            trajectory_[index_frame].end_t = prev.end_t + R_rel * t_rel; // Corrected: Transform t_rel into the new frame

            // Set the begin pose to the previous frame's end pose
            trajectory_[index_frame].begin_R = prev.end_R;
            trajectory_[index_frame].begin_t = prev.end_t;

#ifdef DEBUG
            // [ADDED DEBUG] Show the extrapolated motion
            std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Frame " << index_frame << ": Extrapolating motion." << std::endl;
            std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Relative Motion (t_rel): " << t_rel.transpose() << std::endl;
            std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Extrapolated End Pose (Translation): " << trajectory_[index_frame].end_t.transpose() << std::endl;
            if (!trajectory_[index_frame].begin_R.allFinite() || !trajectory_[index_frame].end_R.allFinite() || !trajectory_[index_frame].begin_t.allFinite() || !trajectory_[index_frame].end_t.allFinite()) {
                std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "CRITICAL: Initial pose T_ms is non-finite (NaN or inf)!" << std::endl;
            } else {
                std::cout << "[INIT MOTION DEBUG | Frame " << index_frame << "] " << "Initial pose T_ms is finite." << std::endl;
            }
#endif
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

    std::vector<Point3D> lidarodom::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
        // this is critical as the code assume T_sr is identity. if T_sr is not identity then we need to add some more algorithm.
#ifdef DEBUG
        // [ADDED DEBUG] Announce entry and check input size
        std::cout << "[FRAME INIT DEBUG | Frame " << index_frame << "] " << "Initializing frame " << index_frame << " with " << const_frame.size() << " input points." << std::endl;
#endif

        // Initialize point cloud
        std::vector<Point3D> frame = const_frame; // Copy necessary due to const input

        // Select voxel size
        const double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;

        // Subsample
        sub_sample_frame(frame, sample_size, options_.sequential_threshold);

#ifdef DEBUG
        // [ADDED DEBUG] Show size after subsampling
        std::cout << "[FRAME INIT DEBUG | Frame " << index_frame << "] " << "Frame size after subsampling: " << frame.size() << " points." << std::endl;
#endif

        // Shuffle points to avoid bias
        std::mt19937_64 g; // Fixed seed for reproducibility
        std::shuffle(frame.begin(), frame.end(), g);

        // Validate poses
        const auto& traj = trajectory_[index_frame]; //contain R_ms and t_ms

#ifdef DEBUG
        // [ADDED DEBUG] Check input poses for validity before using them
        if (!traj.begin_R.allFinite() || !traj.end_R.allFinite() || !traj.begin_t.allFinite() || !traj.end_t.allFinite()) {
            std::cout << "[FRAME INIT DEBUG | Frame " << index_frame << "] " << "CRITICAL: Input trajectory poses for deskewing are non-finite!" << std::endl;
        }
        std::cout << "[FRAME INIT DEBUG | Frame " << index_frame << "] " << "Deskewing with begin_t: " << traj.begin_t.transpose() << " and end_t: " << traj.end_t.transpose() << std::endl;
#endif

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
            // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
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

#ifdef DEBUG
        // [ADDED DEBUG] Final check to ensure all output points are valid
        bool all_points_finite = true;
        for (const auto& point : frame) {
            if (!point.pt.allFinite()) {
                std::cout << "[FRAME INIT DEBUG | Frame " << index_frame << "] " << "CRITICAL: A deskewed point is non-finite (NaN or inf)!" << std::endl;
                all_points_finite = false;
                break;
            }
        }
        if (all_points_finite) {
            std::cout << "[FRAME INIT DEBUG | Frame " << index_frame << "] " << "All " << frame.size() << " deskewed points are finite." << std::endl;
        }
#endif

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

    void lidarodom::updateMap(int index_frame, int update_frame) {
    
#ifdef DEBUG
        // [DEBUG] Announce the start of the function and its parameters
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Starting updateMap" << std::endl;
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Current frame index: " << index_frame << ", Updating with data from frame: " << update_frame << std::endl;
#endif

        // Map parameters
        const double kSizeVoxelMap = options_.size_voxel_map;
        const double kMinDistancePoints = options_.min_distance_points;
        const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

        // Update frame
        auto& frame = trajectory_[update_frame].points;
        if (frame.empty()) {
#ifdef DEBUG
            std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Frame " << update_frame << " is empty. Nothing to add to map. Returning." << std::endl;
#endif
            return; // No points to add
        }

#ifdef DEBUG
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Frame " << update_frame << " contains " << frame.size() << " points to process." << std::endl;
#endif

        // Motion correction with SLAM interpolation
        auto update_trajectory = finalicp::traj::const_vel::Interface::MakeShared(options_.qc_diag);
        const double begin_timestamp = trajectory_[update_frame].begin_timestamp;
        const double end_timestamp = trajectory_[update_frame].end_timestamp;

        const finalicp::traj::Time begin_slam_time(begin_timestamp); // Fixed: Use finalicp::traj::Time
        const finalicp::traj::Time end_slam_time(end_timestamp);     // Fixed: Use finalicp::traj::Time

        // Add trajectory states
        size_t num_states = 0;
        size_t start_idx = std::max(static_cast<int>(to_marginalize_) - 1, 0);
        for (size_t i = start_idx; i < trajectory_vars_.size(); i++) {
            const auto& var = trajectory_vars_.at(i);
            update_trajectory->add(var.time, var.Tm2b, var.wb2m_inr);
            num_states++;
            if (var.time == end_slam_time) break;
            if (var.time > end_slam_time) {
                throw std::runtime_error("Trajectory variable time exceeds end_slam_time in updateMap for frame " + std::to_string(update_frame));
            }
        }

#ifdef DEBUG
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Building interpolation trajectory from state index " << start_idx 
                << " to " << (start_idx + num_states - 1) << " (" << num_states << " states)." << std::endl;
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Trajectory covers time range (inclusive): " << std::fixed << std::setprecision(12) 
                << begin_slam_time.seconds() << " - " << end_slam_time.seconds() 
                << ", with num states: " << num_states << std::endl;
#endif

        // Collect unique timestamps
        std::set<double> unique_point_times_set;
        for (const auto& point : frame) {
            unique_point_times_set.insert(point.timestamp);
        }
        std::vector<double> unique_point_times(unique_point_times_set.begin(), unique_point_times_set.end());
#ifdef DEBUG
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Found " << unique_point_times.size() << " unique timestamps in the point cloud." << std::endl;
#endif
        // Cache interpolated poses
        const Eigen::Matrix4d Ts2b = options_.Tb2s.inverse(); // transformation of sensor relative to robot

        std::map<double, Eigen::Matrix4d> Ts2m_cache_map;
        if (unique_point_times.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential pose interpolation
            for (const auto& ts : unique_point_times) {
                const auto Tm2b_intp_eval = update_trajectory->getPoseInterpolator(finalicp::traj::Time(ts)); // Correctly gets the Robot-in-Map pose (Tb2m).

                const Eigen::Matrix4d Ts2m = Tm2b_intp_eval->value().inverse().matrix() * Ts2b; // This correctly computes the final Sensor-to-Map transformation (T_ms = T_mr * T_rs).

                Ts2m_cache_map[ts] = Ts2m; 
            }
        } else {
            // Parallel pose interpolation with TBB
            // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::concurrent_vector<Eigen::Matrix4d> Ts2m_cache(unique_point_times.size());
            tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
                    for (size_t jj = range.begin(); jj != range.end(); ++jj) {
                        const auto& ts = unique_point_times[jj];
                        const auto Tm2b_intp_eval = update_trajectory->getPoseInterpolator(finalicp::traj::Time(ts));
                        const Eigen::Matrix4d Ts2m = Tm2b_intp_eval->value().inverse().matrix() * Ts2b;
                        Ts2m_cache[jj] = Ts2m;
                    }
                });

            // Populate cache map
            for (size_t jj = 0; jj < unique_point_times.size(); ++jj) {
                Ts2m_cache_map[unique_point_times[jj]] = Ts2m_cache[jj];
            }
        }

#ifdef DEBUG
        // [DEBUG] Verify that cached poses are valid
        bool poses_are_finite = true;
        for(const auto& pair : Ts2m_cache_map) {
            if (!pair.second.allFinite()) {
                poses_are_finite = false;
                std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "CRITICAL: Cached pose for timestamp " << pair.first << " is NOT finite!" << std::endl;
                break;
            }
        }
        if (poses_are_finite) {
            std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "All " << Ts2m_cache_map.size() << " cached poses are finite." << std::endl;
        }
#endif

        // Apply transformations
        if (frame.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential point transformation
            for (auto& point : frame) {
                try {
                    const auto& Ts2m = Ts2m_cache_map.at(point.timestamp);
                    point.pt = Ts2m.block<3, 3>(0, 0) * point.raw_pt + Ts2m.block<3, 1>(0, 3); // Correctly applies the transformation to move a point from the sensor's frame into the global map frame.
                } catch (const std::out_of_range&) {
                    throw std::runtime_error("Timestamp not found in cache in updateMap for frame " + std::to_string(update_frame));
                }
            }
        } else {
            // Parallel point transformation with TBB
            // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        auto& point = frame[i];
                        try {
                            const auto& Ts2m = Ts2m_cache_map.at(point.timestamp);
                            point.pt = Ts2m.block<3, 3>(0, 0) * point.raw_pt + Ts2m.block<3, 1>(0, 3);
                        } catch (const std::out_of_range&) {
                            throw std::runtime_error("Timestamp not found in cache in updateMap for frame " + std::to_string(update_frame));
                        }
                    }
                });
        }

#ifdef DEBUG
        // [DEBUG] Verify that transformed points are valid
        for(const auto& point : frame) {
            if(!point.pt.allFinite()) {
                std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "CRITICAL: Transformed point `pt` is NOT finite!" << std::endl;
                break;
            }
        }
#endif

        // Update map
        map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);

#ifdef DEBUG
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Map size after adding new points " << map_.size() << " points." << std::endl;
#endif

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

#ifdef DEBUG
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Removing points farther than " << kMaxDistance << "m from " << location.transpose() << std::endl;
        std::cout << "[MAP DEBUG | Frame " << index_frame << "] " << "Map size after outlier point removal " << map_.size() << " points." << std::endl;
#endif

    }

    // ########################################################################
    // initialize_gravity 
    // ########################################################################

    /*The lidarodom::initialize_gravity function estimates the initial IMU-to-map transformation (Ti2m) 
    for a LiDAR-inertial odometry system using a vector of IMU data (imu_data_vec). 
    It validates non-empty input and finite accelerations, initializes locked state variables (pose Tm2b, biases, and velocities), 
    and sets up a noise model and L2 loss function based on options_.r_imu_acc and gravity. 
    It creates cost terms for acceleration errors (sequentially or in parallel via TBB, depending on size) 
    and adds a prior cost for Ti2m with covariance from options_.Ti2m_init_cov. 
    A Gauss-Newton solver optimizes the problem to refine Ti2m, returning its 6D vector representation after ensuring finite results, 
    enabling gravity-aligned initialization for odometry.*/

    Eigen::Matrix<double, 6, 1> lidarodom::initialize_gravity(const std::vector<finalicp::IMUData> &imu_data_vec) {

#ifdef DEBUG
        // [ADDED DEBUG] Check if we have any IMU data to begin with
        std::cout << "[GRAVITY INIT DEBUG] Received " << imu_data_vec.size() << " IMU data points for initialization." << std::endl;
        if (imu_data_vec.empty()) {
            std::cout << "[GRAVITY INIT DEBUG] CRITICAL: No IMU data provided, cannot initialize gravity. Returning zero vector." << std::endl;
            return Eigen::Matrix<double, 6, 1>::Zero();
        }
#endif

        // Initialize state variables
        const auto Tm2b_init = finalicp::se3::SE3StateVar::MakeShared(math::se3::Transformation());
        math::se3::Transformation Ti2m;
        const auto Ti2m_var = finalicp::se3::SE3StateVar::MakeShared(Ti2m);
        Tm2b_init->locked() = true;

        Eigen::Matrix<double, 6, 1> b_zero = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> dw_zero = Eigen::Matrix<double, 6, 1>::Zero();
        const auto bias = finalicp::vspace::VSpaceStateVar<6>::MakeShared(b_zero);
        const auto dwb2m_inr = finalicp::vspace::VSpaceStateVar<6>::MakeShared(dw_zero);
        bias->locked() = true;
        dwb2m_inr->locked() = true;

        // Initialize noise model and loss function
        Eigen::Matrix<double, 3, 3> R = options_.r_imu_acc.asDiagonal();
        auto noise_model = finalicp::StaticNoiseModel<3>::MakeShared(R);
        auto loss_func = finalicp::L2LossFunc::MakeShared();

        // Create cost terms
        std::vector<finalicp::BaseCostTerm::ConstPtr> cost_terms;
        cost_terms.reserve(imu_data_vec.size()); // +1 for prior term
        for (const auto &imu_data : imu_data_vec) {
            auto acc_error_func = finalicp::imu::AccelerationError(Tm2b_init, dwb2m_inr, bias, Ti2m_var, imu_data.lin_acc);
            acc_error_func->setGravity(options_.gravity);
            const auto acc_cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(acc_error_func, noise_model, loss_func);
            cost_terms.emplace_back(acc_cost);
        }

#ifdef DEBUG
        // [ADDED DEBUG] Confirm that cost terms were actually created
        std::cout << "[GRAVITY INIT DEBUG] Created " << cost_terms.size() << " acceleration cost terms." << std::endl;
#endif

        {
            // Add prior cost term for Ti2m
            Eigen::Matrix<double, 6, 6> init_Ti2m_cov = options_.Ti2m_init_cov.asDiagonal();
            init_Ti2m_cov(3, 3) = 1.0;
            init_Ti2m_cov(4, 4) = 1.0;
            init_Ti2m_cov(5, 5) = 1.0;
            math::se3::Transformation Ti2m_zero;
            auto Ti2m_error = finalicp::se3::se3_error(Ti2m_var, Ti2m_zero);
            auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(init_Ti2m_cov);
            auto loss_func = finalicp::L2LossFunc::MakeShared();
            cost_terms.emplace_back(finalicp::WeightedLeastSqCostTerm<6>::MakeShared(Ti2m_error, noise_model, loss_func));
        }

        // Solve optimization problem
        finalicp::OptimizationProblem problem;
        for (const auto& cost : cost_terms) {
            problem.addCostTerm(cost);
        }
        problem.addStateVariable(Ti2m_var);

        finalicp::GaussNewtonSolverNVA::Params params;
        params.max_iterations = static_cast<unsigned int>(options_.max_iterations);
        finalicp::GaussNewtonSolverNVA solver(problem, params);
        solver.optimize();

#ifdef DEBUG
        std::cout<< "[GRAVITY INIT DEBUG] Ti2m:\n" << Ti2m_var->value().matrix() << std::endl;
        std::cout << "[GRAVITY INIT DEBUG] Ti2m_var:\n"  << Ti2m_var->value().vec() << std::endl;
        // [ADDED DEBUG] Check if the result of the optimization is valid
        if (!Ti2m_var->value().vec().allFinite()) {
            std::cout << "[GRAVITY INIT DEBUG] CRITICAL: Solver produced a non-finite (NaN or inf) result!" << std::endl;
        } else {
            std::cout << "[GRAVITY INIT DEBUG] Solver finished, result is finite." << std::endl;
        }
#endif
        
        return Ti2m_var->value().vec();
    }

    // ########################################################################
    // icp 
    // ########################################################################

    bool lidarodom::icp(int index_frame, std::vector<Point3D> &keypoints, const std::vector<finalicp::IMUData> &imu_data_vec) {

        using namespace finalicp;
        using namespace finalicp::se3;
        using namespace finalicp::traj;
        using namespace finalicp::vspace;
        using namespace finalicp::imu;

#ifdef DEBUG
        // [DEBUG] Initial check at the start of the function
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Starting with " << keypoints.size() << " keypoints." <<  ", "<< imu_data_vec.size() << " IMU datas."  << std::endl;
#endif

        // Step 1: Declare success flag for ICP
        // icp_success indicates if ICP alignment completes successfully (true by default)
        bool icp_success = true;

        // Step 2: Set up timers to measure performance (if debugging is enabled)
        // timer stores pairs of labels (e.g., "Initialization") and Stopwatch objects
#ifdef DEBUG
        std::vector<std::pair<std::string, std::unique_ptr<finalicp::Stopwatch<>>>> timer;
        // Add timers for different ICP phases (only if debug_print is true)
        timer.emplace_back("Update Transform ............... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("Association .................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("Optimization ................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("Alignment ...................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("Initialization ................. ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("Marginalization ................ ", std::make_unique<finalicp::Stopwatch<>>(false));
#endif

        // Step 3: Start the initialization timer (timer[4] = "Initialization")
        // Measures time taken to set up the SLAM trajectory
        // ######################################################################################
        // INITIALIZATION
        // ######################################################################################

#ifdef DEBUG
        if (!timer.empty()) timer[4].second->start();
#endif

        // Step 4: Create a new SLAM_TRAJ using singer::Interface
        // singer::Interface models the robot's trajectory (pose, velocity, etc.) over time
        // options_.qc_diag and ad_diag define noise models for trajectory dynamics
        auto SLAM_TRAJ = finalicp::traj::const_vel::Interface::MakeShared(options_.qc_diag);

        // Step 5: Initialize containers for state variables and cost terms
        // SLAM_STATE_VAR holds variables to optimize (pose, velocity, etc.)
        // Cost terms define errors for optimization (e.g., point cloud alignment, IMU)
        std::vector<finalicp::StateVarBase::Ptr> SLAM_STATE_VAR;
        std::vector<finalicp::BaseCostTerm::ConstPtr> prior_cost_terms; // Prior constraints
        std::vector<finalicp::BaseCostTerm::ConstPtr> meas_cost_terms; // Point cloud measurements
        std::vector<finalicp::BaseCostTerm::ConstPtr> imu_cost_terms; // IMU measurements
        // std::vector<finalicp::BaseCostTerm::ConstPtr> pose_meas_cost_terms; // Pose measurements
        std::vector<finalicp::BaseCostTerm::ConstPtr> imu_prior_cost_terms; // IMU bias priors
        // std::vector<finalicp::BaseCostTerm::ConstPtr> Ti2m_prior_cost_terms; // Ti2m priors

        // Step 6: Track indices for trajectory variables
        // prev_trajectory_var_index points to the last state in trajectory_vars_
        // curr_trajectory_var_index tracks new states added for this frame
        const size_t prev_trajectory_var_index = trajectory_vars_.size() - 1;   //??
        size_t curr_trajectory_var_index = trajectory_vars_.size() - 1;         //??

        // Step 7: Validate inputs and previous state
        // Ensure index_frame is valid and trajectory_vars_ is not empty
        
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
        math::se3::Transformation prev_Tm2b = PREV_VAR.Tm2b->value(); // Map-to-robot pose
        Eigen::Matrix<double, 6, 1> prev_wb2m_inr = PREV_VAR.wb2m_inr->value(); // Velocity
        Eigen::Matrix<double, 6, 1> prev_imu_biases = PREV_VAR.imu_biases->value(); // IMU biases
        math::se3::Transformation prev_Ti2m = PREV_VAR.Ti2m->value(); // IMU-to-map transformation

#ifdef DEBUG
    // [DEBUG] Check if the state from the previous frame is valid
    if (!prev_Tm2b.matrix().allFinite()) { std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: prev_Tm2b is NOT finite!" << std::endl; }
    if (!prev_wb2m_inr.allFinite()) { std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: prev_wb2m_inr is NOT finite!" << std::endl; }
#endif

        // Step 11: Validate previous state values
        // Ensure all state values are finite (not NaN or infinite)

        // Step 12: Get pointers to previous state variables
        // These are shared pointers to state objects for SLAM optimization
        const auto prev_Tm2b_var = PREV_VAR.Tm2b; // Pose variable
        const auto prev_wb2m_inr_var = PREV_VAR.wb2m_inr; // Velocity variable
        const auto prev_imu_biases_var = PREV_VAR.imu_biases; // IMU biases variable
        auto prev_Ti2m_var = PREV_VAR.Ti2m; // Ti2m variable (non-const for updates)

        // Step 14: Add previous state to SLAM trajectory
        // This anchors the trajectory at the previous frame’s end state
        SLAM_TRAJ->add(prev_slam_time, prev_Tm2b_var, prev_wb2m_inr_var);
#ifdef DEBUG
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: add prev_slam_time." << std::endl; 
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: add prev_Tm2b_var."  << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: add prev_wb2m_inr_var." << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: add prev_dwb2m_inr_var." << std::endl;
#endif

        // Step 15: Add previous state variables to optimization list
        // These variables will be optimized (if not locked) in ICP
        SLAM_STATE_VAR.emplace_back(prev_Tm2b_var); // Add pose
        SLAM_STATE_VAR.emplace_back(prev_wb2m_inr_var); // Add velocity

#ifdef DEBUG
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_STATE_VAR: emplace prev_Tm2b_var." << std::endl; 
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_STATE_VAR: emplace prev_wb2m_inr_var." << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_STATE_VAR: emplace prev_dwb2m_inr_var." << std::endl;
#endif

        // Step 16: Handle IMU-related state variables (if IMU is enabled)
        if (options_.use_imu) {
            // Add IMU biases to optimization (biases evolve over time)
            SLAM_STATE_VAR.emplace_back(prev_imu_biases_var);
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_STATE_VAR: emplace prev_imu_biases_var." << std::endl; 
#endif
            if ((!options_.Ti2m_init_only || index_frame == 1) && options_.use_accel) {
                SLAM_STATE_VAR.emplace_back(prev_Ti2m_var);
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_STATE_VAR: emplace prev_Ti2m_var." << std::endl; 
#endif
            }
        }

        ///################################################################################
        // Step 17: Get the current frame’s end timestamp
        // curr_time tells us when this frame ends
        const double CURR_TIME = trajectory_[index_frame].end_timestamp;

#ifdef DEBUG
        std::cout << std::fixed << std::setprecision(12) 
        << "[ICP DEBUG | Frame " << index_frame << "] " << "LOGGING: PREV_TIME: " << PREV_TIME << ", CURR_TIME: " << CURR_TIME << std::endl;
#endif

        // [DEBUG] THIS IS THE MOST LIKELY CULPRIT
        if (CURR_TIME <= PREV_TIME) {
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: Zero or negative time difference between frames!" << std::endl;
#endif
            return false;
        }

        // Step 18: Calculate the number of new states to add
        // num_extra_states is how many extra points (knots) to add between PREV_TIME and curr_time
        // +1 includes the mandatory end state at curr_time
        const int NUM_STATES = options_.num_extra_states + 1;
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Adding "<< NUM_STATES << " extra number of state between 2 original state." << std::endl;
#endif

        // Step 19: Create timestamps (knot times) for new states
        // knot_times lists when each new state occurs, from PREV_TIME to curr_time
        const double TIME_DIFF = (CURR_TIME - PREV_TIME) / static_cast<double>(NUM_STATES);

#ifdef DEBUG
        // [DEBUG] Check the calculated time difference
        std::cout << std::fixed << std::setprecision(12) << "[ICP DEBUG | Frame " << index_frame << "] " << "Time difference : " << TIME_DIFF << "s" << std::endl;
        if (!std::isfinite(TIME_DIFF) || TIME_DIFF <= 0) {
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: Invalid TIME_DIFF!" << std::endl;
        }
#endif

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
            T_NEXT_MAT = T_NEXT_MAT * options_.Tb2s;
        } else {
            // For early frames, use trajectory interpolation
            T_NEXT_MAT = SLAM_TRAJ->getPoseInterpolator(finalicp::traj::Time(KNOT_TIMES.back()))->value().inverse().matrix();
        }

#ifdef DEBUG
        // [DEBUG] Check the initial pose prediction for NaNs
        if (!T_NEXT_MAT.allFinite()) {
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: Extrapolated pose T_NEXT_MAT is NOT finite!" << std::endl;
        }
#endif

        const math::se3::Transformation T_NEXT(Eigen::Matrix4d(T_NEXT_MAT.inverse()));
        const Eigen::Matrix<double, 6, 1> w_NEXT = Eigen::Matrix<double, 6, 1>::Zero();
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

            const auto Tm2b_var = [&]() -> SE3StateVar::Ptr {
                if (options_.use_elastic_initialization) {
                    return SE3StateVar::MakeShared(T_NEXT);
                } else {
                    const Eigen::Matrix<double, 6, 1> xi_b2m_inr_odo((knot_slam_time - prev_slam_time).seconds() * prev_wb2m_inr);
                    const auto knot_Tm2b = math::se3::Transformation(xi_b2m_inr_odo) * prev_Tm2b;
                    return SE3StateVar::MakeShared(knot_Tm2b);
                }
            }();

            const auto wb2m_inr_var = [&]() -> VSpaceStateVar<6>::Ptr {
                if (options_.use_elastic_initialization) {
                    return VSpaceStateVar<6>::MakeShared(w_NEXT);
                } else {
                    return VSpaceStateVar<6>::MakeShared(prev_wb2m_inr);
                }
            }();

            const auto imu_biases_var = VSpaceStateVar<6>::MakeShared(prev_imu_biases);
            //
            SLAM_TRAJ->add(knot_slam_time, Tm2b_var, wb2m_inr_var);

#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: add Tm2b_var." << std::endl;
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: add wb2m_inr_var." << std::endl;
#endif

            SLAM_STATE_VAR.emplace_back(Tm2b_var);
            SLAM_STATE_VAR.emplace_back(wb2m_inr_var);

#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_STATE_VAR: emplace Tm2b_var." << std::endl; 
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_STATE_VAR: emplace wb2m_inr_var." << std::endl; 
#endif

            const auto Ti2m_var = SE3StateVar::MakeShared(prev_Ti2m);

            if (options_.use_imu) {
                    SLAM_STATE_VAR.emplace_back(imu_biases_var);
                if (options_.Ti2m_init_only || !options_.use_accel) {
                    Ti2m_var->locked() = true;
                } else {
                    SLAM_STATE_VAR.emplace_back(Ti2m_var);
                }
            }

            trajectory_vars_.emplace_back(knot_slam_time, Tm2b_var, wb2m_inr_var, imu_biases_var, Ti2m_var);
            // Update index for next state
            curr_trajectory_var_index++;
        }

        ///################################################################################

        // Step 24: Add prior cost terms for the initial frame (index_frame == 1)
        // Priors set initial guesses for pose, velocity, and acceleration to guide optimization
        if (index_frame == 1) {
            // Get the previous frame’s state variables
            const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

            // Define initial pose (Tm2b, identity), velocity (wb2m_inr, zero), and acceleration (dwb2m_inr, zero)
            math::se3::Transformation Tm2b; // Identity transformation (no initial offset)
            Eigen::Matrix<double, 6, 1> wb2m_inr = Eigen::Matrix<double, 6, 1>::Zero(); // Zero initial velocity
            Eigen::Matrix<double, 12, 12> state_cov = Eigen::Matrix<double, 12, 12>::Identity() * 1e-4;
            // Add prior cost terms to constrain initial state
            SLAM_TRAJ->addStatePrior(PREV_VAR.time, Tm2b, wb2m_inr, state_cov); // Constrain initial pose
            
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: addStatePrior." << std::endl;
#endif

            if (PREV_VAR.time != Time(trajectory_.at(0).end_timestamp)) throw std::runtime_error{"inconsistent timestamp"};
        }

        // Step 25: Add IMU-related prior cost terms (if IMU is enabled)
        if (options_.use_imu && index_frame == 1) {
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
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "imu_prior_cost_terms: Emplace bias_prior_factor." << std::endl;
#endif

            if (options_.use_accel) {  // For subsequent frames, add IMU bias prior if enabled
                Eigen::Matrix<double, 6, 6> init_Ti2m_cov = Eigen::Matrix<double, 6, 6>::Zero();
                init_Ti2m_cov.diagonal() = options_.Ti2m_init_cov;
                math::se3::Transformation Ti2m = PREV_VAR.Ti2m->value();
                auto Ti2m_error = se3_error(PREV_VAR.Ti2m, Ti2m);
                auto noise_model = StaticNoiseModel<6>::MakeShared(init_Ti2m_cov);
                auto loss_func = L2LossFunc::MakeShared();
                const auto Ti2m_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(Ti2m_error, noise_model, loss_func);
                imu_prior_cost_terms.emplace_back(Ti2m_prior_factor);
            }
        }
            // Step 26: Stop the initialization timer
            // Marks the end of the initialization phase (already handled in Step 18, included for completeness)
#ifdef DEBUG
        if (!timer.empty()) timer[4].second->stop();
#endif
        
        ///################################################################################
        // MARGINALIZATION
        ///################################################################################

        // Step 28: Update sliding window variables
        // Add state variables to the sliding window filter for optimization
        {
            // Step 27: Start the marginalization timer
            // Timer[5] measures the time taken to update the sliding window filter
#ifdef DEBUG
            if (!timer.empty()) timer[5].second->start();
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Update sliding window variables." << index_frame << std::endl;
#endif

            // For the initial frame, include the previous frame’s state variables
            if (index_frame == 1) {
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Apply a prior state variable on initial frame." << index_frame << std::endl;
#endif
                // Get the previous frame’s state variables
                const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

                // Add pose, velocity, to the sliding window
                sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.Tm2b, PREV_VAR.wb2m_inr});

                // If IMU is enabled, add IMU biases and optionally Ti2m
                if (options_.use_imu) {
#ifdef DEBUG
                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Apply an IMU bias variable on initial frame " << index_frame << std::endl;
#endif
                    sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.imu_biases});
                    if (options_.use_accel) {
#ifdef DEBUG
                        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Apply a prior Ti2m variable for initial frame " << index_frame << "as we are not using Ti2m_gt" << std::endl;
#endif
                        sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.Ti2m});
                    }
                }
            }

            // Add state variables for new states in the current frame
            for (size_t i = prev_trajectory_var_index + 1; i <= curr_trajectory_var_index; ++i) {
                // Get the current state’s variables
                const auto& VAR = trajectory_vars_.at(i);

                // Add pose, velocity, and acceleration to the sliding window
                sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{VAR.Tm2b, VAR.wb2m_inr});

                // If IMU is enabled, add IMU biases and optionally Ti2m
                if (options_.use_imu) {
                    sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{VAR.imu_biases});
                    if (!options_.Ti2m_init_only && options_.use_accel) {
                        sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{VAR.Ti2m});
                    }
                }
            }
        }

        // Step 29: Marginalize old state variables to keep the sliding window manageable
        // Remove states older than delay_adding_points frames ago
        if ((index_frame - options_.delay_adding_points) > 0) {
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Condition (index_frame > delay_adding_points) met. Entering marginalization." << std::endl;
#endif
            // Collect state variables to marginalize (from to_marginalize_ up to marg_time)
            std::vector<finalicp::StateVarBase::Ptr> MAG_VAR;
            int NUM_STATES = 0;
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
#ifdef DEBUG
                        // Check if the variables are valid *before* marginalizing them
                        if(!VAR.Tm2b->value().matrix().allFinite()) {
                           std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: VAR.Tm2b at index " << i << " is NaN before marginalization!" << std::endl;
                        }
#endif
                    // Add state variables to marginalize
                    MAG_VAR.emplace_back(VAR.Tm2b);
                    MAG_VAR.emplace_back(VAR.wb2m_inr);
                    if (options_.use_imu) {
                        MAG_VAR.emplace_back(VAR.imu_biases);
                        if (options_.use_accel) {
                            if (!VAR.Ti2m->locked()) {MAG_VAR.emplace_back(VAR.Ti2m);}
                        }
                    }
                    NUM_STATES++;
                } else {
                    // Update to_marginalize_ to the first non-marginalized state
                    to_marginalize_ = i;
                    break;
                }
            }
            // Marginalize the collected variables if any
            if (!MAG_VAR.empty()) {
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Collected " << NUM_STATES << " states to marginalize." << std::endl;
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Calling sliding_window_filter_->marginalizeVariable()" << std::endl;
#endif

                sliding_window_filter_->marginalizeVariable(MAG_VAR);
#ifdef DEBUG
                std::cout << std::fixed << std::setprecision(12) 
                << "[ICP DEBUG | Frame " << index_frame << "] " << "Marginalizing time: " << begin_marg_time - end_marg_time << ", with num states: " << NUM_STATES << std::endl;
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Finished marginalization call." << std::endl;
#endif
            }
            // Step 30: Stop the marginalization timer
#ifdef DEBUG
            if (!timer.empty()) timer[5].second->stop();
#endif
        }

        auto imu_options = PreintAccCostTerm::Options();
        imu_options.num_threads = options_.num_threads;
        imu_options.loss_sigma = options_.acc_loss_sigma;
        if (options_.acc_loss_func == "L2") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::L2;
        if (options_.acc_loss_func == "DCS") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::DCS;
        if (options_.acc_loss_func == "CAUCHY") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::CAUCHY;
        if (options_.acc_loss_func == "GM") imu_options.loss_func = PreintAccCostTerm::LOSS_FUNC::GM;
        imu_options.gravity(2, 0) = options_.gravity;
        imu_options.r_imu_acc = options_.r_imu_acc;

        auto gyro_options = GyroSuperCostTerm::Options();
        gyro_options.num_threads = options_.num_threads;
        if (options_.gyro_loss_func == "L2") gyro_options.gyro_loss_func = GyroSuperCostTerm::LOSS_FUNC::L2;
        if (options_.acc_loss_func == "DCS") gyro_options.gyro_loss_func = GyroSuperCostTerm::LOSS_FUNC::DCS;
        if (options_.acc_loss_func == "CAUCHY") gyro_options.gyro_loss_func = GyroSuperCostTerm::LOSS_FUNC::CAUCHY;
        if (options_.acc_loss_func == "GM") gyro_options.gyro_loss_func = GyroSuperCostTerm::LOSS_FUNC::GM;
        gyro_options.r_imu_ang = options_.r_imu_ang;
        gyro_options.gyro_loss_sigma = options_.gyro_loss_sigma;

        if (options_.use_imu) {
            for (size_t i = prev_trajectory_var_index; i < trajectory_vars_.size() - 1; i++) {
                const auto gyro_super_cost_term = GyroSuperCostTerm::MakeShared(
                                SLAM_TRAJ, trajectory_vars_[i].time, trajectory_vars_[i + 1].time, trajectory_vars_[i].imu_biases,
                                trajectory_vars_[i + 1].imu_biases, gyro_options);

                std::vector<finalicp::IMUData> data_vec;

                for (auto imu_data : imu_data_vec) {
                    if (imu_data.timestamp >= trajectory_vars_[i].time.seconds() && imu_data.timestamp < trajectory_vars_[i + 1].time.seconds()) {
                        data_vec.push_back(imu_data);
                    }
                }

                if (!data_vec.size()) {
                    continue;
                }

                gyro_super_cost_term->set(data_vec);
                gyro_super_cost_term->init();
                imu_cost_terms.push_back(gyro_super_cost_term);

                if (options_.use_accel) {
                    const auto preint_cost_term = PreintAccCostTerm::MakeShared(
                        SLAM_TRAJ, trajectory_vars_[i].time, trajectory_vars_[i + 1].time, trajectory_vars_[i].imu_biases,
                        trajectory_vars_[i + 1].imu_biases, trajectory_vars_[i].Ti2m,
                        trajectory_vars_[i + 1].Ti2m, imu_options);
                    preint_cost_term->set(data_vec);
                    preint_cost_term->init();
                    imu_cost_terms.push_back(preint_cost_term);
                }
            }

            {
                Eigen::Matrix<double, 6, 6> bias_cov = Eigen::Matrix<double, 6, 6>::Identity();
                bias_cov.block<3, 3>(0, 0).diagonal() = options_.q_bias_accel;
                bias_cov.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * options_.q_bias_gyro;
                auto noise_model = StaticNoiseModel<6>::MakeShared(bias_cov);
                auto loss_func = L2LossFunc::MakeShared();
                size_t i = prev_trajectory_var_index;
                for (; i < trajectory_vars_.size() - 1; i++) {
                    const auto nbk = finalicp::vspace::NegationEvaluator<6>::MakeShared(trajectory_vars_[i + 1].imu_biases);
                    auto bias_error = finalicp::vspace::AdditionEvaluator<6>::MakeShared(trajectory_vars_[i].imu_biases, nbk);
                    const auto bias_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(bias_error, noise_model, loss_func);
                    imu_prior_cost_terms.emplace_back(bias_prior_factor);
                }
            }

            if (!options_.Ti2m_init_only && options_.use_accel) {
                const auto Ti2m = math::se3::Transformation();
                Eigen::Matrix<double, 6, 6> Ti2m_cov = Eigen::Matrix<double, 6, 6>::Zero();
                Ti2m_cov.diagonal() = options_.qg_diag;
                auto noise_model = StaticNoiseModel<6>::MakeShared(Ti2m_cov);
                auto loss_func = L2LossFunc::MakeShared();
                size_t i = prev_trajectory_var_index;
                for (; i < trajectory_vars_.size() - 1; i++) {
                    auto Ti2m_error = se3_error(compose_rinv(trajectory_vars_[i + 1].Ti2m, trajectory_vars_[i].Ti2m), Ti2m);
                    const auto Ti2m_prior_factor = WeightedLeastSqCostTerm<6>::MakeShared(Ti2m_error, noise_model, loss_func);
                    imu_prior_cost_terms.emplace_back(Ti2m_prior_factor);
                }
            }
        }

        ///################################################################################
        // Step 31: Restart the initialization timer for query point evaluation
        // Timer[4] measures the time taken to process query points and IMU cost terms
#ifdef DEBUG
        if (!timer.empty()) timer[4].second->start();
#endif
        // Step 32: Collect unique timestamps from keypoints for query point evaluation
        // unique_point_times lists distinct timestamps to query the SLAM trajectory
        std::set<double> unique_point_times_set;
        for (const auto& keypoint : keypoints) {
            unique_point_times_set.insert(keypoint.timestamp);
        }
        std::vector<double> unique_point_times(unique_point_times_set.begin(), unique_point_times_set.end());
#ifdef DEBUG
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Found " << unique_point_times.size() << " unique timestamps in the point cloud." << std::endl;
#endif

        ///################################################################################

        // Step 36: Cache interpolation matrices for unique keypoint timestamps
        // interp_mats_ stores matrices (omega, lambda) for efficient pose interpolation
#ifdef DEBUG
        timer[0].second->start(); // Start update transform timer
#endif

        interp_mats_.clear(); // Clear previous interpolation matrices
        const double& time1 = prev_slam_time.seconds(); // Start time of the trajectory segment
        const double& time2 = KNOT_TIMES.back(); // End time of the trajectory segment
        const double T = time2 - time1; // Time duration

        const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones(); // Unit vector for covariance
        const auto Qinv_T = finalicp::traj::const_vel::getQinv(T, ones); // Inverse covariance matrix
        const auto Tran_T = finalicp::traj::const_vel::getTran(T); // Transition matrix

        // Compute interpolation matrices
        if (unique_point_times.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential: Process timestamps one by one for small sizes
            for (size_t i = 0; i < unique_point_times.size(); ++i) {
                const double time = unique_point_times[i];
                const double tau = time - time1; // Time offset from start
                const double kappa = time2 - time; // Time offset from end
                const Matrix12d Q_tau = finalicp::traj::const_vel::getQ(tau, ones); // Covariance at tau
                const Matrix12d Tran_kappa = finalicp::traj::const_vel::getTran(kappa); // Transition at kappa
                const Matrix12d Tran_tau = finalicp::traj::const_vel::getTran(tau); // Transition at tau
                const Matrix12d omega = Q_tau * Tran_kappa.transpose() * Qinv_T; // Interpolation matrix
                const Matrix12d lambda = Tran_tau - omega * Tran_T; // Interpolation matrix
                interp_mats_.emplace(time, std::make_pair(omega, lambda)); // Cache matrices
            }
        } else {
            // Parallel: Process timestamps concurrently with TBB
            tbb::concurrent_hash_map<double, std::pair<Matrix12d, Matrix12d>> temp_interp_mats;
            // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const double time = unique_point_times[i];
                        const double tau = time - time1; // Time offset from start
                        const double kappa = time2 - time; // Time offset from end
                        const Matrix12d Q_tau = finalicp::traj::const_vel::getQ(tau, ones); // Covariance at tau
                        const Matrix12d Tran_kappa = finalicp::traj::const_vel::getTran(kappa); // Transition at kappa
                        const Matrix12d Tran_tau = finalicp::traj::const_vel::getTran(tau); // Transition at tau
                        const Matrix12d omega = Q_tau * Tran_kappa.transpose() * Qinv_T; // Interpolation matrix
                        const Matrix12d lambda = Tran_tau - omega * Tran_T; // Interpolation matrix
                        tbb::concurrent_hash_map<double, std::pair<Matrix12d, Matrix12d>>::accessor acc;
                        temp_interp_mats.insert(acc, time);
                        acc->second = std::make_pair(omega, lambda); // Thread-safe insertion
                    }
                });

            // Transfer from concurrent_hash_map to interp_mats_ sequentially
            for (const auto& entry : temp_interp_mats) {
                interp_mats_.emplace(entry.first, entry.second);
            }
        }

#ifdef DEBUG
        timer[0].second->stop(); // Stop update transform timer
#endif

        // Step 37: Transform keypoints to the map frame using interpolated poses
        // Lambda function to map raw keypoints to the map frame
        auto transform_keypoints = [&]() {
            // Get state variables at the start and end knots
            const auto knot1 = SLAM_TRAJ->get(prev_slam_time);
            const auto knot2 = SLAM_TRAJ->get(finalicp::traj::Time(KNOT_TIMES.back()));
            const auto T1 = knot1->pose()->value(); // Start pose
            const auto w1 = knot1->velocity()->value(); // Start velocity
            const auto T2 = knot2->pose()->value(); // End pose
            const auto w2 = knot2->velocity()->value(); // End velocity

            // Compute relative transformation and Jacobians
            const auto xi_21 = (T2 / T1).vec(); // Relative pose vector
            const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21); // Inverse Jacobian
            const auto J_21_inv_w2 = J_21_inv * w2; // Transformed velocity

            // Step 37.1: Cache interpolated poses for unique timestamps
            // Computes and stores pose matrices (T_mr) for each timestamp in unique_point_times
            std::map<double, Eigen::Matrix4d> Tb2m_cache_map;

            if (unique_point_times.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // Sequential: Process timestamps one by one for small sizes
                for (size_t jj = 0; jj < unique_point_times.size(); ++jj) {
                    const double ts = unique_point_times[jj];
                    const auto& omega = interp_mats_.at(ts).first;
                    const auto& lambda = interp_mats_.at(ts).second;
                    // Compute interpolated pose vector
                    const Eigen::Matrix<double, 6, 1> xi_i1 = lambda.block<6, 6>(0, 6) * w1 + omega.block<6, 6>(0, 0) * xi_21 + omega.block<6, 6>(0, 6) * J_21_inv_w2;

                    const math::se3::Transformation T_i1(xi_i1); // Interpolated pose relative to T1
                    const math::se3::Transformation T_i0 = T_i1 * T1; // Pose in map frame
                    const Eigen::Matrix4d Tb2m = T_i0.inverse().matrix(); // Inverse pose matrix
                    Tb2m_cache_map[ts] = Tb2m; // Cache pose
                }
            } else {
                // Parallel: Process timestamps concurrently with TBB
                tbb::concurrent_hash_map<double, Eigen::Matrix4d> temp_cache_map;
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
                        for (size_t jj = range.begin(); jj != range.end(); ++jj) {
                            const double ts = unique_point_times[jj];
                            const auto& omega = interp_mats_.at(ts).first;
                            const auto& lambda = interp_mats_.at(ts).second;
                            // Compute interpolated pose vector
                            const Eigen::Matrix<double, 6, 1> xi_i1 = lambda.block<6, 6>(0, 6) * w1 + omega.block<6, 6>(0, 0) * xi_21 + omega.block<6, 6>(0, 6) * J_21_inv_w2;
                            const math::se3::Transformation T_i1(xi_i1); // Interpolated pose relative to T1
                            const math::se3::Transformation T_i0 = T_i1 * T1; // Pose in map frame
                            const Eigen::Matrix4d Tb2m = T_i0.inverse().matrix(); // Inverse pose matrix
                            tbb::concurrent_hash_map<double, Eigen::Matrix4d>::accessor acc;
                            temp_cache_map.insert(acc, ts);
                            acc->second = Tb2m; // Thread-safe insertion
                        }
                    });

                // Transfer from concurrent_hash_map to T_mr_cache_map sequentially
                for (const auto& entry : temp_cache_map) {
                    Tb2m_cache_map[entry.first] = entry.second;
                }
            }

            // Step 37.2: Transform keypoints to the map frame
            // Applies cached pose matrices (T_mr) to transform raw keypoint coordinates to the map frame
            if (keypoints.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // Sequential: Transform keypoints one by one for small sizes
                for (size_t jj = 0; jj < keypoints.size(); ++jj) {
                    auto& keypoint = keypoints[jj];
                    const Eigen::Matrix4d& Tb2m = Tb2m_cache_map.at(keypoint.timestamp);
                    keypoint.pt = Tb2m.block<3, 3>(0, 0) * keypoint.raw_pt + Tb2m.block<3, 1>(0, 3); // Transform raw point
                }
            } else {
                // Parallel: Transform keypoints concurrently with TBB
                tbb::concurrent_vector<Point3D> temp_keypoints(keypoints.size()); // Concurrent storage for transformed keypoints
                // Initialize temp_keypoints sequentially to preserve other keypoint fields
                for (size_t jj = 0; jj < keypoints.size(); ++jj) {
                    temp_keypoints[jj] = keypoints[jj];
                }
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
                        for (size_t jj = range.begin(); jj != range.end(); ++jj) {
                            auto& keypoint = temp_keypoints[jj];
                            const Eigen::Matrix4d& Tb2m = Tb2m_cache_map.at(keypoint.timestamp);
                            keypoint.pt = Tb2m.block<3, 3>(0, 0) * keypoint.raw_pt + Tb2m.block<3, 1>(0, 3); // Transform raw point
                        }
                    });

                // Assign temporary keypoints back to keypoints sequentially
                for (size_t jj = 0; jj < temp_keypoints.size(); ++jj) {
                    keypoints[jj] = temp_keypoints[jj];
                }
            }
        };

#define USE_P2P_SUPER_COST_TERM true

        // Step 35: Configure voxel visitation settings
        // Determine how many neighboring voxels to visit along each axis
        const short nb_voxels_visited = index_frame < options_.init_num_frames ? 2 : 1; // More neighbors for early frames
        const int kMinNumNeighbors = options_.min_number_neighbors; // Minimum neighbors for point-to-plane alignment
        auto &current_estimate = trajectory_.at(index_frame);

        // Step 38: Initialize point-to-plane super cost term
        // Constrains the trajectory to align keypoints with the map
        finalicp::P2PCVSuperCostTerm::Options p2p_options;
        // p2p_options.sequential_threshold = options_.sequential_threshold;
        p2p_options.num_threads = options_.num_threads; // Thread count (sequential here, but set for compatibility)
        p2p_options.p2p_loss_sigma = options_.p2p_loss_sigma; // Loss parameter for point-to-plane
        // Map loss function to enum
        switch (options_.p2p_loss_func) {
            case stateestimate::lidarodom::LOSS_FUNC::L2:
                p2p_options.p2p_loss_func = finalicp::P2PCVSuperCostTerm::LOSS_FUNC::L2;
                break;
            case stateestimate::lidarodom::LOSS_FUNC::DCS:
                p2p_options.p2p_loss_func = finalicp::P2PCVSuperCostTerm::LOSS_FUNC::DCS;
                break;
            case stateestimate::lidarodom::LOSS_FUNC::CAUCHY:
                p2p_options.p2p_loss_func = finalicp::P2PCVSuperCostTerm::LOSS_FUNC::CAUCHY;
                break;
            case stateestimate::lidarodom::LOSS_FUNC::GM:
                p2p_options.p2p_loss_func = finalicp::P2PCVSuperCostTerm::LOSS_FUNC::GM;
                break;
            default:
                p2p_options.p2p_loss_func = finalicp::P2PCVSuperCostTerm::LOSS_FUNC::L2;
        }
        const auto p2p_super_cost_term = finalicp::P2PCVSuperCostTerm::MakeShared(SLAM_TRAJ, prev_slam_time, finalicp::traj::Time(KNOT_TIMES.back()), p2p_options);

#ifdef DEBUG
        // Step 39: Stop the initialization timer
        if (!timer.empty()) timer[4].second->stop();
#endif

        ///################################################################################

        // Step 40: Transform keypoints to the robot frame (if using point-to-plane super cost term)
        // Applies the inverse sensor-to-robot transformation (T_rs) to raw keypoint coordinates
#if USE_P2P_SUPER_COST_TERM
#ifdef DEBUG
            timer[0].second->start(); // Start update transform timer
#endif
            // #### This just transform the point from sensor to robot frame
            // sensor to robot frame is identity!
            const Eigen::Matrix4d Ts2b_mat = options_.Tb2s.inverse(); // Inverse sensor-to-robot transformation
            
            if (keypoints.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // Sequential: Transform keypoints one by one for small sizes
                for (size_t i = 0; i < keypoints.size(); ++i) {
                    auto& keypoint = keypoints[i];
                    keypoint.raw_pt = Ts2b_mat.block<3, 3>(0, 0) * keypoint.raw_pt + Ts2b_mat.block<3, 1>(0, 3); // Transform raw point
                }
            } else {
                // Parallel: Transform keypoints concurrently with TBB
                tbb::concurrent_vector<Point3D> temp_keypoints(keypoints.size()); // Concurrent storage for transformed keypoints
                // Initialize temp_keypoints sequentially to preserve other keypoint fields
                for (size_t i = 0; i < keypoints.size(); ++i) {
                    temp_keypoints[i] = keypoints[i];
                }
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            auto& keypoint = temp_keypoints[i];
                            keypoint.raw_pt = Ts2b_mat.block<3, 3>(0, 0) * keypoint.raw_pt + Ts2b_mat.block<3, 1>(0, 3); // Transform raw point
                        }
                    });

                // Assign temporary keypoints back to keypoints sequentially
                keypoints.resize(temp_keypoints.size());
                for (size_t i = 0; i < temp_keypoints.size(); ++i) {
                    keypoints[i] = temp_keypoints[i];
                }
            }

#ifdef  DEBUG
            timer[0].second->stop(); // Stop update transform timer
#endif
#endif

        // Step 41: Initialize the current frame’s pose estimate
        // Computes begin and end poses, velocities, and accelerations for the frame
        auto& p2p_matches = p2p_super_cost_term->get(); // Get point-to-plane matches
        p2p_matches.clear(); // Clear previous matches
        int N_matches = 0; // Track number of matches

        Eigen::Matrix<double, 6, 1> v_begin = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> v_end = Eigen::Matrix<double, 6, 1>::Zero();

        bool swf_inside_icp = false;  // kitti-raw : false
        if (index_frame > options_.init_num_frames || options_.swf_inside_icp_at_begin) {
            swf_inside_icp = true;
        }

        const auto p2p_loss_func = [this]() -> BaseLossFunc::Ptr {
            switch (options_.p2p_loss_func) {
                case LOSS_FUNC::L2: return L2LossFunc::MakeShared();
                case LOSS_FUNC::DCS: return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
                case LOSS_FUNC::CAUCHY:return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
                case LOSS_FUNC::GM: return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
                default:
                return nullptr;
            }
            return nullptr;
        }();

        // ################################################################################
        // Step 43: Start ICP optimization loop ################################################################################
        // ################################################################################
        // Iterates to refine the trajectory using point-to-plane alignment
        for (int iter = 0; iter < options_.num_iters_icp; iter++) {
#ifdef DEBUG
            // [DEBUG] Start of an ICP iteration
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "--- Iteration " << iter << " ---" << std::endl;
#endif 
#ifdef DEBUG
            timer[0].second->start();
#endif 
            transform_keypoints();
#ifdef DEBUG
            timer[0].second->stop();
#endif 

            // Initialize optimization problem based on swf_inside_icp
            const auto problem = [&]() -> finalicp::Problem::Ptr {
                if (swf_inside_icp) {
#ifdef DEBUG
                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "swf_inside_icp is true." << std::endl; 
                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "problem: use SlidingWindowFilter." << std::endl; 
#endif
                    // Use SlidingWindowFilter for sliding window optimization
                    return std::make_shared<finalicp::SlidingWindowFilter>(*sliding_window_filter_);
                } else {
                    // Use OptimizationProblem for full state optimization
                    auto problem = finalicp::OptimizationProblem::MakeShared(options_.num_threads);
                    for (const auto& var : SLAM_STATE_VAR) {
                        problem->addStateVariable(var);
#ifdef DEBUG
                        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "problem: use OptimizationProblem addStateVariable: " << var << std::endl; 
#endif
                    }
                    return problem;
                }
            }();

            // Add prior cost terms to the problem
            SLAM_TRAJ->addPriorCostTerms(*problem);
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: addPriorCostTerms problem: " << std::endl;
#endif
            for (const auto& prior_cost_term : prior_cost_terms) {
                problem->addCostTerm(prior_cost_term);
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "problem: addCostTerm prior_cost_term: " << std::endl;
#endif
            }

            // Step 44: Clear measurement cost terms and prepare for association
#ifdef DEBUG
            timer[1].second->start(); // Start association timer
#endif
            meas_cost_terms.clear(); // Clear previous measurement cost terms
            p2p_matches.clear(); // Clear previous point-to-plane matches

#if USE_P2P_SUPER_COST_TERM
                p2p_matches.reserve(keypoints.size()); // Reserve for new matches
#else
                meas_cost_terms.reserve(keypoints.size()); // Reserve for new cost terms
#endif

#ifdef DEBUG
            // [DEBUG] Check if keypoint coordinates are finite before association
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Keypoint size for association: " << keypoints.size() << std::endl;
            bool keypoints_are_finite = true;
            for (size_t i = 0; i < keypoints.size(); ++i) {
                if (!keypoints[i].pt.allFinite()) {
                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: Keypoint " << i << " coordinate is NOT finite before association!" << std::endl;
                    keypoints_are_finite = false;
                    break;
                }
            }
            if (keypoints_are_finite) {
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "All keypoint coordinates are finite before association." << std::endl;
            }
#endif
            ///################################################################################

            // HYBRID STRATEGY: Use sequential processing for small workloads to avoid parallel overhead.
            // Note: Add 'sequential_threshold' to your options struct to control this behavior.

            if (keypoints.size() < static_cast<size_t>(options_.sequential_threshold)) {
                // --- SEQUENTIAL PATH ---
                for (int i = 0; i < (int)keypoints.size(); i++) {
                    const auto &keypoint = keypoints[i];
                    const auto &pt_keypoint = keypoint.pt;

                    ArrayVector3d vector_neighbors =
                        map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

                    if (static_cast<int>(vector_neighbors.size()) >= kMinNumNeighbors) {
                        auto neighborhood = compute_neighborhood_distribution(vector_neighbors, options_.sequential_threshold);
                        const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
                        const double weight = planarity_weight;
                        Eigen::Vector3d closest_pt = vector_neighbors[0];
                        const double dist_to_plane = std::abs((keypoint.pt - vector_neighbors[0]).transpose() * neighborhood.normal);
#ifdef DEBUG
                        if (i == 0) {
                            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Association for point 0:" << std::endl;
                            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Point coordinate: " << pt_keypoint.transpose() << std::endl;
                            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Neighbors found: " << vector_neighbors.size() << std::endl;
                            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Neighborhood a2D: " << neighborhood.a2D << std::endl;
                            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Dist to plane: " << dist_to_plane << std::endl;
                            if (!std::isfinite(neighborhood.a2D) || !std::isfinite(dist_to_plane)) {
                                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: NaN detected in neighborhood/distance calculation!" << std::endl;
                            }
                        }
#endif
                        if (dist_to_plane < options_.p2p_max_dist) {continue;}
#if USE_P2P_SUPER_COST_TERM
                        Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                        p2p_matches.emplace_back(P2PMatch(keypoint.timestamp, closest_pt, closest_normal, keypoint.raw_pt));
#endif
                    } else if (options_.use_pointtopoint_factors && vector_neighbors.size()){
                        if ((keypoint.pt - vector_neighbors[0]).norm() >= options_.p2p_max_dist) {continue;}
                        Eigen::Vector3d closest_pt = vector_neighbors[0];
                        const auto noise_model = StaticNoiseModel<3>::MakeShared(Eigen::Matrix3d::Identity());
                        const auto Tm2b_intp_eval = SLAM_TRAJ->getPoseInterpolator(Time(keypoint.timestamp));
                        const auto Tb2m_intp_eval = InverseEvaluator::MakeShared(Tm2b_intp_eval);
                        const auto error_func = p2p::p2pError(Tb2m_intp_eval, closest_pt, keypoint.raw_pt);
                        error_func->setTime(Time(keypoint.timestamp));
                        const auto cost = WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, p2p_loss_func);
                        meas_cost_terms.emplace_back(cost);
                    }
                } 
            } else {
                // --- PARALLEL PATH ---
                // Use TBB to process keypoints in parallel, distributing the workload across threads.
                // Results are collected into thread-safe containers to avoid race conditions.
                tbb::concurrent_vector<P2PMatch> local_p2p_matches;
                tbb::concurrent_vector<finalicp::BaseCostTerm::ConstPtr> local_meas_cost_terms;

                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            const auto &keypoint = keypoints[i];
                            const auto &pt_keypoint = keypoint.pt;
                            ArrayVector3d vector_neighbors =
                                map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);
                            if (static_cast<int>(vector_neighbors.size()) >= kMinNumNeighbors) {
                                auto neighborhood = compute_neighborhood_distribution(vector_neighbors, options_.sequential_threshold);
                                const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
                                const double weight = planarity_weight;
                                Eigen::Vector3d closest_pt = vector_neighbors[0];
                                const double dist_to_plane = std::abs((keypoint.pt - vector_neighbors[0]).transpose() * neighborhood.normal);
#ifdef DEBUG
                                // Debug output only for the first thread to avoid excessive logging
                                if (i == range.begin()) {
                                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Parallel association for point " << i << ":" << std::endl;
                                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Point coordinate: " << pt_keypoint.transpose() << std::endl;
                                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Neighbors found: " << vector_neighbors.size() << std::endl;
                                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Neighborhood a2D: " << neighborhood.a2D << std::endl;
                                    std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Dist to plane: " << dist_to_plane << std::endl;
                                    if (!std::isfinite(neighborhood.a2D) || !std::isfinite(dist_to_plane)) {
                                        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: NaN detected in neighborhood/distance calculation!" << std::endl;
                                    }
                                }
#endif
                                if (dist_to_plane < options_.p2p_max_dist) {
                                    continue;
                                }
#if USE_P2P_SUPER_COST_TERM
                                Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                                local_p2p_matches.emplace_back(P2PMatch(keypoint.timestamp, closest_pt, closest_normal, keypoint.raw_pt));
#endif
                            } else if (options_.use_pointtopoint_factors && vector_neighbors.size()) {
                                if ((keypoint.pt - vector_neighbors[0]).norm() >= options_.p2p_max_dist) {
                                    continue;
                                }
                                Eigen::Vector3d closest_pt = vector_neighbors[0];
                                const auto noise_model = StaticNoiseModel<3>::MakeShared(Eigen::Matrix3d::Identity());
                                const auto Tm2b_intp_eval = SLAM_TRAJ->getPoseInterpolator(Time(keypoint.timestamp));
                                const auto Tb2m_intp_eval = InverseEvaluator::MakeShared(Tm2b_intp_eval);
                                const auto error_func = p2p::p2pError(Tb2m_intp_eval, closest_pt, keypoint.raw_pt);
                                error_func->setTime(Time(keypoint.timestamp));
                                const auto cost = WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, p2p_loss_func);
                                local_meas_cost_terms.emplace_back(cost);
                            }
                        }
                    }
                );

                // Copy results from concurrent containers to the final containers
                p2p_matches.insert(p2p_matches.end(), local_p2p_matches.begin(), local_p2p_matches.end());
                meas_cost_terms.insert(meas_cost_terms.end(), local_meas_cost_terms.begin(), local_meas_cost_terms.end());
            }

#if USE_P2P_SUPER_COST_TERM
            N_matches = p2p_matches.size();
#else
            N_matches = meas_cost_terms.size();
#endif

#ifdef DEBUG
            // [ADDED DEBUG] Print the number of matches found before checking
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Found " << N_matches << " point-to-plane matches." << std::endl;
#endif

            p2p_super_cost_term->initP2PMatches();

#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "problem: add meas_cost_terms total: " << meas_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : meas_cost_terms) {
                problem->addCostTerm(cost); // Add point-to-plane cost terms | this is only if not using p2psupercostterm
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "problem: add imu_cost_terms total: " << imu_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : imu_cost_terms) {
                problem->addCostTerm(cost); // Add IMU cost terms
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "problem: add imu_prior_cost_terms total: " << imu_prior_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : imu_prior_cost_terms) {
                problem->addCostTerm(cost); // Add IMU bias prior cost terms
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "problem: add p2p_super_cost_term." << std::endl;
#endif
            problem->addCostTerm(p2p_super_cost_term); // Add point-to-plane super cost term

#ifdef DEBUG
            timer[1].second->stop(); // Stop association timer
#endif

            // Step 46: Check for sufficient keypoints
            // Ensures enough matches for reliable optimization
            if (N_matches < options_.min_number_keypoints) {
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: not enough keypoints selected in icp !" << std::endl;
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Found: " << N_matches << " point-to-plane matches (residuals)." << std::endl;
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Minimum required: " << options_.min_number_keypoints << std::endl;
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Map size: " << map_.size() << " points." << std::endl;
#endif
                icp_success = false;
                break; // Exit the ICP loop if insufficient keypoints
            }

#ifdef DEBUG
            timer[2].second->start(); // Start optimization timer
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Calling solver.optimize()... Number of variables: " << problem->getStateVector()->getNumberOfStates() << ", Number of cost terms: " << problem->getNumberOfCostTerms() << std::endl;
#endif

            finalicp::GaussNewtonSolverNVA::Params params;
            params.verbose = options_.verbose;
            params.max_iterations = static_cast<unsigned int>(options_.max_iterations);
            params.line_search = (iter >= 2 && options_.use_line_search);
            if (swf_inside_icp) {params.reuse_previous_pattern = false;}
            finalicp::GaussNewtonSolverNVA solver(*problem, params);    
            // --- WRAP SOLVER CALL IN A TRY-CATCH BLOCK ---
            try {
                solver.optimize();
            } catch (const finalicp::decomp_failure& e) {
#ifdef DEBUG
                std::cerr << "[ICP DEBUG | Frame " << index_frame << "] " << "CATASTROPHIC SOLVER FAILURE: " << e.what() << std::endl;
                std::cerr << "[ICP DEBUG | Frame " << index_frame << "] " << "This usually means the Hessian matrix is not positive-definite, likely due to an ill-conditioned problem (e.g., bad geometry, insufficient constraints/priors)." << std::endl;
#endif
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::cerr << "[ICP DEBUG | Frame " << index_frame << "] " << "AN UNEXPECTED EXCEPTION OCCURRED DURING SOLVER::OPTIMIZE: " << e.what() << std::endl;
#endif
            }
#ifdef DEBUG
            timer[2].second->stop(); // Stop optimization timer
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Solver finished." << std::endl;
#endif

#ifdef DEBUG
            timer[3].second->start(); // Start alignment timer
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Updating State & Checking Convergence" << std::endl;
#endif

#ifdef DEBUG
            // [ADDED DEBUG] Header for this block to show the current iteration
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Updating State & Checking Convergence (Iteration " << iter << ")" << std::endl;
#endif

            double diff_trans = 0.0, diff_rot = 0.0, diff_vel = 0.0, diff_acc = 0.0;

            // Update begin pose
            finalicp::traj::Time curr_begin_slam_time(static_cast<double>(trajectory_[index_frame].begin_timestamp));
            const Eigen::Matrix4d begin_Tb2m = inverse(SLAM_TRAJ->getPoseInterpolator(curr_begin_slam_time))->evaluate().matrix();
            const Eigen::Matrix4d begin_Ts2m = begin_Tb2m * options_.Tb2s.inverse();
            diff_trans += (current_estimate.begin_t - begin_Ts2m.block<3, 1>(0, 3)).norm();
            diff_rot += AngularDistance(current_estimate.begin_R, begin_Ts2m.block<3, 3>(0, 0));

#ifdef DEBUG
            // [ADDED DEBUG] Print the change in the beginning pose translation
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Begin Translation | Old: " << current_estimate.begin_t.transpose()
                    << " | New: " << begin_Ts2m.block<3, 1>(0, 3).transpose() << std::endl;
#endif

            // Update end pose
            finalicp::traj::Time curr_end_slam_time(static_cast<double>(trajectory_[index_frame].end_timestamp));
            const Eigen::Matrix4d end_Tb2m = inverse(SLAM_TRAJ->getPoseInterpolator(curr_end_slam_time))->evaluate().matrix();
            const Eigen::Matrix4d end_Ts2m = end_Tb2m * options_.Tb2s.inverse();
            diff_trans += (current_estimate.end_t - end_Ts2m.block<3, 1>(0, 3)).norm();
            diff_rot += AngularDistance(current_estimate.end_R, end_Ts2m.block<3, 3>(0, 0));

#ifdef DEBUG
            // [ADDED DEBUG] Print the change in the ending pose translation
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "End Translation   | Old: " << current_estimate.end_t.transpose()
                    << " | New: " << end_Ts2m.block<3, 1>(0, 3).transpose() << std::endl;
#endif

            // Update velocities
            const auto vb = SLAM_TRAJ->getVelocityInterpolator(curr_begin_slam_time)->value();
            const auto ve = SLAM_TRAJ->getVelocityInterpolator(curr_end_slam_time)->value();
            diff_vel += (vb - v_begin).norm();
            diff_vel += (ve - v_end).norm();
            v_begin = vb;
            v_end = ve;

            // Update mid pose
            finalicp::traj::Time curr_mid_slam_time(static_cast<double>(trajectory_[index_frame].getEvalTime()));
            const Eigen::Matrix4d mid_Tb2m = inverse(SLAM_TRAJ->getPoseInterpolator(curr_mid_slam_time))->evaluate().matrix();
            const Eigen::Matrix4d mid_Ts2m = mid_Tb2m * options_.Tb2s.inverse();
            current_estimate.setMidPose(mid_Ts2m);

            // Update current estimate
            current_estimate.begin_R = begin_Ts2m.block<3, 3>(0, 0);
            current_estimate.begin_t = begin_Ts2m.block<3, 1>(0, 3);
            current_estimate.end_R = end_Ts2m.block<3, 3>(0, 0);
            current_estimate.end_t = end_Ts2m.block<3, 1>(0, 3);

            // --- [ADD DEBUG CHECKS AFTER CALCULATING DIFFS] ---
#ifdef DEBUG
            if (!std::isfinite(diff_rot) || !std::isfinite(diff_trans) || !std::isfinite(diff_vel) || !std::isfinite(diff_acc)) {
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "CRITICAL: Non-finite difference detected after optimization! The state is likely corrupted with NaNs." << std::endl;
            }
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "State Change   | d_rot: " << diff_rot << ", d_trans: " << diff_trans << ", d_vel: " << diff_vel << ", d_acc: " << diff_acc << std::endl;
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "End Pose (t)   | " << current_estimate.end_t.transpose() << std::endl;
#endif

#ifdef DEBUG
            timer[3].second->stop(); // Stop alignment timer
#endif

            // Check convergence
            if ((index_frame > 1) &&
                (diff_rot < options_.threshold_orientation_norm &&
                diff_trans < options_.threshold_translation_norm &&
                diff_vel < options_.threshold_translation_norm * 10.0 + options_.threshold_orientation_norm * 10.0)){
#ifdef DEBUG
                std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Finished with N=" << iter << " ICP iterations" << std::endl;
#endif
                if (options_.break_icp_early) {
                    break; // Exit loop if converged and early breaking is enabled
                }
            }
        // ################################################################################
        } // End ICP optimization loop ################################################################################
        // ################################################################################

        // Step 49: Add cost terms to the sliding window filter
        // Includes state priors, point-to-plane, IMU, pose, and Ti2m cost terms
        SLAM_TRAJ->addPriorCostTerms(*sliding_window_filter_); // Add state priors (e.g., for initial state x_0)

#ifdef DEBUG
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "SLAM_TRAJ: addPriorCostTerms with sliding_window_filter_" << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "sliding_window_filter_: add prior_cost_terms total: " << prior_cost_terms.size() << std::endl;
#endif
        for (const auto& prior_cost_term : prior_cost_terms) {
            sliding_window_filter_->addCostTerm(prior_cost_term); // Add prior cost terms | not really adding much
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "sliding_window_filter_: add meas_cost_terms total: " << meas_cost_terms.size() << std::endl;
#endif
        for (const auto& meas_cost_term : meas_cost_terms) {
            sliding_window_filter_->addCostTerm(meas_cost_term); // Add point-to-plane cost terms | this is only if not using p2psupercostterm
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "sliding_window_filter_: add imu_cost_terms total: " << imu_cost_terms.size() << std::endl;
#endif
        for (const auto& imu_cost : imu_cost_terms) {
            sliding_window_filter_->addCostTerm(imu_cost); // Add IMU cost terms
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "sliding_window_filter_: add p2p_super_cost_term total." << std::endl;
#endif
        sliding_window_filter_->addCostTerm(p2p_super_cost_term); // Add point-to-plane super cost term
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "sliding_window_filter_: add imu_prior_cost_terms total: " << imu_prior_cost_terms.size() << std::endl;
#endif
        for (const auto& imu_prior_cost : imu_prior_cost_terms) {
            sliding_window_filter_->addCostTerm(imu_prior_cost); // Add IMU bias prior cost terms
        }
#ifdef DEBUG
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "sliding_window_filter_: number of variables: " << sliding_window_filter_->getNumberOfVariables() << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "sliding_window_filter_: number of cost terms: " << sliding_window_filter_->getNumberOfCostTerms() << std::endl;
#endif

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
        params.max_iterations = 20;
        params.reuse_previous_pattern = false;
        finalicp::GaussNewtonSolverNVA solver(*sliding_window_filter_, params);
        if (!swf_inside_icp) {
            solver.optimize(); // Optimize the sliding window filter if not done in ICP loop
        }

        if (options_.Ti2m_init_only && options_.use_accel) {
            size_t i = prev_trajectory_var_index + 1;
            for (; i < trajectory_vars_.size(); i++) {
                trajectory_vars_[i].Ti2m = finalicp::se3::SE3StateVar::MakeShared(prev_Ti2m_var->value());
                trajectory_vars_[i].Ti2m->locked() = true;
            }
        }

        // Update begin pose
        finalicp::traj::Time curr_begin_slam_time(static_cast<double>(current_estimate.begin_timestamp));
        const Eigen::Matrix4d curr_begin_Tb2m = inverse(SLAM_TRAJ->getPoseInterpolator(curr_begin_slam_time))->evaluate().matrix();
        const Eigen::Matrix4d curr_begin_Ts2m = curr_begin_Tb2m * options_.Tb2s.inverse();

        // Update end pose
        finalicp::traj::Time curr_end_slam_time(static_cast<double>(current_estimate.end_timestamp));
        const Eigen::Matrix4d curr_end_Tb2m = inverse(SLAM_TRAJ->getPoseInterpolator(curr_end_slam_time))->evaluate().matrix();
        const Eigen::Matrix4d curr_end_Ts2m = curr_end_Tb2m * options_.Tb2s.inverse();

        finalicp::traj::Time curr_mid_slam_time(static_cast<double>(trajectory_[index_frame].getEvalTime()));
        const Eigen::Matrix4d mid_Tb2m = inverse(SLAM_TRAJ->getPoseInterpolator(curr_mid_slam_time))->evaluate().matrix();
        const Eigen::Matrix4d mid_Ts2m = mid_Tb2m * options_.Tb2s.inverse();
        current_estimate.setMidPose(mid_Ts2m);

        // Update current estimate
        current_estimate.begin_R = curr_begin_Ts2m.block<3, 3>(0, 0);
        current_estimate.begin_t = curr_begin_Ts2m.block<3, 1>(0, 3);
        current_estimate.end_R = curr_end_Ts2m.block<3, 3>(0, 0);
        current_estimate.end_t = curr_end_Ts2m.block<3, 1>(0, 3);

        // Update debug fields (for plotting)
        current_estimate.mid_w = SLAM_TRAJ->getVelocityInterpolator(curr_mid_slam_time)->value();
        // finalicp::Covariance covariance(solver);
        // current_estimate.mid_state_cov.block<12, 12>(0, 0) = SLAM_TRAJ->getCovariance(covariance, trajectory_vars_[prev_trajectory_var_index].time);
    
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
            if (curr_mid_slam_time.seconds() < trajectory_vars_[i].time.seconds() ||
                curr_mid_slam_time.seconds() >= trajectory_vars_[i + 1].time.seconds()) {
                throw std::runtime_error("Mid time not within knot times in icp: " + std::to_string(curr_mid_slam_time.seconds()) + " at frame: " + std::to_string(index_frame));
            }

            const auto bias_intp_eval = finalicp::vspace::VSpaceInterpolator<6>::MakeShared(curr_mid_slam_time, trajectory_vars_[i].imu_biases, trajectory_vars_[i].time, trajectory_vars_[i + 1].imu_biases, trajectory_vars_[i + 1].time);
            current_estimate.mid_b = bias_intp_eval->value();
#ifdef DEBUG
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "mid_Ti2m:\n" << current_estimate.mid_Ti2m << std::endl;
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "b_begin: " << trajectory_vars_[i].imu_biases->value().transpose() << std::endl;
            std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "b_end: " << trajectory_vars_[i + 1].imu_biases->value().transpose() << std::endl;
#endif
        }

        const auto w = SLAM_TRAJ->getVelocityInterpolator(curr_end_slam_time)->evaluate();

                // Step 54: Validate final estimate parameters
        // Ensures keypoints, velocities, and accelerations are valid
#ifdef DEBUG
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "ESTIMATED PARAMETER" << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Number of keypoints used in CT-ICP : " << N_matches << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "v_begin: " << v_begin.transpose() << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "v_end: " << v_end.transpose() << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "w: " << w.transpose() << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Number iterations CT-ICP : " << options_.num_iters_icp << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Translation Begin: " << trajectory_[index_frame].begin_t.transpose() << std::endl;
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Translation End: " << trajectory_[index_frame].end_t.transpose() << std::endl;
#endif

#ifdef DEBUG
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "INNER LOOP TIMERS" << std::endl;
        for (size_t i = 0; i < timer.size(); i++) {
            std::cout << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
        }
        // [DEBUG] Final status report before returning
        std::cout << "[ICP DEBUG | Frame " << index_frame << "] " << "Finished ICP for frame " << index_frame << ". Success: " << (icp_success ? "true" : "false") << std::endl;
#endif

        return icp_success;
    }
}