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

    void lidarinertialodom::sub_sample_frame(std::vector<Point3D>& frame, double size_voxel, int sequential_threshold) {
        if (frame.empty()) return;

#ifdef DEBUG
        // // [DEBUG] Log entry into the function and initial state
        // const size_t initial_size = frame.size();
        // std::cout << "[SUB_SAMPLE] Starting sub_sample_frame. Initial points: " << initial_size 
        //       << ", Voxel Size: " << size_voxel << std::endl;
#endif

        using VoxelMap = tsl::robin_map<Voxel, Point3D, VoxelHash>;

        // Step 1: Build the downsampled voxel map
        VoxelMap voxel_map;
        if (frame.size() < static_cast<size_t>(sequential_threshold)) {
#ifdef DEBUG
        // std::cout << "[SUB_SAMPLE] Using sequential path (size < " << sequential_threshold << ")." << std::endl;
#endif
            // Use a simple sequential path for small frames
            voxel_map.reserve(frame.size() / 2);
            for (const auto& point : frame) {
                // [DEBUG] Add a check for non-finite points which can corrupt voxel calculation
                if (!point.pt.allFinite()) {
#ifdef DEBUG
                    std::cout << "[SUB_SAMPLE] WARNING: Skipping non-finite point during sequential voxelization." << std::endl;
#endif
                    continue;
                }

                Voxel voxel = Voxel::Coordinates(point.pt, size_voxel);
                voxel_map.try_emplace(voxel, point);
            }
        } else {
#ifdef DEBUG
            // std::cout << "[SUB_SAMPLE] Using parallel path (build_voxel_map)." << std::endl;
#endif
            // Use the efficient parallel builder for large frames
            build_voxel_map(frame, size_voxel, voxel_map, sequential_threshold);
        }

#ifdef DEBUG
        // std::cout << "[SUB_SAMPLE] Voxel map created with " << voxel_map.size() << " unique voxels." << std::endl;
#endif

        // Step 2: Rebuild the frame with the downsampled points
        frame.clear();
        frame.reserve(voxel_map.size());
        for (const auto& pair : voxel_map) {
            frame.push_back(pair.second);
        }
        frame.shrink_to_fit();
    }

    // ########################################################################
    // build_voxel_map
    // ########################################################################

    // Private helper to build the map efficiently in parallel
    void lidarinertialodom::build_voxel_map(const std::vector<Point3D>& frame, double size_voxel, 
                                            tsl::robin_map<Voxel, Point3D, VoxelHash>& voxel_map, int sequential_threshold) {
        
        // Define a concurrent map for parallel insertion
        using ConcurrentVoxelMap = tbb::concurrent_hash_map<Voxel, Point3D, VoxelHash>;
        ConcurrentVoxelMap concurrent_map;

        // Use TBB to build the concurrent map in one parallel pass
        // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size(), sequential_threshold), 
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    const auto& point = frame[i];
                    Voxel voxel = Voxel::Coordinates(point.pt, size_voxel);
                    // Insert directly. The map handles thread-safe, unique insertions.
                    concurrent_map.insert({voxel, point});
                }
            });

        // For compatibility with the rest of the code, copy the result to a standard robin_map.
        // This is a fast, one-time operation.
        voxel_map.reserve(concurrent_map.size());
        for (const auto& pair : concurrent_map) {
            voxel_map.insert(pair);
        }
    }

    // ########################################################################
    // sub_sample_frame_outlier_removal
    // ########################################################################

    // Optimized main function
    // neighborhood-based filter like Radius Outlier Removal
    // Statistical Outlier Removal (SOR) > maybe can try
    // non-Clustering removal > maybe can try
    void lidarinertialodom::sub_sample_frame_outlier_removal(std::vector<Point3D>& frame, double size_voxel, int sequential_threshold) {

        if (frame.empty()) return;

#ifdef DEBUG
        // const size_t initial_size = frame.size();
        // std::cout << "[OUTLIER_REMOVAL] Starting. Initial points: " << initial_size << ", Voxel Size: " << size_voxel << std::endl;
#endif

        using VoxelMap = tsl::robin_map<Voxel, Point3D, VoxelHash>;

        // Step 1: Build the downsampled voxel map using the optimized parallel helper
        VoxelMap voxel_map;
        if (frame.size() < static_cast<size_t>(sequential_threshold)) {
            // Use a simple sequential path for small frames
            voxel_map.reserve(frame.size() / 2);
            for (const auto& point : frame) {
                Voxel voxel = Voxel::Coordinates(point.pt, size_voxel);
                voxel_map.try_emplace(voxel, point);
            }
        } else {
            // Use the efficient parallel builder for large frames
            build_voxel_map(frame, size_voxel, voxel_map, sequential_threshold);
        }

#ifdef DEBUG
        // const size_t downsampled_size = voxel_map.size();
        // std::cout << "[OUTLIER_REMOVAL] After downsampling, points: " << downsampled_size << std::endl;
#endif

        // Step 2: Set up ROR filter parameters
        const double ror_radius_sq = size_voxel * size_voxel * 3.0; // Use squared distance
        const int ror_min_pts = 3;
        const int k = static_cast<int>(std::ceil(std::sqrt(ror_radius_sq) / size_voxel)) + 1;
        const bool approximate = true;

#ifdef DEBUG
        // std::cout << "[OUTLIER_REMOVAL] ROR params: min_pts=" << ror_min_pts << ", neighbor_search_k=" << k << std::endl;
#endif

        // Step 3: Parallel filter. We can iterate directly over the keys of the map.
        // This avoids creating a huge intermediate std::vector of pairs.
        std::vector<Voxel> keys;
        keys.reserve(voxel_map.size());
        for(const auto& pair : voxel_map) {
            keys.push_back(pair.first);
        }
        
        std::vector<bool> keep(keys.size(), false);
        // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size(), sequential_threshold), [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    const Voxel& v = keys[i];
                    int neighbor_count = 0;
                    for (int dx = -k; dx <= k; ++dx) {
                        for (int dy = -k; dy <= k; ++dy) {
                            for (int dz = -k; dz <= k; ++dz) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                
                                Voxel neigh_voxel = {v.x + dx, v.y + dy, v.z + dz};
                                // The map is only read from here, which is thread-safe
                                if (voxel_map.count(neigh_voxel)) {
                                    if (approximate) {
                                        neighbor_count++;
                                    } else {
                                        // A full distance check would require finding the point value
                                        // This is slightly more complex, so sticking to `approximate`
                                        // is often a good performance trade-off.
                                        neighbor_count++; // Simplified for this example
                                    }
                                }
                            }
                        }
                    }
                    if (neighbor_count >= ror_min_pts) {
                        keep[i] = true;
                    }
                }
            });

        // Step 4: Rebuild the frame sequentially from the filtered keys
        frame.clear();
        frame.reserve(keys.size());
        size_t points_kept = 0;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keep[i]) {
                frame.push_back(voxel_map.at(keys[i]));
                points_kept++;
            }
        }
        frame.shrink_to_fit();

#ifdef DEBUG
        // std::cout << "[OUTLIER_REMOVAL] After ROR filter, points kept: " << points_kept 
        //         << " (Removed " << downsampled_size - points_kept << " outliers)." << std::endl;
        // std::cout << "[OUTLIER_REMOVAL] Finished. Final points: " << frame.size() 
        //         << " (Total reduction: " << initial_size - frame.size() << ")." << std::endl;
#endif
    }

    // ########################################################################
    // grid_sampling
    // ########################################################################

    void lidarinertialodom::grid_sampling(const std::vector<Point3D>& frame, std::vector<Point3D>& keypoints, 
                                     double size_voxel_subsampling, int sequential_threshold) {

#ifdef DEBUG
        // std::cout << "[GRID_SAMPLING] Starting grid_sampling. Input points: " << frame.size() << std::endl;
#endif

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
        // Calls sub_sample_frame_outlier_removal to reduce the number of points by keeping one point per voxel
        // Modifies frame_sub in-place, using the provided voxel size, thread count, and threshold
        sub_sample_frame_outlier_removal(frame_sub, size_voxel_subsampling, sequential_threshold);

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

#ifdef DEBUG
        // std::cout << "[GRID_SAMPLING] Finished. Output keypoints: " << keypoints.size() << std::endl;
#endif
    }

    // ########################################################################
    // compute_neighborhood_distribution
    // ########################################################################

    // Assuming ArrayVector3d is std::vector<Eigen::Vector3d>
    // and Neighborhood is a struct with: Eigen::Vector3d center, normal; Eigen::Matrix3d covariance; double a2D;

    lidarinertialodom::Neighborhood lidarinertialodom::compute_neighborhood_distribution(
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

    lidarinertialodom::Options lidarinertialodom::parse_json_options(const std::string& json_path) {
        std::ifstream file(json_path);
        if (!file.is_open()) {throw std::runtime_error("Failed to open JSON file: " + json_path);}

        nlohmann::json json_data;
        try {
            file >> json_data;
        } catch (const nlohmann::json::parse_error& e) {throw std::runtime_error("JSON parse error in " + json_path + ": " + e.what());}

        lidarinertialodom::Options parsed_options;

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
                else {throw std::runtime_error("Invalid p2p_loss_func: " + loss_func);}
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
            if (odometry_options.contains("acc_loss_func")) parsed_options.acc_loss_func = odometry_options["acc_loss_func"].get<std::string>();
            if (odometry_options.contains("acc_loss_sigma")) parsed_options.acc_loss_sigma = odometry_options["acc_loss_sigma"].get<double>();
            if (odometry_options.contains("gyro_loss_func")) parsed_options.gyro_loss_func = odometry_options["gyro_loss_func"].get<std::string>();
            if (odometry_options.contains("gyro_loss_sigma")) parsed_options.gyro_loss_sigma = odometry_options["gyro_loss_sigma"].get<double>();
            
            if (odometry_options.contains("T_mi_init_only")) parsed_options.T_mi_init_only = odometry_options["T_mi_init_only"].get<bool>();
            if (odometry_options.contains("use_T_mi_gt")) parsed_options.use_T_mi_gt = odometry_options["use_T_mi_gt"].get<bool>();
            if (odometry_options.contains("xi_ig") && odometry_options["xi_ig"].is_array() && odometry_options["xi_ig"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.xi_ig(i) = odometry_options["xi_ig"][i].get<double>();
                }
            }
            if (odometry_options.contains("T_mi_init_cov") && odometry_options["T_mi_init_cov"].is_array() && odometry_options["T_mi_init_cov"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.T_mi_init_cov(i) = odometry_options["T_mi_init_cov"][i].get<double>();
                }
            }
            if (odometry_options.contains("qg_diag") && odometry_options["qg_diag"].is_array() && odometry_options["qg_diag"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.qg_diag(i) = odometry_options["qg_diag"][i].get<double>();
                }
            }
            if (odometry_options.contains("T_mi_prior_cov") && odometry_options["T_mi_prior_cov"].is_array() && odometry_options["T_mi_prior_cov"].size() == 6) {
                for (int i = 0; i < 6; ++i) {
                    parsed_options.T_mi_prior_cov(i) = odometry_options["T_mi_prior_cov"][i].get<double>();
                }
            }
            if (odometry_options.contains("use_T_mi_prior_after_init")) parsed_options.use_T_mi_prior_after_init = odometry_options["use_T_mi_prior_after_init"].get<bool>();
            if (odometry_options.contains("use_bias_prior_after_init")) parsed_options.use_bias_prior_after_init = odometry_options["use_bias_prior_after_init"].get<bool>();

            if (odometry_options.contains("T_rm_init") && odometry_options["T_rm_init"].is_array() && odometry_options["T_rm_init"].size() == 16) {
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        parsed_options.T_rm_init(i, j) = odometry_options["T_rm_init"][i * 4 + j].get<double>();
                    }
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
            if (odometry_options.contains("filter_lifetimes")) parsed_options.filter_lifetimes = odometry_options["filter_lifetimes"].get<bool>();
            
        } catch (const nlohmann::json::exception& e) {throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));}

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
            return; // Avoid further operations if file cannot be opened
        }
#ifdef DEBUG
            std::cout << "[DECONSTRUCT] Building full trajectory." << std::endl;
#endif

        // Build full trajectory
        auto full_trajectory =  finalicp::traj::const_acc::Interface::MakeShared(options_.qc_diag);
        for (const auto& var : trajectory_vars_) {
            full_trajectory->add(var.time, var.T_rm, var.w_mr_inr, var.dw_mr_inr);
        }
#ifdef DEBUG
            std::cout << "[DECONSTRUCT] Dumping trajectory." << std::endl;
#endif

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

            buffer << traj_time.nanosecs() << " "
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

#ifdef DEBUG
        std::cout << "[DECONSTRUCT] Dumping trajectory. - DONE" << std::endl;
#endif
    }

    // ########################################################################
    // setInitialPose
    // ########################################################################
    
    void lidarinertialodom::initT(const Eigen::Matrix4d& T) {
        options_.T_rm_init = T;
    }

    // ########################################################################
    // trajectory
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
        const Eigen::Matrix4d T_rs = options_.T_sr.inverse();

        // Sequential update
        for (auto& frame : trajectory_) {
            // Begin pose
            Time begin_slam_time(frame.begin_timestamp);
            const auto begin_T_mr = full_trajectory->getPoseInterpolator(begin_slam_time)->value().inverse().matrix();
            const auto begin_T_ms = begin_T_mr * T_rs;
            frame.begin_R = begin_T_ms.block<3, 3>(0, 0);
            frame.begin_t = begin_T_ms.block<3, 1>(0, 3);

            // Mid pose
            Time mid_slam_time(static_cast<double>(frame.getEvalTime()));
            const auto mid_T_mr = full_trajectory->getPoseInterpolator(mid_slam_time)->value().inverse().matrix();
            const auto mid_T_ms = mid_T_mr * T_rs;
            frame.setMidPose(mid_T_ms);

            // End pose
            Time end_slam_time(frame.end_timestamp);
            const auto end_T_mr = full_trajectory->getPoseInterpolator(end_slam_time)->value().inverse().matrix();
            const auto end_T_ms = end_T_mr * T_rs;
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

#ifdef DEBUG
        // Initialize timers for performance debugging if enabled
        std::vector<std::pair<std::string, std::unique_ptr<finalicp::Stopwatch<>>>> timer;
        timer.emplace_back("initialization ..................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("gridsampling ....................... ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("icp ................................ ", std::make_unique<finalicp::Stopwatch<>>(false));
        timer.emplace_back("updateMap .......................... ", std::make_unique<finalicp::Stopwatch<>>(false));
#endif

        // Step 1: Validate input point cloud
        // Check if the input point cloud is empty; return failure if so
        if (const_frame.pointcloud.empty()) {
#ifdef DEBUG
            std::cout << "[REG DEBUG] CRITICAL: Frame " << trajectory_.size() << " REJECTED: Input point cloud is empty." << std::endl;
#endif
            summary.success = false;
            return summary;
        }

        // Step 2: Add new frame to trajectory
        // Create a new entry in the trajectory vector for the current frame
        int index_frame = trajectory_.size(); // start from 0,1,2,3,4
        trajectory_.emplace_back();

#ifdef DEBUG
        // [DEBUG] Announce the start of processing for a new frame
        std::cout << "[REG DEBUG] Starting RegisterFrame for index " << index_frame << std::endl;
        std::cout << "[REG DEBUG] Input pointcloud size: " << const_frame.pointcloud.size() << std::endl;
#endif

        // Step 3: Initialize frame metadata
        // Set up timestamp and motion data for the new frame
        initializeTimestamp(index_frame, const_frame);                                  //####!!! 1 tbb included // find the min and max timestamp in a single frame

#ifdef DEBUG
        // [DEBUG] Check the timestamps immediately after they are calculated
        // std::cout << std::fixed << std::setprecision(12) 
        //           << "[REG DEBUG] After initializeTimestamp: begin=" << trajectory_[index_frame].begin_timestamp
        //           << ", end=" << trajectory_[index_frame].end_timestamp << std::endl;
#endif

        initializeMotion(index_frame);                                                  //####!!! 2 // estimate the motion based on prev and prev*prev frame

#ifdef DEBUG
        if (index_frame == 0) {
            std::cout << "[REG DEBUG] Frame 0 Initial Pose (R_ms):\n" << trajectory_[index_frame].end_R  << std::endl;
            std::cout << "[REG DEBUG] Frame 0 Initial Pose (t_ms):\n" << trajectory_[index_frame].end_t.transpose() << std::endl;
        }
        // [DEBUG] Check the initial motion prediction
        if (!trajectory_[index_frame].begin_R.allFinite() || !trajectory_[index_frame].end_R.allFinite()) {
            std::cout << "[REG DEBUG] CRITICAL: Non-finite rotation after initializeMotion!" << std::endl;
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
        // std::cout << "[REG DEBUG] After initializeFrame, size is: " << frame.size() << std::endl;
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
            std::cout << "[REG DEBUG] After grid_sampling, keypoints size: " << keypoints.size() << std::endl;
#endif
            // Step 5b: Perform Iterative Closest Point (ICP) registration
            // Align current frame with previous frames using IMU and pose data
            const auto& imu_data_vec = const_frame.imu_data_vec;
            const auto& pose_data_vec = const_frame.pose_data_vec;

#ifdef DEBUG
            if (!timer.empty()) timer[2].second->start();
#endif

            summary.success = icp(index_frame, keypoints, imu_data_vec, pose_data_vec); //####!!! 5

#ifdef DEBUG
            if (!timer.empty()) timer[2].second->stop();
            // [DEBUG] Report ICP result immediately
            std::cout << "[REG DEBUG] ICP finished. Success: " << (summary.success ? "true" : "false") << std::endl;
#endif
            summary.keypoints = keypoints;
            if (!summary.success) {
#ifdef DEBUG
                std::cout << "[REG DEBUG] ICP failed for frame " << index_frame << ". Returning early." << std::endl;
#endif
                return summary;}
        } else { // !!!!!!!!!!! this is responsible for initial frame 0 ######################

#ifdef DEBUG
            // [DEBUG] Announce first frame initialization
            std::cout << "[REG DEBUG] Performing first frame (index 0) initialization." << std::endl;
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
            // Initialize T_mi_var DIRECTLY with the ground truth value
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
            // Initialize T_mi_var DIRECTLY with the ground truth value
            auto end_T_mi_var = SE3StateVar::MakeShared(T_mi); 
            trajectory_vars_.emplace_back(end_slam_time, std::move(end_T_rm_var), std::move(end_w_mr_inr_var),
                                        std::move(end_dw_mr_inr_var), std::move(end_imu_biases), std::move(end_T_mi_var));

#ifdef DEBUG
            std::cout << "[REG DEBUG] Frame 0: Created " << trajectory_vars_.size() << " initial state variables." << std::endl;
            std::cout << "[REG DEBUG] Frame 0 timestamps: begin=" << std::fixed << begin_time << ", end=" << end_time << std::endl;
#endif
            Eigen::Matrix<double, 6, 1> xi_mi = initialize_gravity(const_frame.imu_data_vec);
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

#ifdef DEBUG
            if (!timer.empty()) timer[2].second->stop();
#endif
        }

        // Step 6: Store processed points
        // Save the processed point cloud to the trajectory
        trajectory_[index_frame].points = std::move(frame);

        // Step 7: Update the map
        // Incorporate points into the global map, with optional delay
#ifdef DEBUG
        if (!timer.empty()) timer[3].second->start();
#endif

        if (index_frame == 0) {
#ifdef DEBUG
            std::cout << "[REG DEBUG] Updating map for frame 0." << std::endl;
#endif
            updateMap(index_frame, index_frame);                                        //####!!! 7

        } else if ((index_frame - options_.delay_adding_points) > 0) {
#ifdef DEBUG
            std::cout << "[REG DEBUG] Updating map using frame " << index_frame - options_.delay_adding_points << "." << std::endl;
#endif
            updateMap(index_frame, index_frame - options_.delay_adding_points);
        }

#ifdef DEBUG
        if (!timer.empty()) timer[3].second->stop();
        std::cout << "[REG DEBUG] Map size is now: " << map_.size() << std::endl;
#endif

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
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, concurrent_points.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
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
#ifdef DEBUG
            std::cout << "[REG DEBUG] OUTER LOOP TIMERS" << std::endl;
            for (size_t i = 0; i < timer.size(); i++) {
                std::cout << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
            }
            std::cout << "[REG DEBUG] Finished RegisterFrame for index " << index_frame << std::endl;
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
            std::cout << "[INIT TS DEBUG] Frame " << index_frame << ": min=" << std::fixed << min_timestamp
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

    void lidarinertialodom::initializeMotion(int index_frame) {// the first frame should be initialized with Tmr

#ifdef DEBUG
        // [ADDED DEBUG] Announce entry into the function
        std::cout << "[INIT MOTION DEBUG] Initializing motion for frame " << index_frame << ". ---" << std::endl;
#endif

        if (index_frame == 0) { //MAIN ALLERT as T_sr is identity. T_ms is exactly same as T_mr
            // --- For the very first frame, use the ground truth initial pose ---

            // // 1. Get the ground truth T_rm from options.
            // const Eigen::Matrix4d T_rm = options_.T_rm_init;

            // 1. Let initial as identity
            // const Eigen::Matrix4d T_rm = Eigen::Matrix<double, 4, 4>::Identity();

            // // 2. Efficiently compute its inverse to get T_mr.
            // Eigen::Matrix3d R_rm = T_rm.block<3, 3>(0, 0);
            // Eigen::Vector3d t_rm = T_rm.block<3, 1>(0, 3);

            // Eigen::Matrix3d R_mr = R_rm.transpose();
            // Eigen::Vector3d t_mr = -R_mr * t_rm;

            // Eigen::Matrix4d T_mr = Eigen::Matrix4d::Identity();
            // T_mr.block<3, 3>(0, 0) = R_mr;
            // T_mr.block<3, 1>(0, 3) = t_mr;
            const Eigen::Matrix4d T_mr = Eigen::Matrix<double, 4, 4>::Identity();


            // 3. Get the transformation from sensor to robot (T_rs).
            const Eigen::Matrix4d T_rs = options_.T_sr.inverse();

            // 4. Calculate the initial sensor pose in the map: T_ms = T_mr * T_rs.
            const Eigen::Matrix4d T_ms = T_mr * T_rs;

#ifdef DEBUG
            // [ADDED DEBUG] Print the initial transformations for Frame 0
            std::cout << "[INIT MOTION DEBUG] Frame 0: Using initial pose from options." << std::endl;
            std::cout << "[INIT MOTION DEBUG] Frame 0: Initial Sensor Pose (T_ms):\n" << T_ms << std::endl;
            if (!T_ms.allFinite()) {
                std::cout << "[INIT MOTION DEBUG] CRITICAL: Initial pose T_ms is non-finite (NaN or inf)!" << std::endl;
            } else {
                std::cout << "[INIT MOTION DEBUG] Initial pose T_ms is finite." << std::endl;
            }
#endif
            // 5. Set the trajectory's initial pose.
            trajectory_[index_frame].begin_R = T_ms.block<3, 3>(0, 0);
            trajectory_[index_frame].begin_t = T_ms.block<3, 1>(0, 3);
            trajectory_[index_frame].end_R = T_ms.block<3, 3>(0, 0);
            trajectory_[index_frame].end_t = T_ms.block<3, 1>(0, 3);

        } else if (index_frame == 1) {
            // For the second frame, its motion starts from the end of the first frame.
            trajectory_[index_frame].begin_R = trajectory_[index_frame - 1].end_R;
            trajectory_[index_frame].begin_t = trajectory_[index_frame - 1].end_t;
            trajectory_[index_frame].end_R = trajectory_[index_frame - 1].end_R;
            trajectory_[index_frame].end_t = trajectory_[index_frame - 1].end_t;

#ifdef DEBUG
        // [ADDED DEBUG] Confirm the pose was copied for Frame 1
        std::cout << "[INIT MOTION DEBUG] Frame 1: Copying pose from end of Frame 0." << std::endl;
        std::cout << "[INIT MOTION DEBUG] Frame 1: Initial Pose (Rotation):\n" << trajectory_[index_frame].begin_R << std::endl;
        std::cout << "[INIT MOTION DEBUG] Frame 1: Initial Pose (Translation): " << trajectory_[index_frame].begin_t.transpose() << std::endl;
        if (!trajectory_[index_frame].begin_R.allFinite() || !trajectory_[index_frame].end_R.allFinite() || !trajectory_[index_frame].begin_t.allFinite() || !trajectory_[index_frame].end_t.allFinite()) {
            std::cout << "[INIT MOTION DEBUG] CRITICAL: Initial pose T_ms is non-finite (NaN or inf)!" << std::endl;
        } else {
            std::cout << "[INIT MOTION DEBUG] Initial pose T_ms is finite." << std::endl;
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
            std::cout << "[INIT MOTION DEBUG] Frame " << index_frame << ": Extrapolating motion." << std::endl;
            std::cout << "[INIT MOTION DEBUG] Relative Motion (t_rel): " << t_rel.transpose() << std::endl;
            std::cout << "[INIT MOTION DEBUG] Extrapolated End Pose (Translation): " << trajectory_[index_frame].end_t.transpose() << std::endl;
            if (!trajectory_[index_frame].begin_R.allFinite() || !trajectory_[index_frame].end_R.allFinite() || !trajectory_[index_frame].begin_t.allFinite() || !trajectory_[index_frame].end_t.allFinite()) {
                std::cout << "--- [INIT MOTION DEBUG] CRITICAL: Initial pose T_ms is non-finite (NaN or inf)! ---" << std::endl;
            } else {
                std::cout << "--- [INIT MOTION DEBUG] Initial pose T_ms is finite. ---" << std::endl;
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

    std::vector<Point3D> lidarinertialodom::initializeFrame(int index_frame, const std::vector<Point3D> &const_frame) {
        // this is critical as the code assume T_sr is identity. if T_sr is not identity then we need to add some more algorithm.
#ifdef DEBUG
        // [ADDED DEBUG] Announce entry and check input size
        std::cout << "[FRAME INIT DEBUG] Initializing frame " << index_frame << " with " << const_frame.size() << " input points." << std::endl;
#endif

        // Initialize point cloud
        std::vector<Point3D> frame = const_frame; // Copy necessary due to const input

        // Select voxel size
        const double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;

        // Subsample
        sub_sample_frame(frame, sample_size, options_.sequential_threshold);

#ifdef DEBUG
        // [ADDED DEBUG] Show size after subsampling
        std::cout << "[FRAME INIT DEBUG] Frame size after subsampling: " << frame.size() << " points." << std::endl;
#endif

        // Shuffle points to avoid bias
        std::mt19937_64 g(42); // Fixed seed for reproducibility
        std::shuffle(frame.begin(), frame.end(), g);

        // Validate poses
        const auto& traj = trajectory_[index_frame]; //contain R_ms and t_ms

#ifdef DEBUG
        // [ADDED DEBUG] Check input poses for validity before using them
        if (!traj.begin_R.allFinite() || !traj.end_R.allFinite() || !traj.begin_t.allFinite() || !traj.end_t.allFinite()) {
            std::cout << "[FRAME INIT DEBUG] CRITICAL: Input trajectory poses for deskewing are non-finite!" << std::endl;
        }
        std::cout << "[FRAME INIT DEBUG] Deskewing with begin_t: " << traj.begin_t.transpose() << " and end_t: " << traj.end_t.transpose() << std::endl;
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
                std::cout << "[FRAME INIT DEBUG] CRITICAL: A deskewed point is non-finite (NaN or inf)!" << std::endl;
                all_points_finite = false;
                break;
            }
        }
        if (all_points_finite) {
            std::cout << "[FRAME INIT DEBUG] All " << frame.size() << " deskewed points are finite." << std::endl;
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

    void lidarinertialodom::updateMap(int index_frame, int update_frame) {
    
#ifdef DEBUG
        // [DEBUG] Announce the start of the function and its parameters
        std::cout << "[MAP DEBUG] Starting updateMap" << std::endl;
        std::cout << "[MAP DEBUG] Current frame index: " << index_frame << ", Updating with data from frame: " << update_frame << std::endl;
#endif

        // Map parameters
        const double kSizeVoxelMap = options_.size_voxel_map;
        const double kMinDistancePoints = options_.min_distance_points;
        const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

        // Update frame
        auto& frame = trajectory_[update_frame].points;
        if (frame.empty()) {
#ifdef DEBUG
            std::cout << "[MAP DEBUG] Frame " << update_frame << " is empty. Nothing to add to map. Returning." << std::endl;
#endif
            return; // No points to add
        }

#ifdef DEBUG
        std::cout << "[MAP DEBUG] Frame " << update_frame << " contains " << frame.size() << " points to process." << std::endl;
#endif

        // Motion correction with SLAM interpolation
        auto update_trajectory = finalicp::traj::singer::Interface::MakeShared(options_.qc_diag, options_.ad_diag);
        const double begin_timestamp = trajectory_[update_frame].begin_timestamp;
        const double end_timestamp = trajectory_[update_frame].end_timestamp;

        const finalicp::traj::Time begin_slam_time(begin_timestamp); // Fixed: Use finalicp::traj::Time
        const finalicp::traj::Time end_slam_time(end_timestamp);     // Fixed: Use finalicp::traj::Time

        // Add trajectory states
        size_t num_states = 0;
        size_t start_idx = std::max(static_cast<int>(to_marginalize_) - 1, 0);
        for (size_t i = start_idx; i < trajectory_vars_.size(); i++) {
            const auto& var = trajectory_vars_.at(i);
            update_trajectory->add(var.time, var.T_rm, var.w_mr_inr, var.dw_mr_inr);
            num_states++;
            if (var.time == end_slam_time) break;
            if (var.time > end_slam_time) {
                throw std::runtime_error("Trajectory variable time exceeds end_slam_time in updateMap for frame " + std::to_string(update_frame));
            }
        }

#ifdef DEBUG
        std::cout << "[MAP DEBUG] Building interpolation trajectory from state index " << start_idx 
                << " to " << (start_idx + num_states - 1) << " (" << num_states << " states)." << std::endl;
        std::cout << "[MAP DEBUG] Trajectory covers time range (inclusive): " << std::fixed << std::setprecision(12) 
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
        std::cout << "[MAP DEBUG] Found " << unique_point_times.size() << " unique timestamps in the point cloud." << std::endl;
#endif
        // Cache interpolated poses
        const Eigen::Matrix4d T_rs = options_.T_sr.inverse(); // transformation of sensor relative to robot

        std::map<double, Eigen::Matrix4d> T_ms_cache_map;
        if (unique_point_times.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential pose interpolation
            for (const auto& ts : unique_point_times) {
                const auto T_rm_intp_eval = update_trajectory->getPoseInterpolator(finalicp::traj::Time(ts)); // Correctly gets the Robot-in-Map pose (T_rm).

                const Eigen::Matrix4d T_ms = T_rm_intp_eval->value().inverse().matrix() * T_rs; // This correctly computes the final Sensor-to-Map transformation (T_ms = T_mr * T_rs).

                T_ms_cache_map[ts] = T_ms; 
            }
        } else {
            // Parallel pose interpolation with TBB
            // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::concurrent_vector<Eigen::Matrix4d> T_ms_cache(unique_point_times.size());
            tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
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

#ifdef DEBUG
        // [DEBUG] Verify that cached poses are valid
        bool poses_are_finite = true;
        for(const auto& pair : T_ms_cache_map) {
            if (!pair.second.allFinite()) {
                poses_are_finite = false;
                std::cout << "[MAP DEBUG] CRITICAL: Cached pose for timestamp " << pair.first << " is NOT finite!" << std::endl;
                break;
            }
        }
        if (poses_are_finite) {
            std::cout << "[MAP DEBUG] All " << T_ms_cache_map.size() << " cached poses are finite." << std::endl;
        }
#endif

        // Apply transformations
        if (frame.size() < static_cast<size_t>(options_.sequential_threshold)) {
            // Sequential point transformation
            for (auto& point : frame) {
                try {
                    const auto& T_ms = T_ms_cache_map.at(point.timestamp);
                    point.pt = T_ms.block<3, 3>(0, 0) * point.raw_pt + T_ms.block<3, 1>(0, 3); // Correctly applies the transformation to move a point from the sensor's frame into the global map frame.
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
                            const auto& T_ms = T_ms_cache_map.at(point.timestamp);
                            point.pt = T_ms.block<3, 3>(0, 0) * point.raw_pt + T_ms.block<3, 1>(0, 3);
                        } catch (const std::out_of_range&) {
                            throw std::runtime_error("Timestamp not found in cache in updateMap for frame " + std::to_string(update_frame));
                        }
                    }
                });
        }

#ifdef DEBUG
        // [DEBUG] Verify that transformed points are valid
        bool points_are_finite = true;
        for(const auto& point : frame) {
            if(!point.pt.allFinite()) {
                points_are_finite = false;
                std::cout << "[MAP DEBUG] CRITICAL: Transformed point `pt` is NOT finite!" << std::endl;
                break;
            }
        }
#endif

        // Update map
        map_.add(frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints);

#ifdef DEBUG
        std::cout << "[MAP DEBUG] Map size after adding new points " << map_.size() << " points." << std::endl;
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
        std::cout << "[MAP DEBUG] Removing points farther than " << kMaxDistance << "m from " << location.transpose() << std::endl;
        std::cout << "[MAP DEBUG] Map size after outlier point removal " << map_.size() << " points." << std::endl;
        std::cout << "--- [MAP DEBUG] Finished updateMap ---" << std::endl;
#endif

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

#ifdef DEBUG
        // [ADDED DEBUG] Check if we have any IMU data to begin with
        std::cout << "[GRAVITY INIT DEBUG] Received " << imu_data_vec.size() << " IMU data points for initialization." << std::endl;
        if (imu_data_vec.empty()) {
            std::cout << "[GRAVITY INIT DEBUG] CRITICAL: No IMU data provided, cannot initialize gravity. Returning zero vector." << std::endl;
            return Eigen::Matrix<double, 6, 1>::Zero();
        }
#endif

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
        for (const auto &imu_data : imu_data_vec) {
            auto acc_error_func = finalicp::imu::AccelerationError(T_rm_init, dw_mr_inr, bias, T_mi_var, imu_data.lin_acc);
            acc_error_func->setGravity(options_.gravity);
            const auto acc_cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(acc_error_func, noise_model, loss_func);
            cost_terms.emplace_back(acc_cost);
        }

#ifdef DEBUG
        // [ADDED DEBUG] Confirm that cost terms were actually created
        std::cout << "[GRAVITY INIT DEBUG] Created " << cost_terms.size() << " acceleration cost terms." << std::endl;
#endif

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
        params.max_iterations = static_cast<unsigned int>(options_.max_iterations);
        finalicp::GaussNewtonSolverNVA solver(problem, params);
        solver.optimize();

#ifdef DEBUG
        std::cout<< "[GRAVITY INIT DEBUG] T_mi:\n" << T_mi_var->value().matrix() << std::endl;
        std::cout << "[GRAVITY INIT DEBUG] T_mi_var:\n"  << T_mi_var->value().vec() << std::endl;
        // [ADDED DEBUG] Check if the result of the optimization is valid
        if (!T_mi_var->value().vec().allFinite()) {
            std::cout << "[GRAVITY INIT DEBUG] CRITICAL: Solver produced a non-finite (NaN or inf) result!" << std::endl;
        } else {
            std::cout << "[GRAVITY INIT DEBUG] Solver finished, result is finite." << std::endl;
        }
#endif
        
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

#ifdef DEBUG
    // [DEBUG] Initial check at the start of the function
    std::cout << "[ICP DEBUG | Frame " << index_frame << "]" << std::endl;
    std::cout << "[ICP DEBUG] Starting with " << keypoints.size() << " keypoints." <<  ", "<< imu_data_vec.size() << " IMU datas." <<  ", "<< pose_data_vec.size() << " pose datas." << std::endl;
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

#ifdef DEBUG
    // [DEBUG] Check if the state from the previous frame is valid
    if (!prev_T_rm.matrix().allFinite()) { std::cout << "[ICP DEBUG] CRITICAL: prev_T_rm is NOT finite!" << std::endl; }
    if (!prev_w_mr_inr.allFinite()) { std::cout << "[ICP DEBUG] CRITICAL: prev_w_mr_inr is NOT finite!" << std::endl; }
    if (!prev_dw_mr_inr.allFinite()) { std::cout << "[ICP DEBUG] CRITICAL: prev_dw_mr_inr is NOT finite!" << std::endl; }
#endif

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
#ifdef DEBUG
        std::cout << "[ICP DEBUG] SLAM_TRAJ: add prev_slam_time." << std::endl; 
        std::cout << "[ICP DEBUG] SLAM_TRAJ: add prev_T_rm_var."  << std::endl;
        std::cout << "[ICP DEBUG] SLAM_TRAJ: add prev_w_mr_inr_var." << std::endl;
        std::cout << "[ICP DEBUG] SLAM_TRAJ: add prev_dw_mr_inr_var." << std::endl;
#endif

        // Step 15: Add previous state variables to optimization list
        // These variables will be optimized (if not locked) in ICP
        SLAM_STATE_VAR.emplace_back(prev_T_rm_var); // Add pose
        SLAM_STATE_VAR.emplace_back(prev_w_mr_inr_var); // Add velocity
        SLAM_STATE_VAR.emplace_back(prev_dw_mr_inr_var); // Add acceleration

#ifdef DEBUG
    std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace prev_T_rm_var." << std::endl; 
    std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace prev_w_mr_inr_var." << std::endl;
    std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace prev_dw_mr_inr_var." << std::endl;
#endif

        // Step 16: Handle IMU-related state variables (if IMU is enabled)
        if (options_.use_imu) {
            // Add IMU biases to optimization (biases evolve over time)
            SLAM_STATE_VAR.emplace_back(prev_imu_biases_var);
#ifdef DEBUG
            std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace prev_imu_biases_var." << std::endl; 
#endif

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
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace prev_T_mi_var." << std::endl; 
#endif
                }
                // If T_mi_init_only=true and index_frame>1, T_mi stays as is (not optimized)
            }
        }

        ///################################################################################

        // Step 17: Get the current frame’s end timestamp
        // curr_time tells us when this frame ends
        const double CURR_TIME = trajectory_[index_frame].end_timestamp;

#ifdef DEBUG
        std::cout << std::fixed << std::setprecision(12) 
        << "[ICP DEBUG] LOGGING: PREV_TIME: " << PREV_TIME << ", CURR_TIME: " << CURR_TIME << std::endl;
#endif

        // [DEBUG] THIS IS THE MOST LIKELY CULPRIT
        if (CURR_TIME <= PREV_TIME) {
#ifdef DEBUG
            std::cout << "[ICP DEBUG] CRITICAL: Zero or negative time difference between frames!" << std::endl;
#endif
            return false;
        }

        // Step 18: Calculate the number of new states to add
        // num_extra_states is how many extra points (knots) to add between PREV_TIME and curr_time
        // +1 includes the mandatory end state at curr_time
        const int NUM_STATES = options_.num_extra_states + 1;
#ifdef DEBUG
            std::cout << "[ICP DEBUG] Adding "<< NUM_STATES << " extra number of state between 2 original state." << std::endl;
#endif

        // Step 19: Create timestamps (knot times) for new states
        // knot_times lists when each new state occurs, from PREV_TIME to curr_time
        const double TIME_DIFF = (CURR_TIME - PREV_TIME) / static_cast<double>(NUM_STATES);

#ifdef DEBUG
        // [DEBUG] Check the calculated time difference
        std::cout << std::fixed << std::setprecision(12) << "[ICP DEBUG] Time difference : " << TIME_DIFF << "s" << std::endl;
        if (!std::isfinite(TIME_DIFF) || TIME_DIFF <= 0) {
            std::cout << "[ICP DEBUG] CRITICAL: Invalid TIME_DIFF!" << std::endl;
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
        } else {
            // For early frames, use trajectory interpolation
            T_NEXT_MAT = SLAM_TRAJ->getPoseInterpolator(finalicp::traj::Time(KNOT_TIMES.back()))->value().inverse().matrix();
        }

#ifdef DEBUG
        // [DEBUG] Check the initial pose prediction for NaNs
        if (!T_NEXT_MAT.allFinite()) {
            std::cout << "[ICP DEBUG] CRITICAL: Extrapolated pose T_NEXT_MAT is NOT finite!" << std::endl;
        }
#endif

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

            // Add state to trajectory
            SLAM_TRAJ->add(knot_slam_time, T_rm_var, w_mr_inr_var, dw_mr_inr_var);
#ifdef DEBUG
            std::cout << "[ICP DEBUG] SLAM_TRAJ: add knot_slam_time." << std::endl; 
            std::cout << "[ICP DEBUG] SLAM_TRAJ: add T_rm_var." << std::endl;
            std::cout << "[ICP DEBUG] SLAM_TRAJ: add w_mr_inr_var." << std::endl;
            std::cout << "[ICP DEBUG] SLAM_TRAJ: add dw_mr_inr_var." << std::endl;
#endif

            // Add state variables to optimization list
            SLAM_STATE_VAR.emplace_back(T_rm_var); // Add pose
            SLAM_STATE_VAR.emplace_back(w_mr_inr_var); // Add velocity
            SLAM_STATE_VAR.emplace_back(dw_mr_inr_var); // Add acceleration
#ifdef DEBUG
            std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace T_rm_var." << std::endl; 
            std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace w_mr_inr_var." << std::endl; 
            std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace dw_mr_inr_var." << std::endl; 
#endif
            if (options_.use_imu) {
                SLAM_STATE_VAR.emplace_back(imu_biases_var); // Add IMU biases
#ifdef DEBUG
                std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace imu_biases_var." << std::endl; 
#endif
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
#ifdef DEBUG
                        std::cout << "[ICP DEBUG] SLAM_STATE_VAR: emplace T_mi_var." << std::endl; 
#endif
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
            
#ifdef DEBUG
            std::cout << "[ICP DEBUG] SLAM_TRAJ: addPosePrior." << std::endl;
            std::cout << "[ICP DEBUG] SLAM_TRAJ: addVelocityPrior." << std::endl; 
            std::cout << "[ICP DEBUG] SLAM_TRAJ: addAccelerationPrior."  << std::endl; 
#endif

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
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] imu_prior_cost_terms: Emplace bias_prior_factor." << std::endl;
#endif

            } else if (options_.use_bias_prior_after_init) {  // For subsequent frames, add IMU bias prior if enabled
#ifdef DEBUG
                std::cout << "[ICP DEBUG] Apply a prior on the IMU bias for subsequent frame " << index_frame << std::endl;
#endif
                // controls whether to apply a prior on the IMU bias for all frames after the first one.
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
#ifdef DEBUG
                std::cout << "[ICP DEBUG] imu_prior_cost_terms: Emplace bias_prior_factor." << std::endl;
#endif
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

                // Create cost term to constrain initial T_mi
                auto T_mi_error = finalicp::se3::se3_error(PREV_VAR.T_mi, T_mi);
                auto noise_model = finalicp::StaticNoiseModel<6>::MakeShared(init_T_mi_cov);
                auto loss_func = finalicp::L2LossFunc::MakeShared();
                const auto T_mi_prior_factor = finalicp::WeightedLeastSqCostTerm<6>::MakeShared(T_mi_error, noise_model, loss_func);
                T_mi_prior_cost_terms.emplace_back(T_mi_prior_factor);
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] T_mi_prior_cost_terms: Emplace T_mi_prior_factor." << std::endl;
#endif
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
#ifdef DEBUG
                std::cout << "[ICP DEBUG] T_mi_prior_cost_terms: Emplace T_mi_prior_factor." << std::endl;
#endif
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
            std::cout << "[ICP DEBUG] Update sliding window variables." << index_frame << std::endl;
#endif

            // For the initial frame, include the previous frame’s state variables
            if (index_frame == 1) {
#ifdef DEBUG
                std::cout << "[ICP DEBUG] Apply a prior state variable on initial frame." << index_frame << std::endl;
#endif
                // Get the previous frame’s state variables
                const auto& PREV_VAR = trajectory_vars_.at(prev_trajectory_var_index);

                // Add pose, velocity, and acceleration to the sliding window
                sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.T_rm, PREV_VAR.w_mr_inr, PREV_VAR.dw_mr_inr});

                // If IMU is enabled, add IMU biases and optionally T_mi
                if (options_.use_imu) {
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] Apply an IMU bias variable on initial frame " << index_frame << std::endl;
#endif
                    sliding_window_filter_->addStateVariable(std::vector<finalicp::StateVarBase::Ptr>{PREV_VAR.imu_biases});
                    if (!use_T_mi_gt) {
#ifdef DEBUG
                        std::cout << "[ICP DEBUG] Apply a prior T_mi variable for initial frame " << index_frame << "as we are not using T_mi_gt" << std::endl;
#endif
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
#ifdef DEBUG
            std::cout << "[ICP DEBUG] Condition (index_frame > delay_adding_points) met. Entering marginalization." << std::endl;
#endif
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
#ifdef DEBUG
                        // Check if the variables are valid *before* marginalizing them
                        if(!VAR.T_rm->value().matrix().allFinite()) {
                           std::cout << "[ICP DEBUG] CRITICAL: VAR.T_rm at index " << i << " is NaN before marginalization!" << std::endl;
                        }
#endif
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
#ifdef DEBUG
                std::cout << "[ICP DEBUG] Collected " << num_states << " states to marginalize." << std::endl;
                std::cout << "[ICP DEBUG] Calling sliding_window_filter_->marginalizeVariable()" << std::endl;
#endif

                sliding_window_filter_->marginalizeVariable(marg_vars);
#ifdef DEBUG
                std::cout << std::fixed << std::setprecision(12) 
                << "[ICP DEBUG] Marginalizing time: " << begin_marg_time - end_marg_time << ", with num states: " << num_states << std::endl;
                std::cout << "[ICP DEBUG] Finished marginalization call." << std::endl;
#endif
            }
            // Step 30: Stop the marginalization timer
#ifdef DEBUG
            if (!timer.empty()) timer[5].second->stop();
#endif
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
        std::cout << "[ICP DEBUG] Found " << unique_point_times.size() << " unique timestamps in the point cloud." << std::endl;
#endif
        // Configure IMU cost term options
        auto imu_options = finalicp::IMUSuperCostTerm::Options();
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

#ifdef DEBUG
                    std::cout << "[ICP DEBUG] IMU Data: " << std::setprecision(4) << imu_data.timestamp << " " << imu_data.ang_vel.transpose() << " " << imu_data.lin_acc.transpose() << std::endl;
#endif

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
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] imu_cost_terms: Emplace acc_cost." << std::endl;
#endif
                }

                // Add gyroscope cost term
                const auto gyro_cost = finalicp::WeightedLeastSqCostTerm<3>::MakeShared(gyro_error_func, gyro_noise_model, gyro_loss_func);
                imu_cost_terms.emplace_back(gyro_cost);
#ifdef DEBUG
                std::cout << "[ICP DEBUG] imu_cost_terms: Emplace gyro_cost." << std::endl;
#endif
            }

            // Step 34: Add prior cost terms for IMU biases
            // Constrain changes in IMU biases between consecutive states
            // Get IMU prior cost terms
            
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
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] imu_prior_cost_terms: Emplace bias_prior_factor." << std::endl;
#endif
            }
            
            // Step 35: Add prior cost terms for T_mi (if not init-only and not using ground truth)
            // Constrain changes in T_mi between consecutive states
            // Get T_mi prior cost terms
            if (!options_.T_mi_init_only && !use_T_mi_gt) {
#ifdef DEBUG
                std::cout << "[ICP DEBUG] Apply a prior cost term for T_mi term for initial frame " << index_frame << "as we are not using ground truth and not init only." << std::endl;
#endif 
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
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] T_mi_prior_cost_terms: Emplace T_mi_prior_factor." << std::endl;
#endif
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
#ifdef DEBUG
        timer[0].second->start(); // Start update transform timer
#endif

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
            // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
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
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_point_times.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
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
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
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
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, keypoints.size(), options_.sequential_threshold),[&](const tbb::blocked_range<size_t>& range) {
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

#ifdef  DEBUG
            timer[0].second->stop(); // Stop update transform timer
#endif
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

        // ################################################################################
        // Step 43: Start ICP optimization loop ################################################################################
        // ################################################################################
        // Iterates to refine the trajectory using point-to-plane alignment
        for (int iter(0); iter < options_.num_iters_icp; iter++) {
#ifdef DEBUG
        // [DEBUG] Start of an ICP iteration
        std::cout << "[ICP DEBUG] --- Iteration " << iter << " ---" << std::endl;
#endif
            // Initialize optimization problem based on swf_inside_icp
            const auto problem = [&]() -> finalicp::Problem::Ptr {
                if (swf_inside_icp) {
#ifdef DEBUG
                    std::cout << "[ICP DEBUG] swf_inside_icp is true." << std::endl; 
                    std::cout << "[ICP DEBUG] problem: use SlidingWindowFilter." << std::endl; 
#endif
                    // Use SlidingWindowFilter for sliding window optimization
                    return std::make_shared<finalicp::SlidingWindowFilter>(*sliding_window_filter_);
                } else {
                    // Use OptimizationProblem for full state optimization
                    auto problem = finalicp::OptimizationProblem::MakeShared(options_.num_threads);
                    for (const auto& var : SLAM_STATE_VAR) {
                        problem->addStateVariable(var);
#ifdef DEBUG
                        std::cout << "[ICP DEBUG] problem: use OptimizationProblem addStateVariable: " << var << std::endl; 
#endif
                    }
                    return problem;
                }
            }();

            // Add prior cost terms to the problem
            SLAM_TRAJ->addPriorCostTerms(*problem);
#ifdef DEBUG
            std::cout << "[ICP DEBUG] SLAM_TRAJ: addPriorCostTerms problem: " << std::endl;
#endif
            for (const auto& prior_cost_term : prior_cost_terms) {
                problem->addCostTerm(prior_cost_term);
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: addCostTerm prior_cost_term: " << std::endl;
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
            std::cout << "[ICP DEBUG] Keypoint size for association: " << keypoints.size() << std::endl;
            bool keypoints_are_finite = true;
            for (size_t i = 0; i < keypoints.size(); ++i) {
                if (!keypoints[i].pt.allFinite()) {
                    std::cout << "[ICP DEBUG] CRITICAL: Keypoint " << i << " coordinate is NOT finite before association!" << std::endl;
                    keypoints_are_finite = false;
                    break;
                }
            }
            if (keypoints_are_finite) {
                std::cout << "[ICP DEBUG] All keypoint coordinates are finite before association." << std::endl;
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

                    if ((int)vector_neighbors.size() < kMinNumNeighbors) {
                        continue;
                    }

                    auto neighborhood = compute_neighborhood_distribution(vector_neighbors, options_.sequential_threshold);
                    const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
                    const double weight = planarity_weight;
                    const double dist_to_plane = std::abs((keypoint.pt - vector_neighbors[0]).transpose() * neighborhood.normal);

#ifdef DEBUG
                    if (i == 0) {
                        std::cout << "[ICP DEBUG] Association for point 0:" << std::endl;
                        std::cout << "[ICP DEBUG] Point coordinate: " << pt_keypoint.transpose() << std::endl;
                        std::cout << "[ICP DEBUG] Neighbors found: " << vector_neighbors.size() << std::endl;
                        std::cout << "[ICP DEBUG] Neighborhood a2D: " << neighborhood.a2D << std::endl;
                        std::cout << "[ICP DEBUG] Dist to plane: " << dist_to_plane << std::endl;
                        if (!std::isfinite(neighborhood.a2D) || !std::isfinite(dist_to_plane)) {
                            std::cout << "[ICP DEBUG] CRITICAL: NaN detected in neighborhood/distance calculation!" << std::endl;
                        }
                    }
#endif

                    if (dist_to_plane < options_.p2p_max_dist) {
#if USE_P2P_SUPER_COST_TERM
                        p2p_matches.emplace_back(P2PMatch(keypoint.timestamp, vector_neighbors[0],
                                                        weight * neighborhood.normal, keypoint.raw_pt));
#else
                        Eigen::Vector3d closest_pt = vector_neighbors[0];
                        Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                        Eigen::Matrix3d W = (closest_normal * closest_normal.transpose() + 1e-5 * Eigen::Matrix3d::Identity());
                        const auto noise_model = finalicp::StaticNoiseModel<3>::MakeShared(W, NoiseType::INFORMATION);
                        const auto &T_mr_intp_eval = T_mr_intp_eval_map.at(keypoint.timestamp);
                        const auto error_func = p2p::p2pError(T_mr_intp_eval, closest_pt, keypoint.raw_pt);
                        error_func->setTime(Time(keypoint.timestamp));

                        const auto loss_func = [this]() -> BaseLossFunc::Ptr {
                        switch (options_.p2p_loss_func) {
                            case lidarinertialodom::LOSS_FUNC::L2: return L2LossFunc::MakeShared();
                            case lidarinertialodom::LOSS_FUNC::DCS: return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
                            case lidarinertialodom::LOSS_FUNC::CAUCHY: return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
                            case lidarinertialodom::LOSS_FUNC::GM: return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
                            default: return nullptr;
                        }
                        }();
                        meas_cost_terms.emplace_back(finalicp::WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, loss_func));
#endif
                    }
                }
            } else {
                // --- PARALLEL PATH --- using the most scalable TBB pattern
                // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);

                // Structure to hold thread-local results
                struct ReductionResult {
                    std::vector<P2PMatch> p2p_matches_local;
#if !USE_P2P_SUPER_COST_TERM
                    std::vector<BaseCostTerm::ConstPtr> meas_cost_terms_local;
#endif
                };

                // Perform the parallel reduction
                ReductionResult final_result = tbb::parallel_reduce(
                    // Range with optimal grainsize
                    tbb::blocked_range<int>(0, (int)keypoints.size(), options_.sequential_threshold),
                    // Identity element (starts with empty vectors)
                    ReductionResult{},
                    // Body lambda (processes a sub-range)
                    [&](const tbb::blocked_range<int> &range, ReductionResult res) -> ReductionResult {
                        // Pre-allocate thread-local vectors
#if USE_P2P_SUPER_COST_TERM
                        res.p2p_matches_local.reserve(range.size());
#else
                        res.meas_cost_terms_local.reserve(range.size());
#endif
                        for (int i = range.begin(); i != range.end(); ++i) {
                            const auto &keypoint = keypoints[i];
                            const auto &pt_keypoint = keypoint.pt;
                            ArrayVector3d vector_neighbors =
                                map_.searchNeighbors(pt_keypoint, nb_voxels_visited, options_.size_voxel_map, options_.max_number_neighbors);

                            if ((int)vector_neighbors.size() < kMinNumNeighbors) {
                                continue;
                            }

                            auto neighborhood = compute_neighborhood_distribution(vector_neighbors, options_.sequential_threshold);
                            const double planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);
                            const double weight = planarity_weight;
                            const double dist_to_plane = std::abs((keypoint.pt - vector_neighbors[0]).transpose() * neighborhood.normal);

                            if (dist_to_plane < options_.p2p_max_dist) {
#if USE_P2P_SUPER_COST_TERM
                                res.p2p_matches_local.emplace_back(P2PMatch(
                                    keypoint.timestamp, vector_neighbors[0], weight * neighborhood.normal, keypoint.raw_pt));
#else
                                Eigen::Vector3d closest_pt = vector_neighbors[0];
                                Eigen::Vector3d closest_normal = weight * neighborhood.normal;
                                Eigen::Matrix3d W = (closest_normal * closest_normal.transpose() + 1e-5 * Eigen::Matrix3d::Identity());
                                const auto noise_model = finalicp::StaticNoiseModel<3>::MakeShared(W, NoiseType::INFORMATION);
                                const auto &T_mr_intp_eval = T_mr_intp_eval_map.at(keypoint.timestamp);
                                const auto error_func = p2p::p2pError(T_mr_intp_eval, closest_pt, keypoint.raw_pt);
                                error_func->setTime(Time(keypoint.timestamp));

                                const auto loss_func = [this]() -> BaseLossFunc::Ptr {
                                switch (options_.p2p_loss_func) {
                                    case lidarinertialodom::LOSS_FUNC::L2: return L2LossFunc::MakeShared();
                                    case lidarinertialodom::LOSS_FUNC::DCS: return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
                                    case lidarinertialodom::LOSS_FUNC::CAUCHY: return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
                                    case lidarinertialodom::LOSS_FUNC::GM: return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
                                    default: return nullptr;
                                }
                                }();
                                res.meas_cost_terms_local.emplace_back(finalicp::WeightedLeastSqCostTerm<3>::MakeShared(error_func, noise_model, loss_func));
#endif
                            }
                        }
                        return res;
                    },
                    // Joining lambda (merges two thread-local results)
                    [](ReductionResult a, const ReductionResult &b) -> ReductionResult {
                        a.p2p_matches_local.insert(a.p2p_matches_local.end(), b.p2p_matches_local.begin(), b.p2p_matches_local.end());
#if !USE_P2P_SUPER_COST_TERM
                        a.meas_cost_terms_local.insert(a.meas_cost_terms_local.end(), b.meas_cost_terms_local.begin(), b.meas_cost_terms_local.end());
#endif
                        return a;
                    });

                // Efficiently move results from the reduction into the main vectors
                p2p_matches = std::move(final_result.p2p_matches_local);
#if !USE_P2P_SUPER_COST_TERM
                meas_cost_terms = std::move(final_result.meas_cost_terms_local);
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

#ifdef DEBUG
            // [ADDED DEBUG] Print the number of matches found before checking
            std::cout << "[ICP DEBUG] Found " << N_matches << " point-to-plane matches." << std::endl;
#endif
            p2p_super_cost_term->initP2PMatches(); // Initialize point-to-plane matches
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: add meas_cost_terms total: " << meas_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : meas_cost_terms) {
                problem->addCostTerm(cost); // Add point-to-plane cost terms | this is only if not using p2psupercostterm
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: add imu_cost_terms total: " << imu_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : imu_cost_terms) {
                problem->addCostTerm(cost); // Add IMU cost terms
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: add pose_meas_cost_terms total: " << pose_meas_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : pose_meas_cost_terms) {
                problem->addCostTerm(cost); // Add pose measurement cost terms
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: add imu_prior_cost_terms total: " << imu_prior_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : imu_prior_cost_terms) {
                problem->addCostTerm(cost); // Add IMU bias prior cost terms
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: add T_mi_prior_cost_terms total: " << T_mi_prior_cost_terms.size() << std::endl;
#endif
            for (const auto& cost : T_mi_prior_cost_terms) {
                problem->addCostTerm(cost); // Add T_mi prior cost terms
            }
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: add p2p_super_cost_term total: " << p2p_super_cost_term.size() << std::endl;
#endif
            problem->addCostTerm(p2p_super_cost_term); // Add point-to-plane super cost term
#ifdef DEBUG
                std::cout << "[ICP DEBUG] problem: add imu_super_cost_term total: " << imu_super_cost_term.size() << std::endl;
#endif
            if (options_.use_imu) {
                problem->addCostTerm(imu_super_cost_term); // Add IMU super cost term
            }

#ifdef DEBUG
            timer[1].second->stop(); // Stop association timer
#endif

            // Step 46: Check for sufficient keypoints
            // Ensures enough matches for reliable optimization
            if (N_matches < options_.min_number_keypoints) {
#ifdef DEBUG
                std::cout << "[ICP DEBUG] CRITICAL: not enough keypoints selected in icp !" << std::endl;
                std::cout << "[ICP DEBUG] Found: " << N_matches << " point-to-plane matches (residuals)." << std::endl;
                std::cout << "[ICP DEBUG] Minimum required: " << options_.min_number_keypoints << std::endl;
                std::cout << "[ICP DEBUG] Map size: " << map_.size() << " points." << std::endl;
#endif
                icp_success = false;
                break; // Exit the ICP loop if insufficient keypoints
            }

            // Step 47: Solve the optimization problem
            // Uses Gauss-Newton solver to refine the trajectory
#ifdef DEBUG
            timer[2].second->start(); // Start optimization timer
            std::cout << "[ICP DEBUG] Calling solver.optimize()... Number of variables: " << problem->getStateVector()->getNumberOfStates() << ", Number of cost terms: " << problem->getNumberOfCostTerms() << std::endl;
#endif
            
            finalicp::GaussNewtonSolverNVA::Params params;
            params.verbose = options_.verbose;
            params.max_iterations = static_cast<unsigned int>(options_.max_iterations);
            params.line_search = (iter >= 2 && options_.use_line_search); // Enable line search after 2 iterations if configured
            if (swf_inside_icp) {params.reuse_previous_pattern = false;}
            finalicp::GaussNewtonSolverNVA solver(*problem, params);

            // --- WRAP SOLVER CALL IN A TRY-CATCH BLOCK ---
            try {
                solver.optimize();
            } catch (const finalicp::decomp_failure& e) {
#ifdef DEBUG
                std::cerr << "[ICP DEBUG] CATASTROPHIC SOLVER FAILURE: " << e.what() << std::endl;
                std::cerr << "[ICP DEBUG] This usually means the Hessian matrix is not positive-definite, likely due to an ill-conditioned problem (e.g., bad geometry, insufficient constraints/priors)." << std::endl;
#endif
                icp_success = false;
                break;
            } catch (const std::exception& e) {
#ifdef DEBUG
                std::cerr << "[ICP DEBUG] AN UNEXPECTED EXCEPTION OCCURRED DURING SOLVER::OPTIMIZE: " << e.what() << std::endl;
#endif
                icp_success = false;
                break;
            }

#ifdef DEBUG
            timer[2].second->stop(); // Stop optimization timer
            std::cout << "[ICP DEBUG] Solver finished." << std::endl;
#endif

            // Step 48: Update the trajectory estimate and check convergence
            // Computes differences in pose, velocity, and acceleration to determine if converged
#ifdef DEBUG
            timer[3].second->start(); // Start alignment timer
            std::cout << "[ICP DEBUG] Updating State & Checking Convergence" << std::endl;
#endif

#ifdef DEBUG
            // [ADDED DEBUG] Header for this block to show the current iteration
            std::cout << "[ICP DEBUG] Updating State & Checking Convergence (Iteration " << iter << ")" << std::endl;
#endif

            double diff_trans = 0.0, diff_rot = 0.0, diff_vel = 0.0, diff_acc = 0.0;

            // Update begin pose
            finalicp::traj::Time curr_begin_slam_time(static_cast<double>(trajectory_[index_frame].begin_timestamp));
            const Eigen::Matrix4d begin_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_begin_slam_time)->value().inverse().matrix();
            const Eigen::Matrix4d begin_T_ms = begin_T_mr * options_.T_sr.inverse();
            diff_trans += (current_estimate.begin_t - begin_T_ms.block<3, 1>(0, 3)).norm();
            diff_rot += AngularDistance(current_estimate.begin_R, begin_T_ms.block<3, 3>(0, 0));

#ifdef DEBUG
            // [ADDED DEBUG] Print the change in the beginning pose translation
            std::cout << "[ICP DEBUG] Begin Translation | Old: " << current_estimate.begin_t.transpose()
                    << " | New: " << begin_T_ms.block<3, 1>(0, 3).transpose() << std::endl;
#endif

            // Update end pose
            finalicp::traj::Time curr_end_slam_time(static_cast<double>(trajectory_[index_frame].end_timestamp));
            const Eigen::Matrix4d end_T_mr = SLAM_TRAJ->getPoseInterpolator(curr_end_slam_time)->value().inverse().matrix();
            const Eigen::Matrix4d end_T_ms = end_T_mr * options_.T_sr.inverse();
            diff_trans += (current_estimate.end_t - end_T_ms.block<3, 1>(0, 3)).norm();
            diff_rot += AngularDistance(current_estimate.end_R, end_T_ms.block<3, 3>(0, 0));

#ifdef DEBUG
            // [ADDED DEBUG] Print the change in the ending pose translation
            std::cout << "[ICP DEBUG] End Translation   | Old: " << current_estimate.end_t.transpose()
                    << " | New: " << end_T_ms.block<3, 1>(0, 3).transpose() << std::endl;
#endif

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
                    throw std::runtime_error("(1) Mid time not within knot times in icp: " + std::to_string(curr_mid_slam_time.seconds()) + " ,at frame: " + std::to_string(index_frame));
                }
                current_estimate.mid_b = trajectory_vars_[i].imu_biases->value();
            }
            // --- [ADD DEBUG CHECKS AFTER CALCULATING DIFFS] ---
#ifdef DEBUG
            if (!std::isfinite(diff_rot) || !std::isfinite(diff_trans) || !std::isfinite(diff_vel) || !std::isfinite(diff_acc)) {
                std::cout << "[ICP DEBUG] CRITICAL: Non-finite difference detected after optimization! The state is likely corrupted with NaNs." << std::endl;
            }
            std::cout << "[ICP DEBUG] State Change   | d_rot: " << diff_rot << ", d_trans: " << diff_trans << ", d_vel: " << diff_vel << ", d_acc: " << diff_acc << std::endl;
            std::cout << "[ICP DEBUG] End Pose (t)   | " << current_estimate.end_t.transpose() << std::endl;
#endif


            // Check convergence
            if ((index_frame > 1) &&
                (diff_rot < options_.threshold_orientation_norm &&
                diff_trans < options_.threshold_translation_norm &&
                diff_vel < options_.threshold_translation_norm * 10.0 + options_.threshold_orientation_norm * 10.0 &&
                diff_acc < options_.threshold_translation_norm * 100.0 + options_.threshold_orientation_norm * 100.0)){
#ifdef DEBUG
                std::cout << "[ICP DEBUG] Finished with N=" << iter << " ICP iterations" << std::endl;
#endif
                if (options_.break_icp_early) {
                    break; // Exit loop if converged and early breaking is enabled
                }
            }

            // Re-transform keypoints for the next iteration
#ifdef DEBUG
            timer[0].second->start(); // Start update transform timer
            std::cout << "[ICP DEBUG] Performing initial keypoint transformation after ICP loop." << std::endl;
#endif
            transform_keypoints(); // Updates keypoints.pt using the latest trajectory
#ifdef DEBUG
            timer[0].second->stop(); // Stop update transform timer
            timer[3].second->stop(); // Stop alignment timer
#endif
        // ################################################################################
        } // End ICP optimization loop ################################################################################
        // ################################################################################

        // Step 49: Add cost terms to the sliding window filter
        // Includes state priors, point-to-plane, IMU, pose, and T_mi cost terms
        SLAM_TRAJ->addPriorCostTerms(*sliding_window_filter_); // Add state priors (e.g., for initial state x_0)
#ifdef DEBUG
            std::cout << "[ICP DEBUG] SLAM_TRAJ: addPriorCostTerms with sliding_window_filter_" << std::endl;
            std::cout << "[ICP DEBUG] sliding_window_filter_: add prior_cost_terms total: " << prior_cost_terms.size() << std::endl;
#endif
        for (const auto& prior_cost_term : prior_cost_terms) {
            sliding_window_filter_->addCostTerm(prior_cost_term); // Add prior cost terms | not really adding much
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG] sliding_window_filter_: add meas_cost_terms total: " << meas_cost_terms.size() << std::endl;
#endif
        for (const auto& meas_cost_term : meas_cost_terms) {
            sliding_window_filter_->addCostTerm(meas_cost_term); // Add point-to-plane cost terms | this is only if not using p2psupercostterm
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG] sliding_window_filter_: add pose_meas_cost_terms total: " << pose_meas_cost_terms.size() << std::endl;
#endif
        for (const auto& pose_cost : pose_meas_cost_terms) {
            sliding_window_filter_->addCostTerm(pose_cost); // Add pose measurement cost terms
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG] sliding_window_filter_: add imu_cost_terms total: " << imu_cost_terms.size() << std::endl;
#endif
        for (const auto& imu_cost : imu_cost_terms) {
            sliding_window_filter_->addCostTerm(imu_cost); // Add IMU cost terms
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG] sliding_window_filter_: add imu_prior_cost_terms total: " << imu_prior_cost_terms.size() << std::endl;
#endif
        for (const auto& imu_prior_cost : imu_prior_cost_terms) {
            sliding_window_filter_->addCostTerm(imu_prior_cost); // Add IMU bias prior cost terms
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG] sliding_window_filter_: add T_mi_prior_cost_terms total: " << T_mi_prior_cost_terms.size() << std::endl;
#endif
        for (const auto& T_mi_prior_cost : T_mi_prior_cost_terms) {
            sliding_window_filter_->addCostTerm(T_mi_prior_cost); // Add T_mi prior cost terms
        }
#ifdef DEBUG
            std::cout << "[ICP DEBUG] sliding_window_filter_: add p2p_super_cost_term total: " << p2p_super_cost_term.size() << std::endl;
#endif
        sliding_window_filter_->addCostTerm(p2p_super_cost_term); // Add point-to-plane super cost term
        if (options_.use_imu) {
#ifdef DEBUG
            std::cout << "[ICP DEBUG] sliding_window_filter_: add imu_super_cost_term total: " << imu_super_cost_term.size() << std::endl;
#endif
            sliding_window_filter_->addCostTerm(imu_super_cost_term); // Add IMU super cost term
        }

#ifdef DEBUG
        std::cout << "[ICP DEBUG] sliding_window_filter_: number of variables: " << sliding_window_filter_->getNumberOfVariables() << std::endl;
        std::cout << "[ICP DEBUG] sliding_window_filter_: number of cost terms: " << sliding_window_filter_->getNumberOfCostTerms() << std::endl;
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
         // ADD THIS LINE

        finalicp::Covariance covariance(solver);
#ifdef DEBUG
        std::cout << "[ICP DEBUG] SLAM_TRAJ: getCovariance." << std::endl;
#endif
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
                throw std::runtime_error("(2) Mid time not within knot times in icp: " + std::to_string(curr_mid_slam_time.seconds()) + " at frame: " + std::to_string(index_frame));
            }

            const auto bias_intp_eval = finalicp::vspace::VSpaceInterpolator<6>::MakeShared(curr_mid_slam_time, trajectory_vars_[i].imu_biases, trajectory_vars_[i].time, trajectory_vars_[i + 1].imu_biases, trajectory_vars_[i + 1].time);
            current_estimate.mid_b = bias_intp_eval->value();
#ifdef DEBUG
            std::cout << "[ICP DEBUG] mid_T_mi: " << current_estimate.mid_T_mi << std::endl;
            std::cout << "[ICP DEBUG] b_begin: " << trajectory_vars_[i].imu_biases->value().transpose() << std::endl;
            std::cout << "[ICP DEBUG] b_end: " << trajectory_vars_[i + 1].imu_biases->value().transpose() << std::endl;
#endif
        }

        // Step 54: Validate final estimate parameters
        // Ensures keypoints, velocities, and accelerations are valid
#ifdef DEBUG
        std::cout << "[ICP DEBUG] ESTIMATED PARAMETER" << std::endl;
        std::cout << "[ICP DEBUG] Number of keypoints used in CT-ICP : " << N_matches << std::endl;
        std::cout << "[ICP DEBUG] v_begin: " << v_begin.transpose() << std::endl;
        std::cout << "[ICP DEBUG] v_end: " << v_end.transpose() << std::endl;
        std::cout << "[ICP DEBUG] a_begin: " << a_begin.transpose() << std::endl;
        std::cout << "[ICP DEBUG] a_end: " << a_end.transpose() << std::endl;
        std::cout << "[ICP DEBUG] Number iterations CT-ICP : " << options_.num_iters_icp << std::endl;
        std::cout << "[ICP DEBUG] Translation Begin: " << trajectory_[index_frame].begin_t.transpose() << std::endl;
        std::cout << "[ICP DEBUG] Translation End: " << trajectory_[index_frame].end_t.transpose() << std::endl;
#endif

#ifdef DEBUG
        std::cout << "[ICP DEBUG] INNER LOOP TIMERS" << std::endl;
        for (size_t i = 0; i < timer.size(); i++) {
            std::cout << "Elapsed " << timer[i].first << *(timer[i].second) << std::endl;
        }
        // [DEBUG] Final status report before returning
        std::cout << "[ICP DEBUG] Finished ICP for frame " << index_frame << ". Success: " << (icp_success ? "true" : "false") << std::endl;
#endif

        // Step 55: Return success status
        // Completes the icp function, returning whether the frame was successfully processed
        return icp_success;
    }
}  // namespace stateestimate