#pragma once

#include <vector>
#include <queue>
#include <tuple>
#include <cmath> // For std::floor, std::max, std::abs
#include <limits> // For std::numeric_limits
#include <algorithm> // For std::max

#include <Eigen/Dense>
#include <robin_map.h>

// TBB headers for parallel processing
#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/global_control.h>
#include <tbb/blocked_range.h>

// Your project's custom headers
#include "voxel.hpp"
#include "point.hpp"

namespace stateestimate {

    class Map {
    public:
        // --- Types and Structs ---

        // ########################################################################
        // setInitialPose
        // ########################################################################
        // Hash compare struct for TBB's concurrent_hash_map
        struct VoxelTBBHashCompare {
            size_t hash(const Voxel& v) const { return VoxelHash()(v); }
            bool equal(const Voxel& a, const Voxel& b) const { return a == b; }
        };
        
        // ########################################################################
        // setInitialPose
        // ########################################################################
        // --- Constructors and Basic Setup ---
        Map() = default;

        // ########################################################################
        // setInitialPose
        // ########################################################################
        explicit Map(int default_lifetime, int default_sequential_threshold, unsigned int default_num_threads)
            : default_lifetime_(default_lifetime), 
              default_sequential_threshold_(default_sequential_threshold), 
              default_num_threads_(default_num_threads) {}

        // ########################################################################
        // setInitialPose
        // ########################################################################
        void initialize(int default_lifetime, int default_sequential_threshold, unsigned int default_num_threads) {
            default_lifetime_ = default_lifetime;
            default_sequential_threshold_ = default_sequential_threshold; 
            default_num_threads_ = default_num_threads;
        }

        // ########################################################################
        // setInitialPose
        // ########################################################################
        void clear() {
            voxel_map_.clear();
            total_points_ = 0;
        }

        // ########################################################################
        // setInitialPose
        // ########################################################################
        // --- Point Cloud and Size Queries ---
        // Extracts all points from the map into a single vector. Low overhead.
        [[nodiscard]] ArrayVector3d pointcloud() const {
            ArrayVector3d points;
            points.reserve(size());
            for (const auto& [_, block] : voxel_map_) {
                points.insert(points.end(), block.points.begin(), block.points.end());
            }
            return points;
        }

        // ########################################################################
        // setInitialPose
        // ########################################################################
        // Returns the total number of points in the map in O(1) time.
        [[nodiscard]] size_t size() const {
            return total_points_;
        }

        // ########################################################################
        // setInitialPose
        // ########################################################################
        // --- Map Modification ---
        void add(const std::vector<Point3D>& points, double voxel_size, int max_num_points_in_voxel,
                 double min_distance_points, int min_num_points = 0);
        
        // ########################################################################
        // setInitialPose
        // ########################################################################
        void add(const Eigen::Vector3d& point, double voxel_size, int max_num_points_in_voxel,
                 double min_distance_points, int min_num_points = 0);
        
        // ########################################################################
        // setInitialPose
        // ########################################################################
        // Removes voxels whose representative point is farther than 'distance' from 'location'.
        void remove(const Eigen::Vector3d& location, double distance) {
            std::vector<Voxel> voxels_to_erase;
            voxels_to_erase.reserve(voxel_map_.size() / 10);
            const double sq_distance = distance * distance;

            for (const auto& [voxel, block] : voxel_map_) {
                if (!block.points.empty() && (block.points[0] - location).squaredNorm() > sq_distance) {
                    voxels_to_erase.push_back(voxel);
                }
            }

            for (const auto& voxel : voxels_to_erase) {
                auto it = voxel_map_.find(voxel);
                if (it != voxel_map_.end()) {
                    total_points_ -= it->second.NumPoints(); // Update cached size
                    voxel_map_.erase(it);
                }
            }
        }

        // ########################################################################
        // setInitialPose
        // ########################################################################
        // Decrements lifetimes of all voxels and removes those that have expired.
        void update_and_filter_lifetimes() {
                std::vector<Voxel> voxels_to_erase;
                for (VoxelHashMap::iterator it = voxel_map_.begin(); it != voxel_map_.end(); it++) {
                    auto& voxel_block = it.value();
                    voxel_block.life_time -= 1;
                    if (voxel_block.life_time <= 0) voxels_to_erase.push_back(it->first);
                }
                // Sequentially erase
            for (const auto& voxel : voxels_to_erase) {
                auto it = voxel_map_.find(voxel);
                if (it != voxel_map_.end()) {
                    total_points_ -= it->second.NumPoints();
                    voxel_map_.erase(it);
                }
            }
        }

        // ########################################################################
        // setInitialPose
        // ########################################################################
        void dumpToStream(std::ostream& os, int precision = 12) const {
            os << std::fixed << std::setprecision(precision);
            for (const auto& [_, block] : voxel_map_) {
                for (const auto& point : block.points) {
                    os << point.x() << " " << point.y() << " " << point.z() << "\n";
                }
            }
        }

        // ########################################################################
        // setInitialPose
        // ########################################################################
        // --- Neighbor Search ---
        using pair_distance_t = std::tuple<double, Eigen::Vector3d, Voxel>;

        // ########################################################################
        // setInitialPose
        // ########################################################################
        struct Comparator {
            bool operator()(const pair_distance_t& left, const pair_distance_t& right) const {
                return std::get<0>(left) < std::get<0>(right); // Max-heap: top is largest distance
            }
        };

        // ########################################################################
        // setInitialPose
        // ########################################################################
        using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, Comparator>;

        // ########################################################################
        // setInitialPose
        // ########################################################################
        ArrayVector3d searchNeighbors(const Eigen::Vector3d& point, int nb_voxels_visited, double size_voxel_map,
                                      int max_num_neighbors, int threshold_voxel_capacity = 1, std::vector<Voxel>* voxels = nullptr);

    private:
        // --- Private Members ---
        VoxelHashMap voxel_map_;
        int default_lifetime_ = 20;
        int default_sequential_threshold_ = 1000;
        unsigned int default_num_threads_ = 8;
        std::atomic<size_t> total_points_ = 0;
        
        // ########################################################################
        // setInitialPose
        // ########################################################################
        // --- Private Helpers for `add` ---
        void merge_into_existing_voxel(VoxelBlock& block, const std::vector<Eigen::Vector3d>& new_points,
                                       double min_distance_points, int min_num_points);

        // ########################################################################
        // setInitialPose
        // ########################################################################
        void create_and_fill_new_voxel(const Voxel& key, const std::vector<Eigen::Vector3d>& new_points,
                                       int max_points, double min_distance_points, int min_num_points);
        
        // ########################################################################
        // setInitialPose
        // ########################################################################
        bool add_point_to_block_with_filter(VoxelBlock& block, const Eigen::Vector3d& point,
                                            double min_distance_points, int min_num_points);
    };
    
    // ########################################################################
    // setInitialPose
    // ########################################################################
    // --- Implementation of Member Functions ---
    inline void Map::add(const std::vector<Point3D>& points, double voxel_size, int max_num_points_in_voxel,
                         double min_distance_points, int min_num_points) {
        if (points.empty()) return;

        // Use sequential method for small batches to avoid parallel overhead
        if (points.size() < static_cast<size_t>(default_sequential_threshold_)) {
            for (const auto& point_item : points) {
                add(point_item.pt, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points);
            }
            return;
        }
        
        // Parallel "Map" phase: Group points by target voxel
        using TempMap = tbb::concurrent_hash_map<Voxel, std::vector<Eigen::Vector3d>, VoxelTBBHashCompare>;
        TempMap temp_voxel_map;
        // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, default_num_threads_);
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const Eigen::Vector3d& point = points[i].pt;
                Voxel voxel_key(
                    static_cast<int32_t>(std::floor(point.x() / voxel_size)),
                    static_cast<int32_t>(std::floor(point.y() / voxel_size)),
                    static_cast<int32_t>(std::floor(point.z() / voxel_size))
                );
                TempMap::accessor acc;
                temp_voxel_map.insert(acc, voxel_key);
                acc->second.push_back(point);
            }
        });

        // Sequential "Reduce" phase: Merge grouped points into the main map
        for (const auto& [voxel_key, new_points] : temp_voxel_map) {
            auto it = voxel_map_.find(voxel_key);
            if (it != voxel_map_.end()) {
                merge_into_existing_voxel(it.value(), new_points, min_distance_points, min_num_points);
            } else {
                create_and_fill_new_voxel(voxel_key, new_points, max_num_points_in_voxel, min_distance_points, min_num_points);
            }
        }
    }

    // ########################################################################
    // setInitialPose
    // ########################################################################
    inline void Map::add(const Eigen::Vector3d& point, double voxel_size, int max_num_points_in_voxel,
                        double min_distance_points, int min_num_points) {
        Voxel voxel_key(
            static_cast<int32_t>(std::floor(point.x() / voxel_size)),
            static_cast<int32_t>(std::floor(point.y() / voxel_size)),
            static_cast<int32_t>(std::floor(point.z() / voxel_size))
        );
        auto it = voxel_map_.find(voxel_key);
        if (it != voxel_map_.end()) {
            add_point_to_block_with_filter(it.value(), point, min_distance_points, min_num_points);
        } else {
            if (min_num_points <= 0) {
                VoxelBlock block(max_num_points_in_voxel);
                if (block.AddPoint(point)) { // Add the first point without distance check
                    block.life_time = default_lifetime_;
                    voxel_map_[voxel_key] = std::move(block);
                    total_points_++;
                }
            }
        }
    }

    // ########################################################################
    // setInitialPose
    // ########################################################################
    inline void Map::merge_into_existing_voxel(VoxelBlock& block, const std::vector<Eigen::Vector3d>& new_points,
                                              double min_distance_points, int min_num_points) {
        for (const auto& point : new_points) {
            add_point_to_block_with_filter(block, point, min_distance_points, min_num_points);
        }
        block.life_time = default_lifetime_;
    }

    // ########################################################################
    // setInitialPose
    // ########################################################################
    inline void Map::create_and_fill_new_voxel(const Voxel& key, const std::vector<Eigen::Vector3d>& new_points,
                                              int max_points, double min_distance_points, int min_num_points) {
        if (min_num_points > 0) return;
        VoxelBlock block(max_points);
        for (const auto& point : new_points) {
            // For a new block, min_num_points is 0 for self-check
            add_point_to_block_with_filter(block, point, min_distance_points, 0);
        }
        if (block.NumPoints() > 0) {
            block.life_time = default_lifetime_;
            voxel_map_[key] = std::move(block);
        }
    }

    // ########################################################################
    // setInitialPose
    // ########################################################################
    inline bool Map::add_point_to_block_with_filter(VoxelBlock& block, const Eigen::Vector3d& point,
                                                   double min_distance_points, int min_num_points) {
        if (block.IsFull() || block.NumPoints() < min_num_points) return false;
        
        if (block.NumPoints() > 0) {
            // Use Eigen for fast vectorized distance check
            Eigen::MatrixXd existing_points(block.NumPoints(), 3);
            for (int i = 0; i < block.NumPoints(); ++i) {
                existing_points.row(i) = block.points[i];
            }
            double min_sq_dist = (existing_points.rowwise() - point.transpose()).rowwise().squaredNorm().minCoeff();
            if (min_sq_dist <= (min_distance_points * min_distance_points)) return false;
        }

        if (block.AddPoint(point)) {
            total_points_++;
            return true;
        }
        return false;
    }

    // ########################################################################
    // setInitialPose
    // ########################################################################
    /**
     * @brief Performs a K-Nearest Neighbor (KNN) search for a given point within the voxel map.
     * @details This function efficiently finds the `max_num_neighbors` closest points by searching
     * outward from the query point in concentric shells of voxels. It uses several
     * pruning strategies to speed up the search significantly.
     * @param point The 3D query point.
     * @param nb_voxels_visited The maximum search radius in number of voxels from the center.
     * @param size_voxel_map The side length of a single cubic voxel.
     * @param max_num_neighbors The maximum number of neighbors to return (K in KNN).
     * @param threshold_voxel_capacity A voxel must have at least this many points to be considered.
     * @param voxels An optional output vector to store the Voxel from which each neighbor was found.
     * @return An array of the closest neighboring points found, sorted from nearest to farthest.
     */
    inline ArrayVector3d Map::searchNeighbors(const Eigen::Vector3d& point, int nb_voxels_visited, double size_voxel_map,
                                                int max_num_neighbors, int threshold_voxel_capacity, std::vector<Voxel>* voxels) {
        // Clear the optional output vector if it's provided.
        if (voxels) voxels->clear();
        
        // Determine the voxel containing the query point.
        const Voxel center = Voxel::Coordinates(point, size_voxel_map);
        // A max-priority queue to keep track of the K nearest neighbors. The top element is the farthest.
        priority_queue_t priority_queue;
        // The squared distance to the farthest neighbor found so far.
        double max_dist_sq = std::numeric_limits<double>::infinity();

        // A helper lambda to process all points within a single voxel.
        auto process_voxel = [&](const Voxel& voxel) {
            auto search = voxel_map_.find(voxel);
            // Skip if the voxel doesn't exist or is too sparse.
            if (search == voxel_map_.end() || search->second.NumPoints() < threshold_voxel_capacity) {
                return;
            }
            const auto& block = search->second;

            // Iterate through all points in the voxel and update the KNN priority queue.
            for (const auto& neighbor : block.points) {
                double dist_sq = (neighbor - point).squaredNorm();
                // If the queue isn't full, just add the new point.
                if (priority_queue.size() < static_cast<size_t>(max_num_neighbors)) {
                    priority_queue.emplace(dist_sq, neighbor, voxel);
                    // If the queue just became full, update the max distance.
                    if (priority_queue.size() == static_cast<size_t>(max_num_neighbors)) {
                        max_dist_sq = std::get<0>(priority_queue.top());
                    }
                } else if (dist_sq < max_dist_sq) {
                    // If the new point is closer than the farthest one in the queue, replace it.
                    priority_queue.pop();
                    priority_queue.emplace(dist_sq, neighbor, voxel);
                    max_dist_sq = std::get<0>(priority_queue.top());
                }
            }
        };

        const double half_size = size_voxel_map / 2.0;

        // --- Main Search Loop: Traverse concentric shells of voxels ---
        for (int d = 0; d <= nb_voxels_visited; ++d) {
            // --- Pruning Strategy 1: Shell-level early exit ---
            // Calculate the minimum possible distance to the current shell of voxels.
            double shell_min_dist = (d > 0 ? (d - 1) * size_voxel_map : 0.0);
            // If this minimum distance is greater than our farthest neighbor, we can stop searching.
            if (shell_min_dist * shell_min_dist > max_dist_sq) break;

            // Iterate through the voxels forming the surface of a cube with radius 'd'.
            for (int dx = -d; dx <= d; ++dx) {
                for (int dy = -d; dy <= d; ++dy) {
                    for (int dz = -d; dz <= d; ++dz) {
                        // Process only the boundary of the cube to avoid redundant checks.
                        if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != d) continue;

                        Voxel voxel{center.x + dx, center.y + dy, center.z + dz};

                        // --- Pruning Strategy 2: Voxel-level distance check ---
                        // Calculate the minimum squared distance from the query point to this voxel's bounding box.
                        double vx_center = voxel.x * size_voxel_map + half_size;
                        double vy_center = voxel.y * size_voxel_map + half_size;
                        double vz_center = voxel.z * size_voxel_map + half_size;

                        double dx_min = std::max(0.0, std::abs(point.x() - vx_center) - half_size);
                        double dy_min = std::max(0.0, std::abs(point.y() - vy_center) - half_size);
                        double dz_min = std::max(0.0, std::abs(point.z() - vz_center) - half_size);
                        double voxel_min_dist_sq = dx_min * dx_min + dy_min * dy_min + dz_min * dz_min;

                        // If the closest point in this voxel is farther than our farthest neighbor, skip the whole voxel.
                        if (voxel_min_dist_sq > max_dist_sq) continue;

                        process_voxel(voxel);
                    }
                }
            }
        }

        // --- Result Extraction ---
        // Copy the results from the priority queue into an output vector.
        const auto size = priority_queue.size();
        ArrayVector3d closest_neighbors(size);
        if (voxels) voxels->resize(size);

        // Popping from the max-heap gives elements from farthest to nearest, so we fill the output vector backwards.
        for (size_t i = size; i > 0; --i) {
            closest_neighbors[i - 1] = std::get<1>(priority_queue.top());
            if (voxels) (*voxels)[i - 1] = std::get<2>(priority_queue.top());
            priority_queue.pop();
        }
        return closest_neighbors;
    }
} // namespace stateestimate