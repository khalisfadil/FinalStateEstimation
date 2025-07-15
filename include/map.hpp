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

        // Hash compare struct for TBB's concurrent_hash_map
        struct VoxelTBBHashCompare {
            size_t hash(const Voxel& v) const { return VoxelHash()(v); }
            bool equal(const Voxel& a, const Voxel& b) const { return a == b; }
        };
        
        // --- Constructors and Basic Setup ---

        Map() = default;

        explicit Map(int default_lifetime, int default_sequential_threshold, unsigned int default_num_threads)
            : default_lifetime_(default_lifetime), 
              default_sequential_threshold_(default_sequential_threshold), 
              default_num_threads_(default_num_threads) {}

        void initialize(int default_lifetime, int default_sequential_threshold, unsigned int default_num_threads) {
            default_lifetime_ = default_lifetime;
            default_sequential_threshold_ = default_sequential_threshold; 
            default_num_threads_ = default_num_threads;
        }

        void clear() {
            voxel_map_.clear();
            total_points_ = 0;
        }

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

        // Returns the total number of points in the map in O(1) time.
        [[nodiscard]] size_t size() const {
            return total_points_;
        }

        // --- Map Modification ---

        void add(const std::vector<Point3D>& points, double voxel_size, int max_num_points_in_voxel,
                 double min_distance_points, int min_num_points = 0);

        void add(const Eigen::Vector3d& point, double voxel_size, int max_num_points_in_voxel,
                 double min_distance_points, int min_num_points = 0);

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

        // --- Neighbor Search ---

        using pair_distance_t = std::tuple<double, Eigen::Vector3d, Voxel>;

        struct Comparator {
            bool operator()(const pair_distance_t& left, const pair_distance_t& right) const {
                return std::get<0>(left) < std::get<0>(right); // Max-heap: top is largest distance
            }
        };

        using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, Comparator>;

        ArrayVector3d searchNeighbors(const Eigen::Vector3d& point, int nb_voxels_visited, double size_voxel_map,
                                      int max_num_neighbors, int threshold_voxel_capacity = 1, std::vector<Voxel>* voxels = nullptr);

    private:
        // --- Private Members ---
        VoxelHashMap voxel_map_;
        int default_lifetime_ = 20;
        int default_sequential_threshold_ = 500;
        unsigned int default_num_threads_ = 4;
        size_t total_points_ = 0; // Cached total points for fast O(1) size queries.

        // --- Private Helpers for `add` ---
        void merge_into_existing_voxel(VoxelBlock& block, const std::vector<Eigen::Vector3d>& new_points,
                                       double min_distance_points, int min_num_points);
        
        void create_and_fill_new_voxel(const Voxel& key, const std::vector<Eigen::Vector3d>& new_points,
                                       int max_points, double min_distance_points, int min_num_points);

        bool add_point_to_block_with_filter(VoxelBlock& block, const Eigen::Vector3d& point,
                                            double min_distance_points, int min_num_points);
    };
    
    // --- Implementation of Member Functions ---

    inline void Map::add(const std::vector<Point3D>& points, double voxel_size, int max_num_points_in_voxel,
                         double min_distance_points, int min_num_points) {
        if (points.empty()) return;

        // Use sequential method for small batches to avoid parallel overhead
        if (points.size() < default_sequential_threshold_) {
            for (const auto& point_item : points) {
                add(point_item.pt, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points);
            }
            return;
        }
        
        // Parallel "Map" phase: Group points by target voxel
        using TempMap = tbb::concurrent_hash_map<Voxel, std::vector<Eigen::Vector3d>, VoxelTBBHashCompare>;
        TempMap temp_voxel_map;
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, default_num_threads_);
        
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

    inline void Map::merge_into_existing_voxel(VoxelBlock& block, const std::vector<Eigen::Vector3d>& new_points,
                                              double min_distance_points, int min_num_points) {
        for (const auto& point : new_points) {
            add_point_to_block_with_filter(block, point, min_distance_points, min_num_points);
        }
        block.life_time = default_lifetime_;
    }

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

    inline ArrayVector3d Map::searchNeighbors(const Eigen::Vector3d& point, int nb_voxels_visited, double size_voxel_map,
                                              int max_num_neighbors, int threshold_voxel_capacity, std::vector<Voxel>* voxels) {
        if (voxels) voxels->clear();
        
        const Voxel center = Voxel::Coordinates(point, size_voxel_map);
        priority_queue_t priority_queue;
        double max_dist_sq = std::numeric_limits<double>::infinity();

        auto process_voxel = [&](const Voxel& voxel) {
            auto search = voxel_map_.find(voxel);
            if (search == voxel_map_.end() || search->second.NumPoints() < threshold_voxel_capacity) {
                return;
            }
            const auto& block = search->second;

            for (const auto& neighbor : block.points) {
                double dist_sq = (neighbor - point).squaredNorm();
                if (priority_queue.size() < static_cast<size_t>(max_num_neighbors)) {
                    priority_queue.emplace(dist_sq, neighbor, voxel);
                    if (priority_queue.size() == static_cast<size_t>(max_num_neighbors)) {
                        max_dist_sq = std::get<0>(priority_queue.top());
                    }
                } else if (dist_sq < max_dist_sq) {
                    priority_queue.pop();
                    priority_queue.emplace(dist_sq, neighbor, voxel);
                    max_dist_sq = std::get<0>(priority_queue.top());
                }
            }
        };

        const double half_size = size_voxel_map / 2.0;

        // Traverse concentric shells (Chebyshev distance)
        for (int d = 0; d <= nb_voxels_visited; ++d) {
            // Early exit if the minimum distance to the current shell is already
            // greater than the furthest neighbor we've found.
            double shell_min_dist = (d > 0 ? (d - 1) * size_voxel_map : 0.0);
            if (shell_min_dist * shell_min_dist > max_dist_sq) break;

            for (int dx = -d; dx <= d; ++dx) {
                for (int dy = -d; dy <= d; ++dy) {
                    for (int dz = -d; dz <= d; ++dz) {
                        // Process only the boundary of the cube at distance `d`
                        if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != d) continue;

                        Voxel voxel{center.x + dx, center.y + dy, center.z + dz};

                        // Voxel-level pruning: check the minimum possible distance from the
                        // query point to this voxel's bounding box.
                        double vx_center = voxel.x * size_voxel_map + half_size;
                        double vy_center = voxel.y * size_voxel_map + half_size;
                        double vz_center = voxel.z * size_voxel_map + half_size;

                        double dx_min = std::max(0.0, std::abs(point.x() - vx_center) - half_size);
                        double dy_min = std::max(0.0, std::abs(point.y() - vy_center) - half_size);
                        double dz_min = std::max(0.0, std::abs(point.z() - vz_center) - half_size);
                        double voxel_min_dist_sq = dx_min * dx_min + dy_min * dy_min + dz_min * dz_min;

                        if (voxel_min_dist_sq > max_dist_sq) continue;

                        process_voxel(voxel);
                    }
                }
            }
        }

        // Extract results from the priority queue
        const auto size = priority_queue.size();
        ArrayVector3d closest_neighbors(size);
        if (voxels) voxels->resize(size);

        for (size_t i = size; i > 0; --i) {
            closest_neighbors[i - 1] = std::get<1>(priority_queue.top());
            if (voxels) (*voxels)[i - 1] = std::get<2>(priority_queue.top());
            priority_queue.pop();
        }
        return closest_neighbors;
    }

} // namespace stateestimate