#pragma once

#include <queue>
#include <Eigen/Dense>
#include <robin_map.h>
#include <point.hpp>
#include <voxel.hpp>

namespace stateestimate {

    class Map {

        public:

            // -----------------------------------------------------------------------------

            Map() = default;

            // -----------------------------------------------------------------------------

            explicit Map(int default_lifetime) : default_lifetime_(default_lifetime) {}

            // -----------------------------------------------------------------------------

            [[nodiscard]] ArrayVector3d pointcloud() const {
                ArrayVector3d points;
                points.reserve(size());
                for (const auto& [_, block] : voxel_map_) {
                    points.insert(points.end(), block.points.begin(), block.points.end());
                }
                return points;
            }

            // -----------------------------------------------------------------------------

            [[nodiscard]] size_t size() const {
                size_t map_size = 0;
                for (const auto& [_, block] : voxel_map_) {
                    map_size += block.NumPoints();
                }
                return map_size;
            }

            // -----------------------------------------------------------------------------

            void remove(const Eigen::Vector3d& location, double distance) {
                std::vector<Voxel> voxels_to_erase;
                voxels_to_erase.reserve(voxel_map_.size() / 10); // Heuristic reservation
                const double sq_distance = distance * distance;
                for (const auto& [voxel, block] : voxel_map_) {
                    if (!block.points.empty() && (block.points[0] - location).squaredNorm() > sq_distance) {
                        voxels_to_erase.push_back(voxel);
                    }
                }
                for (const auto& voxel : voxels_to_erase) {
                    voxel_map_.erase(voxel);
                }
            }

            // -----------------------------------------------------------------------------

            void update_and_filter_lifetimes() {
                std::vector<Voxel> voxels_to_erase;
                for (VoxelHashMap::iterator it = voxel_map_.begin(); it != voxel_map_.end(); it++) {
                    auto& voxel_block = it.value();
                    voxel_block.life_time -= 1;
                    if (voxel_block.life_time <= 0) voxels_to_erase.push_back(it->first);
                }
                for (auto &vox : voxels_to_erase) voxel_map_.erase(vox);
            }

            // -----------------------------------------------------------------------------

            void setDefaultLifeTime(int default_lifetime) { 
                default_lifetime_ = default_lifetime; 
            }

            // -----------------------------------------------------------------------------

            void clear() { 
                voxel_map_.clear(); 
            }

            // -----------------------------------------------------------------------------

            void add(const std::vector<Point3D>& points, double voxel_size, int max_num_points_in_voxel,
                    double min_distance_points, int min_num_points = 0) {
                for (const auto& point : points) {
                    add(point.pt, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points);
                }
            }

            // -----------------------------------------------------------------------------

            void add(const ArrayVector3d& points, double voxel_size, int max_num_points_in_voxel, 
                    double min_distance_points) {
                for (const auto& point : points) {
                    add(point, voxel_size, max_num_points_in_voxel, min_distance_points);
                }
            }

            // -----------------------------------------------------------------------------

            void add(const Eigen::Vector3d &point, double voxel_size, int max_num_points_in_voxel, 
                                        double min_distance_points, int min_num_points = 0) {
                int16_t kx = static_cast<int16_t>(point[0] / voxel_size);
                int16_t ky = static_cast<int16_t>(point[1] / voxel_size);
                int16_t kz = static_cast<int16_t>(point[2] / voxel_size);

                VoxelHashMap::iterator search = voxel_map_.find(Voxel(kx, ky, kz));
                if (search != voxel_map_.end()) {

                    auto &voxel_block = (search.value());

                    if (!voxel_block.IsFull()) {
                        double sq_dist_min_to_points = 10 * voxel_size * voxel_size;

                        for (int i(0); i < voxel_block.NumPoints(); ++i) {
                            auto &_point = voxel_block.points[i];
                            double sq_dist = (_point - point).squaredNorm();
                            if (sq_dist < sq_dist_min_to_points) {
                                sq_dist_min_to_points = sq_dist;
                            }
                        }

                        if (sq_dist_min_to_points > (min_distance_points * min_distance_points)) {
                            if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points) {
                                voxel_block.AddPoint(point);
                            }
                        }
                    }
                    voxel_block.life_time = default_lifetime_;
                } else {
                    if (min_num_points <= 0) {
                        // Do not add points (avoids polluting the map)
                        VoxelBlock block(max_num_points_in_voxel);
                        block.AddPoint(point);
                        block.life_time = default_lifetime_;
                        voxel_map_[Voxel(kx, ky, kz)] = std::move(block);
                    }
                }
            }

            // -----------------------------------------------------------------------------

            using pair_distance_t = std::tuple<double, Eigen::Vector3d, Voxel>;

            // -----------------------------------------------------------------------------

            struct Comparator {
                bool operator()(const pair_distance_t& left, const pair_distance_t& right) const {
                    return std::get<0>(left) > std::get<0>(right); // Min-heap
                }
            };

            // -----------------------------------------------------------------------------

            using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, Comparator>;

            // -----------------------------------------------------------------------------

            ArrayVector3d searchNeighbors(const Eigen::Vector3d& point, int nb_voxels_visited, double size_voxel_map,
                                            int max_num_neighbors, int threshold_voxel_capacity = 1, std::vector<Voxel>* voxels = nullptr) {
                // Reserve space for output
                if (voxels) voxels->reserve(max_num_neighbors);

                // Compute center voxel coordinates
                const Voxel center = Voxel::Coordinates(point, size_voxel_map);
                const int16_t kx = center.x;
                const int16_t ky = center.y;
                const int16_t kz = center.z;

                // Initialize min-heap for closest points
                priority_queue_t priority_queue;

                // Precompute search bounds
                const int16_t x_min = kx - nb_voxels_visited;
                const int16_t x_max = kx + nb_voxels_visited + 1;
                const int16_t y_min = ky - nb_voxels_visited;
                const int16_t y_max = ky + nb_voxels_visited + 1;
                const int16_t z_min = kz - nb_voxels_visited;
                const int16_t z_max = kz + nb_voxels_visited + 1;

                // Track max distance for pruning
                double max_distance = std::numeric_limits<double>::max();

                // Spiral traversal: process voxels layer by layer
                for (int16_t d = 0; d <= nb_voxels_visited; ++d) {
                    for (int16_t dx = -d; dx <= d; ++dx) {
                        for (int16_t dy = -d; dy <= d; ++dy) {
                            for (int16_t dz = -d; dz <= d; ++dz) {
                                // Only process boundary voxels at distance d
                                if (std::abs(dx) != d && std::abs(dy) != d && std::abs(dz) != d) continue;

                                Voxel voxel{kx + dx, ky + dy, kz + dz};

                                // Skip out-of-bounds voxels
                                if (voxel.x < x_min || voxel.x >= x_max ||
                                    voxel.y < y_min || voxel.y >= y_max ||
                                    voxel.z < z_min || voxel.z >= z_max) continue;

                                // Early pruning: skip voxels too far away
                                Eigen::Vector3d voxel_center(
                                    voxel.x * size_voxel_map + size_voxel_map / 2.0,
                                    voxel.y * size_voxel_map + size_voxel_map / 2.0,
                                    voxel.z * size_voxel_map + size_voxel_map / 2.0
                                );
                                
                                if ((voxel_center - point).norm() > max_distance + size_voxel_map) continue;

                                // Look up voxel in map
                                auto search = voxel_map_.find(voxel);
                                if (search == voxel_map_.end()) continue;

                                const auto& voxel_block = search->second;
                                if (voxel_block.NumPoints() < threshold_voxel_capacity) continue;

                                // Process points in voxel
                                for (const auto& neighbor : voxel_block.points) {
                                    double distance = (neighbor - point).norm();
                                    if (priority_queue.size() < static_cast<size_t>(max_num_neighbors)) {
                                        priority_queue.emplace(distance, neighbor, voxel);
                                        if (priority_queue.size() == static_cast<size_t>(max_num_neighbors)) {
                                            max_distance = std::get<0>(priority_queue.top());
                                        }
                                    } else if (distance < max_distance) {
                                        priority_queue.pop();
                                        priority_queue.emplace(distance, neighbor, voxel);
                                        max_distance = std::get<0>(priority_queue.top());
                                    }
                                }
                            }
                        }
                    }
                }

                // Extract results
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

        private:
            VoxelHashMap voxel_map_;
            int default_lifetime_ = 10;
    };

} // namespace stateestimate