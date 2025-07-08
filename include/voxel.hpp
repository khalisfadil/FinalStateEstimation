#pragma once

#include <Eigen/Dense>
#include <robin_map.h>
#include <point.hpp>

namespace stateestimate {

    using ArrayVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

    struct Voxel {
        int32_t x = 0;
        int32_t y = 0;
        int32_t z = 0;

        Voxel() = default;
        Voxel(int32_t x, int32_t y, int32_t z) : x(x), y(y), z(z) {}

        bool operator==(const Voxel& other) const {
            return x == other.x && y == other.y && z == other.z;
        }

        // Note: operator< is unused in tsl::robin_map but can be added for ordered containers:
        bool operator<(const Voxel& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }

        static Voxel Coordinates(const Eigen::Vector3d& point, double voxel_size) {
            return {
                static_cast<int32_t>((point.x()) / voxel_size),
                static_cast<int32_t>((point.y()) / voxel_size),
                static_cast<int32_t>((point.z()) / voxel_size)
            };
        }
    };

    struct VoxelHash {
        size_t operator()(const Voxel& voxel) const {
            uint64_t packed = (static_cast<uint64_t>(static_cast<int32_t>(voxel.x)) << 0) |
                            (static_cast<uint64_t>(static_cast<int32_t>(voxel.y)) << 16) |
                            (static_cast<uint64_t>(static_cast<int32_t>(voxel.z)) << 32);
            packed ^= packed >> 33;
            packed *= 0x9E3779B97F4A7C15ULL;
            packed ^= packed >> 29;
            packed *= 0xBF58476D1CE4E5B9ULL;
            packed ^= packed >> 27;
            return static_cast<size_t>(packed);
        }
    };

    struct VoxelBlock {
            explicit VoxelBlock(int32_t capacity = 20) : capacity_(capacity) {
                points.reserve(capacity);
            }

            bool IsFull() const {
                return points.size() >= static_cast<size_t>(capacity_);
            }

            bool AddPoint(const Eigen::Vector3d& point) {
                if (IsFull()) return false;
                points.push_back(point);
                return true;
            }

            int NumPoints() const {
                return static_cast<int>(points.size());
            }

            int32_t Capacity() const {
                return capacity_;
            }

            ArrayVector3d points;
            int32_t life_time = 0; // 0 means no expiration

        private:
            int32_t capacity_;
    };

    using VoxelHashMap = tsl::robin_map<Voxel, VoxelBlock, VoxelHash>;

} //namespace stateestimate




