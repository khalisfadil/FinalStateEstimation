#pragma once

#include <Eigen/Dense>
#include <robin_map.h> // A high-performance hash map library
#include <point.hpp>   // Assumed to define Point3D

namespace stateestimate {

    /// @brief Type alias for a std::vector of Eigen::Vector3d with an aligned allocator.
    using ArrayVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

    /**
     * @brief Defines a unique 3D grid cell index (voxel) using integer coordinates.
     * @details This is used to discretize a 3D space into a grid.
     */
    struct Voxel {
        int32_t x = 0;
        int32_t y = 0;
        int32_t z = 0;

        Voxel() = default;
        Voxel(int32_t x, int32_t y, int32_t z) : x(x), y(y), z(z) {}

        /// @brief Equality operator for comparing two voxels.
        bool operator==(const Voxel& other) const {
            return x == other.x && y == other.y && z == other.z;
        }

        /// @brief Converts a 3D point's continuous coordinates to discrete voxel coordinates.
        static Voxel Coordinates(const Eigen::Vector3d& point, double voxel_size) {
            return {
                static_cast<int32_t>(point.x() / voxel_size),
                static_cast<int32_t>(point.y() / voxel_size),
                static_cast<int32_t>(point.z() / voxel_size)
            };
        }
    };

    /**
     * @brief A custom hash function for the Voxel struct.
     * @details This allows Voxel objects to be used efficiently as keys in hash maps.
     * It packs the coordinates and applies mixing functions to ensure a good distribution.
     */
    struct VoxelHash {
        size_t operator()(const Voxel& voxel) const {
            // Pack 16-bit representations of x, y, and z into a 64-bit integer.
            // This assumes coordinates will fit within a 16-bit range for non-overlapping packing.
            uint64_t packed = (static_cast<uint64_t>(static_cast<int16_t>(voxel.x))) |
                            (static_cast<uint64_t>(static_cast<int16_t>(voxel.y)) << 16) |
                            (static_cast<uint64_t>(static_cast<int16_t>(voxel.z)) << 32);
            
            // Apply a series of mixing operations to distribute the hash value well.
            // These constants are commonly used in high-quality hash functions.
            packed ^= packed >> 33;
            packed *= 0xFF51AFD7ED558CCDULL; // Fibonacci hashing multiplier
            packed ^= packed >> 33;
            packed *= 0xC4CEB9FE1A85EC53ULL; // Another multiplier
            packed ^= packed >> 33;
            return static_cast<size_t>(packed);
        }
    };

    /**
     * @brief Contains the data associated with a single voxel, primarily a collection of points.
     * @details This structure manages point density within a voxel and can support expiration policies.
     */
    struct VoxelBlock {
            /// @brief Constructs a VoxelBlock, reserving memory for a given point capacity.
            explicit VoxelBlock(int32_t capacity = 20) : capacity_(capacity) {
                points.reserve(capacity);
            }

            /// @brief Checks if the block has reached its point capacity.
            bool IsFull() const {
                return points.size() >= static_cast<size_t>(capacity_);
            }

            /// @brief Adds a point to the block if it is not full.
            bool AddPoint(const Eigen::Vector3d& point) {
                if (IsFull()) return false;
                points.push_back(point);
                return true;
            }

            /// @brief Returns the current number of points in the block.
            int NumPoints() const {
                return static_cast<int>(points.size());
            }

            /// @brief Returns the maximum capacity of the block.
            int32_t Capacity() const {
                return capacity_;
            }

            ArrayVector3d points;      ///< The list of 3D points contained within this voxel.
            int32_t life_time = 0;     ///< A counter for expiration logic (0 means it never expires).

        private:
            int32_t capacity_;         ///< The maximum number of points this block can hold.
    };

    /// @brief A type alias for a high-performance hash map that maps voxel indices to voxel content.
    using VoxelHashMap = tsl::robin_map<Voxel, VoxelBlock, VoxelHash>;

} //namespace stateestimate