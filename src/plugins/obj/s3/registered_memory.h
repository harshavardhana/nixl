/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_OBJ_S3_REGISTERED_MEMORY_H
#define NIXL_SRC_PLUGINS_OBJ_S3_REGISTERED_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace nixl_obj_rdma {

enum class RegisteredMemoryType {
    Dram,
    Vram,
};

struct RegisteredMemoryRange {
    uintptr_t descriptorBase = 0;
    size_t length = 0;
    RegisteredMemoryType memoryType = RegisteredMemoryType::Dram;
};

enum class RangeInsertResult {
    Inserted,
    Duplicate,
    InvalidRange,
    Overlap,
    DuplicateOwner,
};

enum class RangeRemoveResult {
    Removed,
    Retained,
    NotFound,
    OwnerNotFound,
};

enum class RangeResolveStatus {
    Resolved,
    NotFound,
    CrossRegistration,
    ZeroLength,
    AddressOverflow,
};

struct RegisteredMemoryResolution {
    RangeResolveStatus status = RangeResolveStatus::NotFound;
    uintptr_t descriptorBase = 0;
    size_t registrationOffset = 0;
    size_t registeredLength = 0;
    RegisteredMemoryType memoryType = RegisteredMemoryType::Dram;
    size_t ownerCount = 0;
};

/**
 * A resolved transfer range together with a pin on its cuObject descriptor.
 *
 * Copies share one pin. The descriptor remains registered until the final copy
 * is destroyed or reset, even if the logical NIXL registration is concurrently
 * deregistered.
 */
class RegisteredMemoryLease {
public:
    RegisteredMemoryLease() = default;

    const RegisteredMemoryResolution &
    resolution() const {
        return resolution_;
    }

    bool
    valid() const {
        return resolution_.status == RangeResolveStatus::Resolved &&
               static_cast<bool>(guard_);
    }

    void
    reset() {
        guard_.reset();
        resolution_ = {};
    }

private:
    friend class RegisteredMemoryManager;

    RegisteredMemoryLease(RegisteredMemoryResolution resolution, std::shared_ptr<void> guard)
        : resolution_(resolution),
          guard_(std::move(guard)) {}

    RegisteredMemoryResolution resolution_;
    std::shared_ptr<void> guard_;
};

/**
 * Thread-safe registry of non-overlapping descriptor ranges.
 *
 * Exact duplicates with the same memory type are reference-counted by owner.
 * Exact duplicates with a different type and all partial overlaps are rejected.
 * Adjacent ranges are valid and remain distinct registrations.
 */
class RegisteredMemoryRegistry {
public:
    RangeInsertResult
    checkInsert(const RegisteredMemoryRange &range) const;

    RangeInsertResult
    insert(const RegisteredMemoryRange &range, uint64_t owner_id);

    RangeRemoveResult
    remove(const RegisteredMemoryRange &range, uint64_t owner_id);

    RegisteredMemoryResolution
    resolve(uintptr_t request_addr, size_t request_len) const;

    size_t
    size() const;

private:
    struct Entry {
        RegisteredMemoryRange range;
        uintptr_t lastAddress = 0;
        std::unordered_set<uint64_t> owners;
    };

    static bool
    getLastAddress(uintptr_t base, size_t length, uintptr_t &last_address);

    RangeInsertResult
    checkInsertLocked(const RegisteredMemoryRange &range) const;

    mutable std::mutex mutex_;
    std::map<uintptr_t, Entry> ranges_;
};

struct LogicalMemoryRegistration {
    uint64_t ownerId = 0;
    std::vector<RegisteredMemoryRange> chunks;

    bool
    valid() const {
        return ownerId != 0 && !chunks.empty();
    }
};

/**
 * Transactional logical-registration manager.
 *
 * Descriptor callbacks are deliberately injected so chunking and rollback can
 * be tested without cuObject or a multi-GiB allocation. The manager serializes
 * descriptor acquisition/release with range publication and lookup.
 */
class RegisteredMemoryManager {
public:
    using AcquireDescriptor =
        std::function<bool(uintptr_t descriptor_base, size_t registered_length)>;
    using ReleaseDescriptor = std::function<bool(uintptr_t descriptor_base)>;

    explicit RegisteredMemoryManager(size_t max_registration_size);

    bool
    registerMemory(uintptr_t base,
                   size_t length,
                   RegisteredMemoryType memory_type,
                   const AcquireDescriptor &acquire_descriptor,
                   const ReleaseDescriptor &release_descriptor,
                   LogicalMemoryRegistration &registration);

    /**
     * Removes every chunk from lookup before releasing descriptors.
     *
     * Each descriptor is released at most once. A release failure is reported
     * to the caller, but is not retried because cuObject does not define whether
     * a failed put consumed the descriptor. The registration handle is consumed
     * even on failure.
     */
    bool
    deregisterMemory(LogicalMemoryRegistration &registration,
                     const ReleaseDescriptor &release_descriptor);

    RegisteredMemoryResolution
    resolve(uintptr_t request_addr, size_t request_len) const;

    /**
     * Resolve and pin a range for transfer preparation.
     *
     * Deregistration removes the range from new lookups immediately, but
     * descriptor release waits for all returned leases to be destroyed.
     */
    RegisteredMemoryLease
    resolveAndAcquire(uintptr_t request_addr, size_t request_len) const;

    size_t
    rangeCount() const;

private:
    struct DescriptorLifetime {
        std::mutex mutex;
        std::condition_variable condition;
        size_t activeLeases = 0;
        bool retiring = false;
    };

    struct LeaseToken {
        explicit LeaseToken(std::shared_ptr<DescriptorLifetime> lifetime)
            : lifetime(std::move(lifetime)) {}
        ~LeaseToken();

        std::shared_ptr<DescriptorLifetime> lifetime;
    };

    static bool
    isValidRange(uintptr_t base, size_t length);

    bool
    makeChunks(uintptr_t base,
               size_t length,
               RegisteredMemoryType memory_type,
               std::vector<RegisteredMemoryRange> &chunks) const;

    bool
    overlapsRetiringRange(const RegisteredMemoryRange &range) const;

    const size_t maxRegistrationSize_;
    mutable std::mutex mutex_;
    RegisteredMemoryRegistry registry_;
    std::map<uintptr_t, std::shared_ptr<DescriptorLifetime>> lifetimes_;
    std::map<uintptr_t, RegisteredMemoryRange> retiringRanges_;
    uint64_t nextOwnerId_ = 1;
};

} // namespace nixl_obj_rdma

#endif // NIXL_SRC_PLUGINS_OBJ_S3_REGISTERED_MEMORY_H
