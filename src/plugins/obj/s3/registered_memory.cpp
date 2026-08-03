/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "registered_memory.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "common/nixl_log.h"

namespace nixl_obj_rdma {

bool
RegisteredMemoryRegistry::getLastAddress(uintptr_t base, size_t length, uintptr_t &last_address) {
    if (length == 0 || length - 1 > std::numeric_limits<uintptr_t>::max() - base) {
        return false;
    }
    last_address = base + length - 1;
    return true;
}

RangeInsertResult
RegisteredMemoryRegistry::checkInsertLocked(const RegisteredMemoryRange &range) const {
    uintptr_t last_address = 0;
    if (!getLastAddress(range.descriptorBase, range.length, last_address)) {
        return RangeInsertResult::InvalidRange;
    }

    const RegisteredMemoryKey key{range.memoryType, range.descriptorBase};
    auto next = ranges_.lower_bound(key);
    if (next != ranges_.end() && next->first == key) {
        if (next->second.range.length == range.length) {
            return RangeInsertResult::Duplicate;
        }
        return RangeInsertResult::Overlap;
    }

    if (next != ranges_.end() && next->first.first == range.memoryType &&
        next->first.second <= last_address) {
        return RangeInsertResult::Overlap;
    }
    if (next != ranges_.begin()) {
        const auto previous = std::prev(next);
        if (previous->first.first == range.memoryType &&
            previous->second.lastAddress >= range.descriptorBase) {
            return RangeInsertResult::Overlap;
        }
    }
    return RangeInsertResult::Inserted;
}

RangeInsertResult
RegisteredMemoryRegistry::checkInsert(const RegisteredMemoryRange &range) const {
    const std::shared_lock<std::shared_mutex> lock(mutex_);
    return checkInsertLocked(range);
}

RangeInsertResult
RegisteredMemoryRegistry::insert(const RegisteredMemoryRange &range, uint64_t owner_id) {
    const std::unique_lock<std::shared_mutex> lock(mutex_);
    const RangeInsertResult result = checkInsertLocked(range);
    if (result == RangeInsertResult::Inserted) {
        uintptr_t last_address = 0;
        getLastAddress(range.descriptorBase, range.length, last_address);
        ranges_.emplace(RegisteredMemoryKey{range.memoryType, range.descriptorBase},
                        Entry{range, last_address, std::unordered_set<uint64_t>{owner_id}});
        return result;
    }
    if (result == RangeInsertResult::Duplicate) {
        auto &owners = ranges_.at({range.memoryType, range.descriptorBase}).owners;
        if (!owners.insert(owner_id).second) {
            return RangeInsertResult::DuplicateOwner;
        }
    }
    return result;
}

RangeRemoveResult
RegisteredMemoryRegistry::remove(const RegisteredMemoryRange &range, uint64_t owner_id) {
    const std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = ranges_.find({range.memoryType, range.descriptorBase});
    if (it == ranges_.end() || it->second.range.length != range.length) {
        return RangeRemoveResult::NotFound;
    }
    if (it->second.owners.erase(owner_id) == 0) {
        return RangeRemoveResult::OwnerNotFound;
    }
    if (!it->second.owners.empty()) {
        return RangeRemoveResult::Retained;
    }
    ranges_.erase(it);
    return RangeRemoveResult::Removed;
}

RegisteredMemoryResolution
RegisteredMemoryRegistry::resolve(uintptr_t request_addr,
                                  size_t request_len,
                                  RegisteredMemoryType memory_type) const {
    if (request_len == 0) {
        return {RangeResolveStatus::ZeroLength};
    }

    uintptr_t request_last = 0;
    if (!getLastAddress(request_addr, request_len, request_last)) {
        return {RangeResolveStatus::AddressOverflow};
    }

    const std::shared_lock<std::shared_mutex> lock(mutex_);
    auto next = ranges_.upper_bound({memory_type, request_addr});
    if (next == ranges_.begin()) {
        return {RangeResolveStatus::NotFound};
    }

    auto containing_start = std::prev(next);
    if (containing_start->first.first != memory_type ||
        request_addr > containing_start->second.lastAddress) {
        return {RangeResolveStatus::NotFound};
    }

    const Entry &entry = containing_start->second;
    if (request_last <= entry.lastAddress) {
        return {RangeResolveStatus::Resolved,
                entry.range.descriptorBase,
                static_cast<size_t>(request_addr - entry.range.descriptorBase),
                entry.range.length,
                entry.range.memoryType,
                entry.owners.size()};
    }
    // Distinguish a request fully covered by adjacent registrations from a
    // partial/unregistered request. Single-descriptor callers receive a stable
    // cross-registration result; aggregate callers resolve each chunk in order.
    // partial/unregistered request. Phase 3 rejects both, but callers need a
    // stable cross-registration result for the former.
    auto current = std::next(containing_start);
    uintptr_t covered_last = entry.lastAddress;
    while (current != ranges_.end() && current->first.first == memory_type &&
           covered_last != std::numeric_limits<uintptr_t>::max() &&
           current->first.second == covered_last + 1) {
        covered_last = current->second.lastAddress;
        if (request_last <= covered_last) {
            return {RangeResolveStatus::CrossRegistration};
        }
        ++current;
    }
    return {RangeResolveStatus::NotFound};
}

RegisteredMemoryFragments
RegisteredMemoryManager::resolveAndAcquireFragments(uintptr_t request_addr,
                                                    size_t request_len,
                                                    RegisteredMemoryType memory_type) const {
    RegisteredMemoryFragments fragments;
    if (request_len == 0) {
        fragments.status = RangeResolveStatus::ZeroLength;
        return fragments;
    }
    if (request_len - 1 > std::numeric_limits<uintptr_t>::max() - request_addr) {
        fragments.status = RangeResolveStatus::AddressOverflow;
        return fragments;
    }

    const std::shared_lock<std::shared_mutex> lock(mutex_);
    uintptr_t current = request_addr;
    size_t remaining = request_len;
    while (remaining != 0) {
        const RegisteredMemoryResolution resolution = registry_.resolve(current, 1, memory_type);
        if (resolution.status != RangeResolveStatus::Resolved) {
            fragments.leases.clear();
            fragments.status = resolution.status;
            return fragments;
        }

        auto lifetime = lifetimes_.find({memory_type, resolution.descriptorBase});
        if (lifetime == lifetimes_.end()) {
            fragments.leases.clear();
            fragments.status = RangeResolveStatus::NotFound;
            return fragments;
        }

        const size_t available = resolution.registeredLength - resolution.registrationOffset;
        const size_t fragment_length = std::min(remaining, available);
        std::shared_ptr<LeaseToken> token;
        {
            const std::lock_guard<std::mutex> lifetime_lock(lifetime->second->mutex);
            if (lifetime->second->retiring) {
                fragments.leases.clear();
                fragments.status = RangeResolveStatus::NotFound;
                return fragments;
            }
            ++lifetime->second->activeLeases;
        }

        try {
            token = std::make_shared<LeaseToken>(lifetime->second);
        }
        catch (...) {
            const std::lock_guard<std::mutex> lifetime_lock(lifetime->second->mutex);
            --lifetime->second->activeLeases;
            if (lifetime->second->activeLeases == 0) {
                lifetime->second->condition.notify_all();
            }
            fragments.leases.clear();
            fragments.status = RangeResolveStatus::NotFound;
            return fragments;
        }

        try {
            RegisteredMemoryLease lease(resolution, token);
            fragments.leases.push_back(std::move(lease));
        }
        catch (...) {
            fragments.leases.clear();
            fragments.status = RangeResolveStatus::NotFound;
            return fragments;
        }

        remaining -= fragment_length;
        if (remaining != 0) {
            current += fragment_length;
        }
    }

    fragments.status = RangeResolveStatus::Resolved;
    return fragments;
}

size_t
RegisteredMemoryRegistry::size() const {
    const std::shared_lock<std::shared_mutex> lock(mutex_);
    return ranges_.size();
}

RegisteredMemoryManager::RegisteredMemoryManager(size_t max_registration_size,
                                                 AcquireDescriptor acquire_descriptor,
                                                 ReleaseDescriptor release_descriptor)
    : maxRegistrationSize_(max_registration_size),
      acquireDescriptor_(std::move(acquire_descriptor)),
      releaseDescriptor_(std::move(release_descriptor)) {}

RegisteredMemoryManager::LeaseToken::~LeaseToken() {
    const std::lock_guard<std::mutex> lock(lifetime->mutex);
    if (lifetime->activeLeases != 0) {
        --lifetime->activeLeases;
    }
    if (lifetime->activeLeases == 0) {
        lifetime->condition.notify_all();
    }
}

bool
RegisteredMemoryManager::isValidRange(uintptr_t base, size_t length) {
    return length != 0 && length - 1 <= std::numeric_limits<uintptr_t>::max() - base;
}

bool
RegisteredMemoryManager::makeChunks(uintptr_t base,
                                    size_t length,
                                    RegisteredMemoryType memory_type,
                                    std::vector<RegisteredMemoryRange> &chunks) const {
    chunks.clear();
    if (maxRegistrationSize_ == 0 || !isValidRange(base, length)) {
        return false;
    }

    size_t remaining = length;
    uintptr_t chunk_base = base;
    while (remaining != 0) {
        const size_t chunk_length = std::min(remaining, maxRegistrationSize_);
        chunks.push_back({chunk_base, chunk_length, memory_type});
        remaining -= chunk_length;
        if (remaining != 0) {
            chunk_base += chunk_length;
        }
    }
    return true;
}

bool
RegisteredMemoryManager::overlapsRetiringRange(const RegisteredMemoryRange &range) const {
    if (!isValidRange(range.descriptorBase, range.length)) {
        return true;
    }
    const uintptr_t range_last = range.descriptorBase + range.length - 1;

    const RegisteredMemoryKey key{range.memoryType, range.descriptorBase};
    auto next = retiringRanges_.lower_bound(key);
    if (next != retiringRanges_.end() && next->first.first == range.memoryType &&
        next->first.second <= range_last) {
        return true;
    }
    if (next != retiringRanges_.begin()) {
        const auto previous = std::prev(next);
        if (previous->first.first == range.memoryType) {
            const uintptr_t previous_last =
                previous->second.descriptorBase + previous->second.length - 1;
            if (previous_last >= range.descriptorBase) {
                return true;
            }
        }
    }
    return false;
}

bool
RegisteredMemoryManager::registerMemory(uintptr_t base,
                                        size_t length,
                                        RegisteredMemoryType memory_type,
                                        LogicalMemoryRegistration &registration) {
    registration = {};
    NIXL_INFO << "registerMemory: base=" << reinterpret_cast<void *>(base) << " length=" << length
              << " max_chunk_size=" << maxRegistrationSize_;

    if (!acquireDescriptor_ || !releaseDescriptor_) {
        NIXL_ERROR << "registerMemory: descriptor callback missing";
        return false;
    }

    std::vector<RegisteredMemoryRange> chunks;
    if (!makeChunks(base, length, memory_type, chunks)) {
        NIXL_ERROR << "registerMemory: cannot chunk range"
                   << " base=" << reinterpret_cast<void *>(base) << " length=" << length;
        return false;
    }

    const std::unique_lock<std::shared_mutex> lock(mutex_);
    uint64_t owner_id = nextOwnerId_++;
    if (owner_id == 0) {
        owner_id = nextOwnerId_++;
    }

    struct CompletedChunk {
        RegisteredMemoryRange range;
        bool acquiredDescriptor;
    };

    std::vector<CompletedChunk> completed;
    completed.reserve(chunks.size());

    auto rollback = [&]() {
        NIXL_INFO << "registerMemory: rolling back " << completed.size() << " chunk(s)";
        for (auto it = completed.rbegin(); it != completed.rend(); ++it) {
            if (registry_.remove(it->range, owner_id) == RangeRemoveResult::Removed) {
                lifetimes_.erase({it->range.memoryType, it->range.descriptorBase});
            }
            if (it->acquiredDescriptor) {
                try {
                    releaseDescriptor_(it->range.descriptorBase);
                }
                catch (...) {
                }
            }
        }
    };

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto &chunk = chunks[i];

        if (overlapsRetiringRange(chunk)) {
            NIXL_ERROR << "registerMemory: chunk " << (i + 1) << "/" << chunks.size()
                       << " overlaps a retiring registration";
            rollback();
            return false;
        }

        const RangeInsertResult check = registry_.checkInsert(chunk);
        if (check != RangeInsertResult::Inserted && check != RangeInsertResult::Duplicate) {
            NIXL_ERROR << "registerMemory: chunk " << (i + 1) << "/" << chunks.size()
                       << " rejected by registry";
            rollback();
            return false;
        }

        bool acquired = false;
        if (check == RangeInsertResult::Inserted) {
            if (!acquireDescriptor_(chunk.descriptorBase, chunk.length)) {
                NIXL_ERROR << "registerMemory: descriptor acquisition failed for chunk " << (i + 1)
                           << "/" << chunks.size();
                rollback();
                return false;
            }
            acquired = true;
        }

        const RangeInsertResult result = registry_.insert(chunk, owner_id);
        if (result == RangeInsertResult::Inserted) {
            try {
                lifetimes_.emplace(RegisteredMemoryKey{chunk.memoryType, chunk.descriptorBase},
                                   std::make_shared<DescriptorLifetime>());
            }
            catch (...) {
                registry_.remove(chunk, owner_id);
                try {
                    releaseDescriptor_(chunk.descriptorBase);
                }
                catch (...) {
                }
                rollback();
                return false;
            }
        } else if (result == RangeInsertResult::Duplicate) {
            if (lifetimes_.find({chunk.memoryType, chunk.descriptorBase}) == lifetimes_.end()) {
                NIXL_ERROR << "registerMemory: duplicate chunk has no lifetime";
                registry_.remove(chunk, owner_id);
                rollback();
                return false;
            }
        } else {
            if (acquired) {
                try {
                    releaseDescriptor_(chunk.descriptorBase);
                }
                catch (...) {
                }
            }
            rollback();
            return false;
        }

        completed.push_back({chunk, acquired});
    }

    registration.ownerId = owner_id;
    registration.chunks = std::move(chunks);
    NIXL_INFO << "registerMemory: complete"
              << " owner_id=" << owner_id << " chunks=" << registration.chunks.size();
    return true;
}

bool
RegisteredMemoryManager::deregisterMemory(LogicalMemoryRegistration &registration) {
    if (!registration.valid() || !releaseDescriptor_) {
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    bool success = true;

    struct PendingRelease {
        RegisteredMemoryRange range;
        std::shared_ptr<DescriptorLifetime> lifetime;
    };

    std::vector<PendingRelease> descriptors_to_release;
    descriptors_to_release.reserve(registration.chunks.size());
    for (const auto &chunk : registration.chunks) {
        const RangeRemoveResult result = registry_.remove(chunk, registration.ownerId);
        if (result == RangeRemoveResult::Removed) {
            auto lifetime = lifetimes_.find({chunk.memoryType, chunk.descriptorBase});
            if (lifetime == lifetimes_.end()) {
                success = false;
                continue;
            }
            {
                const std::lock_guard<std::mutex> lifetime_lock(lifetime->second->mutex);
                lifetime->second->retiring = true;
            }
            retiringRanges_.emplace(
                RegisteredMemoryKey{chunk.memoryType, chunk.descriptorBase}, chunk);
            descriptors_to_release.push_back({chunk, lifetime->second});
        } else if (result != RangeRemoveResult::Retained) {
            success = false;
        }
    }

    registration = {};
    lock.unlock();

    // No registry/manager lock is held while waiting for in-flight transfers.
    for (const auto &pending : descriptors_to_release) {
        std::unique_lock<std::mutex> lifetime_lock(pending.lifetime->mutex);
        pending.lifetime->condition.wait(
            lifetime_lock, [&pending]() { return pending.lifetime->activeLeases == 0; });
    }

    for (const auto &pending : descriptors_to_release) {
        try {
            if (!releaseDescriptor_(pending.range.descriptorBase)) {
                success = false;
            }
        }
        catch (...) {
            success = false;
        }
    }

    lock.lock();
    for (const auto &pending : descriptors_to_release) {
        auto lifetime = lifetimes_.find(
            {pending.range.memoryType, pending.range.descriptorBase});
        if (lifetime != lifetimes_.end() && lifetime->second == pending.lifetime) {
            lifetimes_.erase(lifetime);
        }
        retiringRanges_.erase(
            {pending.range.memoryType, pending.range.descriptorBase});
    }
    return success;
}

RegisteredMemoryResolution
RegisteredMemoryManager::resolve(uintptr_t request_addr,
                                 size_t request_len,
                                 RegisteredMemoryType memory_type) const {
    const std::shared_lock<std::shared_mutex> lock(mutex_);
    return registry_.resolve(request_addr, request_len, memory_type);
}

RegisteredMemoryLease
RegisteredMemoryManager::resolveAndAcquire(uintptr_t request_addr,
                                           size_t request_len,
                                           RegisteredMemoryType memory_type) const {
    const std::shared_lock<std::shared_mutex> lock(mutex_);
    const RegisteredMemoryResolution resolution =
        registry_.resolve(request_addr, request_len, memory_type);
    if (resolution.status != RangeResolveStatus::Resolved) {
        return RegisteredMemoryLease(resolution, {});
    }

    auto lifetime = lifetimes_.find({memory_type, resolution.descriptorBase});
    if (lifetime == lifetimes_.end()) {
        return RegisteredMemoryLease({RangeResolveStatus::NotFound}, {});
    }

    {
        const std::lock_guard<std::mutex> lifetime_lock(lifetime->second->mutex);
        if (lifetime->second->retiring) {
            return RegisteredMemoryLease({RangeResolveStatus::NotFound}, {});
        }
        ++lifetime->second->activeLeases;
    }

    try {
        auto token = std::make_shared<LeaseToken>(lifetime->second);
        return RegisteredMemoryLease(resolution, std::move(token));
    }
    catch (...) {
        const std::lock_guard<std::mutex> lifetime_lock(lifetime->second->mutex);
        --lifetime->second->activeLeases;
        if (lifetime->second->activeLeases == 0) {
            lifetime->second->condition.notify_all();
        }
        return RegisteredMemoryLease({RangeResolveStatus::NotFound}, {});
    }
}

size_t
RegisteredMemoryManager::rangeCount() const {
    const std::shared_lock<std::shared_mutex> lock(mutex_);
    return registry_.size();
}

} // namespace nixl_obj_rdma
