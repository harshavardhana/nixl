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

    auto next = ranges_.lower_bound(range.descriptorBase);
    if (next != ranges_.end() && next->first == range.descriptorBase) {
        if (next->second.range.length == range.length &&
            next->second.range.memoryType == range.memoryType) {
            return RangeInsertResult::Duplicate;
        }
        return RangeInsertResult::Overlap;
    }

    if (next != ranges_.end() && next->first <= last_address) {
        return RangeInsertResult::Overlap;
    }
    if (next != ranges_.begin()) {
        const auto previous = std::prev(next);
        if (previous->second.lastAddress >= range.descriptorBase) {
            return RangeInsertResult::Overlap;
        }
    }
    return RangeInsertResult::Inserted;
}

RangeInsertResult
RegisteredMemoryRegistry::checkInsert(const RegisteredMemoryRange &range) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return checkInsertLocked(range);
}

RangeInsertResult
RegisteredMemoryRegistry::insert(const RegisteredMemoryRange &range, uint64_t owner_id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const RangeInsertResult result = checkInsertLocked(range);
    if (result == RangeInsertResult::Inserted) {
        uintptr_t last_address = 0;
        getLastAddress(range.descriptorBase, range.length, last_address);
        ranges_.emplace(range.descriptorBase,
                        Entry{range, last_address, std::unordered_set<uint64_t>{owner_id}});
        return result;
    }
    if (result == RangeInsertResult::Duplicate) {
        auto &owners = ranges_.at(range.descriptorBase).owners;
        if (!owners.insert(owner_id).second) {
            return RangeInsertResult::DuplicateOwner;
        }
    }
    return result;
}

RangeRemoveResult
RegisteredMemoryRegistry::remove(const RegisteredMemoryRange &range, uint64_t owner_id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto it = ranges_.find(range.descriptorBase);
    if (it == ranges_.end() || it->second.range.length != range.length ||
        it->second.range.memoryType != range.memoryType) {
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
RegisteredMemoryRegistry::resolve(uintptr_t request_addr, size_t request_len) const {
    if (request_len == 0) {
        return {RangeResolveStatus::ZeroLength};
    }

    uintptr_t request_last = 0;
    if (!getLastAddress(request_addr, request_len, request_last)) {
        return {RangeResolveStatus::AddressOverflow};
    }

    const std::lock_guard<std::mutex> lock(mutex_);
    auto next = ranges_.upper_bound(request_addr);
    if (next == ranges_.begin()) {
        return {RangeResolveStatus::NotFound};
    }

    auto containing_start = std::prev(next);
    if (request_addr > containing_start->second.lastAddress) {
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
    while (current != ranges_.end() && covered_last != std::numeric_limits<uintptr_t>::max() &&
           current->first == covered_last + 1) {
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
                                                    size_t request_len) const {
    RegisteredMemoryFragments fragments;
    if (request_len == 0) {
        fragments.status = RangeResolveStatus::ZeroLength;
        return fragments;
    }
    if (request_len - 1 > std::numeric_limits<uintptr_t>::max() - request_addr) {
        fragments.status = RangeResolveStatus::AddressOverflow;
        return fragments;
    }

    const std::lock_guard<std::mutex> lock(mutex_);
    uintptr_t current = request_addr;
    size_t remaining = request_len;
    while (remaining != 0) {
        const RegisteredMemoryResolution resolution = registry_.resolve(current, 1);
        if (resolution.status != RangeResolveStatus::Resolved) {
            fragments.leases.clear();
            fragments.status = resolution.status;
            return fragments;
        }

        auto lifetime = lifetimes_.find(resolution.descriptorBase);
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
    const std::lock_guard<std::mutex> lock(mutex_);
    return ranges_.size();
}

RegisteredMemoryManager::RegisteredMemoryManager(size_t max_registration_size)
    : maxRegistrationSize_(max_registration_size) {}

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

    auto next = retiringRanges_.lower_bound(range.descriptorBase);
    if (next != retiringRanges_.end() && next->first <= range_last) {
        return true;
    }
    if (next != retiringRanges_.begin()) {
        const auto previous = std::prev(next);
        const uintptr_t previous_last =
            previous->second.descriptorBase + previous->second.length - 1;
        if (previous_last >= range.descriptorBase) {
            return true;
        }
    }
    return false;
}

bool
RegisteredMemoryManager::registerMemory(uintptr_t base,
                                        size_t length,
                                        RegisteredMemoryType memory_type,
                                        const AcquireDescriptor &acquire_descriptor,
                                        const ReleaseDescriptor &release_descriptor,
                                        LogicalMemoryRegistration &registration) {
    registration = {};
    const char *memory_type_name = memory_type == RegisteredMemoryType::Vram ? "VRAM" : "DRAM";
    NIXL_INFO << "registerMemory: request base=" << reinterpret_cast<void *>(base)
              << " length=" << length << " memory_type=" << memory_type_name
              << " max_chunk_size=" << maxRegistrationSize_;

    if (!acquire_descriptor || !release_descriptor) {
        NIXL_ERROR << "registerMemory: descriptor callback missing"
                   << " acquire=" << static_cast<bool>(acquire_descriptor)
                   << " release=" << static_cast<bool>(release_descriptor);
        return false;
    }

    std::vector<RegisteredMemoryRange> chunks;
    if (!makeChunks(base, length, memory_type, chunks)) {
        NIXL_ERROR << "registerMemory: cannot chunk invalid range"
                   << " base=" << reinterpret_cast<void *>(base) << " length=" << length
                   << " max_chunk_size=" << maxRegistrationSize_;
        return false;
    }

    NIXL_INFO << "registerMemory: generated " << chunks.size() << " chunk(s)";
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto &chunk = chunks[i];
        NIXL_INFO << "registerMemory: planned chunk=" << (i + 1) << "/" << chunks.size()
                  << " base=" << reinterpret_cast<void *>(chunk.descriptorBase)
                  << " last=" << reinterpret_cast<void *>(chunk.descriptorBase + chunk.length - 1)
                  << " length=" << chunk.length << " memory_type=" << memory_type_name;
    }

    const std::lock_guard<std::mutex> lock(mutex_);
    std::vector<bool> needs_descriptor;
    needs_descriptor.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto &chunk = chunks[i];
        if (overlapsRetiringRange(chunk)) {
            NIXL_ERROR << "registerMemory: planned chunk overlaps a retiring registration"
                       << " chunk=" << (i + 1) << "/" << chunks.size()
                       << " base=" << reinterpret_cast<void *>(chunk.descriptorBase)
                       << " length=" << chunk.length;
            return false;
        }
        const RangeInsertResult result = registry_.checkInsert(chunk);
        const char *registry_state = result == RangeInsertResult::Inserted ? "new" :
            result == RangeInsertResult::Duplicate                         ? "duplicate" :
                                                                             "rejected";
        NIXL_INFO << "registerMemory: checked chunk=" << (i + 1) << "/" << chunks.size()
                  << " base=" << reinterpret_cast<void *>(chunk.descriptorBase)
                  << " length=" << chunk.length << " registry_state=" << registry_state;
        if (result != RangeInsertResult::Inserted && result != RangeInsertResult::Duplicate) {
            NIXL_ERROR << "registerMemory: chunk rejected by registry"
                       << " chunk=" << (i + 1) << "/" << chunks.size()
                       << " base=" << reinterpret_cast<void *>(chunk.descriptorBase)
                       << " length=" << chunk.length
                       << " insert_result=" << static_cast<int>(result);
            return false;
        }
        needs_descriptor.push_back(result == RangeInsertResult::Inserted);
    }

    std::vector<uintptr_t> acquired;
    acquired.reserve(chunks.size());
    bool acquisition_succeeded = true;
    size_t failed_chunk_index = chunks.size();
    try {
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (!needs_descriptor[i]) {
                NIXL_INFO << "registerMemory: reusing descriptor"
                          << " chunk=" << (i + 1) << "/" << chunks.size()
                          << " base=" << reinterpret_cast<void *>(chunks[i].descriptorBase)
                          << " length=" << chunks[i].length;
                continue;
            }
            NIXL_INFO << "registerMemory: acquiring descriptor"
                      << " chunk=" << (i + 1) << "/" << chunks.size()
                      << " base=" << reinterpret_cast<void *>(chunks[i].descriptorBase)
                      << " length=" << chunks[i].length;
            if (!acquire_descriptor(chunks[i].descriptorBase, chunks[i].length)) {
                NIXL_ERROR << "registerMemory: descriptor acquisition failed"
                           << " chunk=" << (i + 1) << "/" << chunks.size()
                           << " base=" << reinterpret_cast<void *>(chunks[i].descriptorBase)
                           << " length=" << chunks[i].length;
                failed_chunk_index = i;
                acquisition_succeeded = false;
                break;
            }
            acquired.push_back(chunks[i].descriptorBase);
            NIXL_INFO << "registerMemory: descriptor acquired"
                      << " chunk=" << (i + 1) << "/" << chunks.size()
                      << " base=" << reinterpret_cast<void *>(chunks[i].descriptorBase)
                      << " length=" << chunks[i].length;
        }
    }
    catch (...) {
        NIXL_ERROR << "registerMemory: descriptor acquisition threw an exception";
        acquisition_succeeded = false;
    }
    if (!acquisition_succeeded) {
        NIXL_ERROR << "registerMemory: acquisition failure dump"
                   << " request_base=" << reinterpret_cast<void *>(base)
                   << " request_length=" << length << " memory_type=" << memory_type_name
                   << " chunks=" << chunks.size() << " failed_chunk="
                   << (failed_chunk_index < chunks.size() ? failed_chunk_index + 1 : 0);
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto &chunk = chunks[i];
            const bool was_acquired =
                std::find(acquired.begin(), acquired.end(), chunk.descriptorBase) != acquired.end();
            const char *descriptor_state = !needs_descriptor[i] ? "reused" :
                was_acquired                                    ? "acquired" :
                i == failed_chunk_index                         ? "failed" :
                                                                  "not-attempted";
            NIXL_ERROR << "registerMemory: failure chunk=" << (i + 1) << "/" << chunks.size()
                       << " base=" << reinterpret_cast<void *>(chunk.descriptorBase) << " last="
                       << reinterpret_cast<void *>(chunk.descriptorBase + chunk.length - 1)
                       << " length=" << chunk.length << " descriptor_state=" << descriptor_state;
        }
        NIXL_INFO << "registerMemory: rolling back " << acquired.size()
                  << " acquired descriptor(s)";
        for (auto it = acquired.rbegin(); it != acquired.rend(); ++it) {
            try {
                NIXL_INFO << "registerMemory: releasing descriptor during rollback"
                          << " base=" << reinterpret_cast<void *>(*it);
                release_descriptor(*it);
            }
            catch (...) {
            }
        }
        return false;
    }

    std::vector<std::shared_ptr<DescriptorLifetime>> new_lifetimes(chunks.size());
    try {
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (needs_descriptor[i]) {
                new_lifetimes[i] = std::make_shared<DescriptorLifetime>();
            }
        }
    }
    catch (...) {
        for (auto it = acquired.rbegin(); it != acquired.rend(); ++it) {
            try {
                release_descriptor(*it);
            }
            catch (...) {
            }
        }
        return false;
    }

    uint64_t owner_id = nextOwnerId_++;
    if (owner_id == 0) {
        owner_id = nextOwnerId_++;
    }

    std::vector<RegisteredMemoryRange> committed;
    committed.reserve(chunks.size());
    bool publication_succeeded = true;
    try {
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto &chunk = chunks[i];
            const RangeInsertResult result = registry_.insert(chunk, owner_id);
            if (result != RangeInsertResult::Inserted && result != RangeInsertResult::Duplicate) {
                NIXL_ERROR << "registerMemory: failed to publish chunk"
                           << " owner_id=" << owner_id << " chunk=" << (i + 1) << "/"
                           << chunks.size()
                           << " base=" << reinterpret_cast<void *>(chunk.descriptorBase)
                           << " length=" << chunk.length
                           << " insert_result=" << static_cast<int>(result);
                publication_succeeded = false;
                break;
            }
            committed.push_back(chunk);
            NIXL_INFO << "registerMemory: published chunk"
                      << " owner_id=" << owner_id << " chunk=" << (i + 1) << "/" << chunks.size()
                      << " base=" << reinterpret_cast<void *>(chunk.descriptorBase)
                      << " length=" << chunk.length << " registry_state="
                      << (result == RangeInsertResult::Inserted ? "new" : "duplicate");
            if (result == RangeInsertResult::Inserted) {
                lifetimes_.emplace(chunk.descriptorBase, std::move(new_lifetimes[i]));
            } else if (lifetimes_.find(chunk.descriptorBase) == lifetimes_.end()) {
                NIXL_ERROR << "registerMemory: duplicate chunk has no descriptor lifetime"
                           << " owner_id=" << owner_id
                           << " base=" << reinterpret_cast<void *>(chunk.descriptorBase);
                publication_succeeded = false;
                break;
            }
        }
    }
    catch (...) {
        NIXL_ERROR << "registerMemory: publication threw an exception";
        publication_succeeded = false;
    }
    if (!publication_succeeded) {
        NIXL_INFO << "registerMemory: rolling back publication"
                  << " owner_id=" << owner_id << " committed_chunks=" << committed.size()
                  << " acquired_descriptors=" << acquired.size();
        for (auto it = committed.rbegin(); it != committed.rend(); ++it) {
            if (registry_.remove(*it, owner_id) == RangeRemoveResult::Removed) {
                lifetimes_.erase(it->descriptorBase);
            }
        }
        for (auto it = acquired.rbegin(); it != acquired.rend(); ++it) {
            try {
                release_descriptor(*it);
            }
            catch (...) {
            }
        }
        return false;
    }

    registration.ownerId = owner_id;
    registration.chunks = std::move(chunks);
    NIXL_INFO << "registerMemory: registration complete"
              << " owner_id=" << registration.ownerId << " base=" << reinterpret_cast<void *>(base)
              << " length=" << length << " chunks=" << registration.chunks.size()
              << " newly_acquired_descriptors=" << acquired.size();
    return true;
}

bool
RegisteredMemoryManager::deregisterMemory(LogicalMemoryRegistration &registration,
                                          const ReleaseDescriptor &release_descriptor) {
    if (!registration.valid() || !release_descriptor) {
        return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);
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
            auto lifetime = lifetimes_.find(chunk.descriptorBase);
            if (lifetime == lifetimes_.end()) {
                success = false;
                continue;
            }
            {
                const std::lock_guard<std::mutex> lifetime_lock(lifetime->second->mutex);
                lifetime->second->retiring = true;
            }
            retiringRanges_.emplace(chunk.descriptorBase, chunk);
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
            if (!release_descriptor(pending.range.descriptorBase)) {
                success = false;
            }
        }
        catch (...) {
            success = false;
        }
    }

    lock.lock();
    for (const auto &pending : descriptors_to_release) {
        auto lifetime = lifetimes_.find(pending.range.descriptorBase);
        if (lifetime != lifetimes_.end() && lifetime->second == pending.lifetime) {
            lifetimes_.erase(lifetime);
        }
        retiringRanges_.erase(pending.range.descriptorBase);
    }
    return success;
}

RegisteredMemoryResolution
RegisteredMemoryManager::resolve(uintptr_t request_addr, size_t request_len) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return registry_.resolve(request_addr, request_len);
}

RegisteredMemoryLease
RegisteredMemoryManager::resolveAndAcquire(uintptr_t request_addr, size_t request_len) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const RegisteredMemoryResolution resolution = registry_.resolve(request_addr, request_len);
    if (resolution.status != RangeResolveStatus::Resolved) {
        return RegisteredMemoryLease(resolution, {});
    }

    auto lifetime = lifetimes_.find(resolution.descriptorBase);
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
    const std::lock_guard<std::mutex> lock(mutex_);
    return registry_.size();
}

} // namespace nixl_obj_rdma
