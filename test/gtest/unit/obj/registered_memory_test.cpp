/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/registered_memory.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace gtest::obj {
namespace {

using nixl_obj_rdma::LogicalMemoryRegistration;
using nixl_obj_rdma::RangeInsertResult;
using nixl_obj_rdma::RangeRemoveResult;
using nixl_obj_rdma::RangeResolveStatus;
using nixl_obj_rdma::RegisteredMemoryManager;
using nixl_obj_rdma::RegisteredMemoryRange;
using nixl_obj_rdma::RegisteredMemoryRegistry;
using nixl_obj_rdma::RegisteredMemoryType;

constexpr uintptr_t kBase = 0x100000;

TEST(RegisteredMemoryRegistryTest, ResolvesBaseInteriorAndFinalByte) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024}, 1),
              RangeInsertResult::Inserted);

    const auto at_base = registry.resolve(kBase, 32, RegisteredMemoryType::Dram);
    EXPECT_EQ(at_base.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(at_base.descriptorBase, kBase);
    EXPECT_EQ(at_base.registrationOffset, 0u);

    const auto interior = registry.resolve(kBase + 127, 64, RegisteredMemoryType::Dram);
    EXPECT_EQ(interior.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(interior.descriptorBase, kBase);
    EXPECT_EQ(interior.registrationOffset, 127u);
    EXPECT_EQ(interior.registeredLength, 1024u);

    const auto final_byte = registry.resolve(kBase + 1023, 1, RegisteredMemoryType::Dram);
    EXPECT_EQ(final_byte.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(final_byte.registrationOffset, 1023u);
}

TEST(RegisteredMemoryRegistryTest, RejectsUnregisteredAndPartiallyContainedRequests) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024}, 1),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.resolve(kBase - 1, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase + 1024, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase - 1, 2, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase + 1023, 2, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
}

TEST(RegisteredMemoryRegistryTest, ReportsRequestAcrossAdjacentRanges) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024}, 1),
              RangeInsertResult::Inserted);
    ASSERT_EQ(registry.insert({kBase + 1024, 512}, 1),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.resolve(kBase + 1023, 2, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::CrossRegistration);
    const auto second = registry.resolve(kBase + 1024, 1, RegisteredMemoryType::Dram);
    EXPECT_EQ(second.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(second.descriptorBase, kBase + 1024);
}

TEST(RegisteredMemoryRegistryTest, DistinguishesGapFromCrossRegistration) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024}, 1),
              RangeInsertResult::Inserted);
    ASSERT_EQ(registry.insert({kBase + 1025, 512}, 1),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.resolve(kBase + 1023, 3, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
}

TEST(RegisteredMemoryRegistryTest, RejectsZeroLengthAndAddressOverflow) {
    RegisteredMemoryRegistry registry;
    EXPECT_EQ(registry.insert({kBase, 0}, 1),
              RangeInsertResult::InvalidRange);
    EXPECT_EQ(registry.resolve(kBase, 0, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::ZeroLength);

    const uintptr_t max_address = std::numeric_limits<uintptr_t>::max();
    EXPECT_EQ(registry.insert({max_address, 2}, 1),
              RangeInsertResult::InvalidRange);
    EXPECT_EQ(registry.resolve(max_address, 2, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::AddressOverflow);

    ASSERT_EQ(registry.insert({max_address, 1}, 2),
              RangeInsertResult::Inserted);
    EXPECT_EQ(registry.resolve(max_address, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::Resolved);
}

TEST(RegisteredMemoryRegistryTest, ReferenceCountsDuplicatesAndRejectsOverlaps) {
    RegisteredMemoryRegistry registry;
    const RegisteredMemoryRange range{kBase, 1024};
    ASSERT_EQ(registry.insert(range, 11), RangeInsertResult::Inserted);
    EXPECT_EQ(registry.insert(range, 12), RangeInsertResult::Duplicate);
    EXPECT_EQ(registry.resolve(kBase, 1, RegisteredMemoryType::Dram).ownerCount, 2u);

    EXPECT_EQ(registry.insert({kBase + 1, 1024}, 13),
              RangeInsertResult::Overlap);
    EXPECT_EQ(registry.insert({kBase - 1, 2}, 13),
              RangeInsertResult::Overlap);
    EXPECT_EQ(registry.insert({kBase + 1024, 1}, 13),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.remove(range, 11), RangeRemoveResult::Retained);
    EXPECT_EQ(registry.resolve(kBase, 1, RegisteredMemoryType::Dram).ownerCount, 1u);
    EXPECT_EQ(registry.remove(range, 12), RangeRemoveResult::Removed);
    EXPECT_EQ(registry.resolve(kBase, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
}

TEST(RegisteredMemoryRegistryTest, TracksDramAndVramAsIndependentAddressSpaces) {
    RegisteredMemoryRegistry registry;
    const RegisteredMemoryRange dram{kBase, 1024, RegisteredMemoryType::Dram};
    const RegisteredMemoryRange vram{kBase, 1024, RegisteredMemoryType::Vram};

    ASSERT_EQ(registry.insert(dram, 1), RangeInsertResult::Inserted);
    ASSERT_EQ(registry.insert(vram, 2), RangeInsertResult::Inserted);
    EXPECT_EQ(registry.size(), 2u);

    const auto dram_resolution =
        registry.resolve(kBase + 16, 32, RegisteredMemoryType::Dram);
    const auto vram_resolution =
        registry.resolve(kBase + 16, 32, RegisteredMemoryType::Vram);
    ASSERT_EQ(dram_resolution.status, RangeResolveStatus::Resolved);
    ASSERT_EQ(vram_resolution.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(dram_resolution.memoryType, RegisteredMemoryType::Dram);
    EXPECT_EQ(vram_resolution.memoryType, RegisteredMemoryType::Vram);

    EXPECT_EQ(registry.remove(dram, 1), RangeRemoveResult::Removed);
    EXPECT_EQ(registry.resolve(kBase, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase, 1, RegisteredMemoryType::Vram).status,
              RangeResolveStatus::Resolved);
}

TEST(RegisteredMemoryRegistryTest, DoesNotResolveOrJoinRangesAcrossMemoryTypes) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::Inserted);
    ASSERT_EQ(registry.insert({kBase + 1024, 1024, RegisteredMemoryType::Vram}, 2),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.resolve(kBase, 1, RegisteredMemoryType::Vram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase + 1023, 2, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase + 1023, 2, RegisteredMemoryType::Vram).status,
              RangeResolveStatus::NotFound);
}

class DescriptorMock {
public:
    bool
    acquire(uintptr_t base, size_t length) {
        const std::lock_guard<std::mutex> lock(mutex);
        acquisitions.emplace_back(base, length);
        return failAcquisition == 0 || acquisitions.size() != failAcquisition;
    }

    bool
    release(uintptr_t base) {
        const std::lock_guard<std::mutex> lock(mutex);
        releases.push_back(base);
        if (releaseObserver) {
            releaseObserver(base);
        }
        return failRelease == 0 || releases.size() != failRelease;
    }

    RegisteredMemoryManager::AcquireDescriptor
    acquireCallback() {
        return [this](uintptr_t base, size_t length) { return acquire(base, length); };
    }

    RegisteredMemoryManager::ReleaseDescriptor
    releaseCallback() {
        return [this](uintptr_t base) { return release(base); };
    }

    size_t failAcquisition = 0;
    size_t failRelease = 0;
    std::function<void(uintptr_t)> releaseObserver;
    std::vector<std::pair<uintptr_t, size_t>> acquisitions;
    std::vector<uintptr_t> releases;
    std::mutex mutex;
};

constexpr size_t kCuObjMaxMemoryRegistrationSize = 4ULL * 1024 * 1024 * 1024;

TEST(RegisteredMemoryManagerTest, KeepsRegistrationsAtOrBelowLimitInOneChunk) {
    for (const size_t length :
         {kCuObjMaxMemoryRegistrationSize - 1, kCuObjMaxMemoryRegistrationSize}) {
        DescriptorMock mock;
        RegisteredMemoryManager manager(kCuObjMaxMemoryRegistrationSize,
                                        mock.acquireCallback(),
                                        mock.releaseCallback());
        LogicalMemoryRegistration registration;

        ASSERT_TRUE(manager.registerMemory(kBase,
                                           length,
                                           RegisteredMemoryType::Dram, registration));
        ASSERT_EQ(registration.chunks.size(), 1u);
        EXPECT_EQ(registration.chunks[0].length, length);
        ASSERT_EQ(mock.acquisitions.size(), 1u);
        EXPECT_LE(mock.acquisitions[0].second, kCuObjMaxMemoryRegistrationSize);
    }
}

TEST(RegisteredMemoryManagerTest, SplitsMaximumPlusOneAndMultipleChunks) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(kCuObjMaxMemoryRegistrationSize,
                                    mock.acquireCallback(),
                                    mock.releaseCallback());
    LogicalMemoryRegistration registration;
    const size_t length = 2 * kCuObjMaxMemoryRegistrationSize + 17;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       length,
                                       RegisteredMemoryType::Dram, registration));
    ASSERT_EQ(registration.chunks.size(), 3u);
    EXPECT_EQ(registration.chunks[0].descriptorBase, kBase);
    EXPECT_EQ(registration.chunks[0].length, kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[1].descriptorBase,
              kBase + kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[1].length, kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[2].descriptorBase,
              kBase + 2 * kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[2].length, 17u);
    for (const auto &acquisition : mock.acquisitions) {
        EXPECT_LE(acquisition.second, kCuObjMaxMemoryRegistrationSize);
    }
}

TEST(RegisteredMemoryManagerTest, MaximumPlusOneMakesTwoChunks) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(kCuObjMaxMemoryRegistrationSize,
                                    mock.acquireCallback(),
                                    mock.releaseCallback());
    LogicalMemoryRegistration registration;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       kCuObjMaxMemoryRegistrationSize + 1,
                                       RegisteredMemoryType::Dram, registration));
    ASSERT_EQ(registration.chunks.size(), 2u);
    EXPECT_EQ(registration.chunks[0].length, kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[1].length, 1u);
}

TEST(RegisteredMemoryManagerTest, RollsBackEveryAcquiredChunkOnFailure) {
    constexpr size_t kChunkSize = 1024;
    DescriptorMock mock;
    RegisteredMemoryManager manager(kChunkSize, mock.acquireCallback(), mock.releaseCallback());
    mock.failAcquisition = 3;
    LogicalMemoryRegistration registration;

    EXPECT_FALSE(manager.registerMemory(kBase,
                                        3 * kChunkSize,
                                        RegisteredMemoryType::Dram, registration));
    EXPECT_FALSE(registration.valid());
    EXPECT_EQ(manager.rangeCount(), 0u);
    ASSERT_EQ(mock.releases.size(), 2u);
    EXPECT_EQ(mock.releases[0], kBase + kChunkSize);
    EXPECT_EQ(mock.releases[1], kBase);
    EXPECT_EQ(manager.resolve(kBase, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
}

TEST(RegisteredMemoryManagerTest, DeregistrationReleasesAllAndOnlyOwnedChunks) {
    constexpr size_t kChunkSize = 1024;
    DescriptorMock mock;
    RegisteredMemoryManager manager(kChunkSize, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration registration;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       2 * kChunkSize + 1,
                                       RegisteredMemoryType::Dram, registration));
    EXPECT_TRUE(manager.deregisterMemory(registration));
    EXPECT_FALSE(registration.valid());
    EXPECT_EQ(manager.rangeCount(), 0u);
    EXPECT_EQ(mock.releases,
              (std::vector<uintptr_t>{kBase, kBase + kChunkSize, kBase + 2 * kChunkSize}));
    EXPECT_FALSE(manager.deregisterMemory(registration));
    EXPECT_EQ(mock.releases.size(), 3u);
}

TEST(RegisteredMemoryManagerTest, ExactDuplicatesShareDescriptorsUntilLastOwnerLeaves) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(1024, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration first;
    LogicalMemoryRegistration second;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram, first));
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram, second));
    EXPECT_EQ(mock.acquisitions.size(), 1u);
    EXPECT_EQ(manager.resolve(kBase, 1, RegisteredMemoryType::Dram).ownerCount, 2u);

    EXPECT_TRUE(manager.deregisterMemory(first));
    EXPECT_TRUE(mock.releases.empty());
    EXPECT_EQ(manager.resolve(kBase, 1, RegisteredMemoryType::Dram).ownerCount, 1u);
    EXPECT_TRUE(manager.deregisterMemory(second));
    EXPECT_EQ(mock.releases, (std::vector<uintptr_t>{kBase}));
}

TEST(RegisteredMemoryManagerTest, ReleaseFailureIsReportedWithoutRetry) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(1024, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration registration;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       2048,
                                       RegisteredMemoryType::Dram, registration));
    mock.failRelease = 1;

    EXPECT_FALSE(manager.deregisterMemory(registration));
    EXPECT_FALSE(registration.valid());
    EXPECT_EQ(manager.rangeCount(), 0u);
    EXPECT_EQ(mock.releases.size(), 2u);
}

TEST(RegisteredMemoryManagerTest, DeregistrationWaitsForTransferLeaseWithoutHoldingRegistryLock) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(1024, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration registration;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram, registration));

    auto lease =
        manager.resolveAndAcquire(kBase + 64, 128, RegisteredMemoryType::Dram);
    ASSERT_TRUE(lease.valid());
    EXPECT_EQ(lease.resolution().descriptorBase, kBase);
    EXPECT_EQ(lease.resolution().registrationOffset, 64u);

    std::atomic<bool> release_called{false};
    mock.releaseObserver = [&](uintptr_t) { release_called.store(true); };
    auto deregistration =
        std::async(std::launch::async, [&]() { return manager.deregisterMemory(registration); });

    for (int attempt = 0; attempt < 100 &&
                          manager.resolve(kBase, 1, RegisteredMemoryType::Dram).status !=
                              RangeResolveStatus::NotFound;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(manager.resolve(kBase, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(deregistration.wait_for(std::chrono::milliseconds(10)),
              std::future_status::timeout);
    EXPECT_FALSE(release_called.load());

    // A non-overlapping registration can proceed while deregistration waits,
    // proving that no registry lock is held across the in-flight lifetime.
    LogicalMemoryRegistration adjacent;
    EXPECT_TRUE(manager.registerMemory(kBase + 2048,
                                       64,
                                       RegisteredMemoryType::Dram, adjacent));

    lease.reset();
    EXPECT_TRUE(deregistration.get());
    EXPECT_TRUE(release_called.load());
    EXPECT_TRUE(manager.deregisterMemory(adjacent));
}

TEST(RegisteredMemoryManagerTest, ConcurrentLookupDuplicateAndAdjacentRegistrationAreSafe) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(1024, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration persistent;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram, persistent));

    std::atomic<bool> failed{false};
    std::thread lookup_thread([&]() {
        for (int i = 0; i < 500; ++i) {
            auto lease = manager.resolveAndAcquire(
                kBase + (i % 512), 1, RegisteredMemoryType::Dram);
            if (!lease.valid() || lease.resolution().descriptorBase != kBase) {
                failed.store(true);
            }
        }
    });
    std::thread duplicate_thread([&]() {
        for (int i = 0; i < 100; ++i) {
            LogicalMemoryRegistration duplicate;
            if (!manager.registerMemory(kBase,
                                        1024,
                                        RegisteredMemoryType::Dram, duplicate) ||
                !manager.deregisterMemory(duplicate)) {
                failed.store(true);
            }
        }
    });
    std::thread adjacent_thread([&]() {
        for (int i = 0; i < 100; ++i) {
            LogicalMemoryRegistration adjacent;
            if (!manager.registerMemory(kBase + 1024,
                                        1024,
                                        RegisteredMemoryType::Dram, adjacent) ||
                !manager.deregisterMemory(adjacent)) {
                failed.store(true);
            }
        }
    });

    lookup_thread.join();
    duplicate_thread.join();
    adjacent_thread.join();
    EXPECT_FALSE(failed.load());
    EXPECT_TRUE(manager.deregisterMemory(persistent));
    EXPECT_EQ(manager.rangeCount(), 0u);
}

TEST(RegisteredMemoryManagerTest, ResolvesAndPinsOrderedCrossDescriptorFragments) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(1024, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration registration;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       3072,
                                       RegisteredMemoryType::Dram, registration));

    auto fragments = manager.resolveAndAcquireFragments(
        kBase + 100, 2500, RegisteredMemoryType::Dram);
    ASSERT_TRUE(fragments.valid());
    ASSERT_EQ(fragments.leases.size(), 3u);
    EXPECT_EQ(fragments.leases[0].resolution().descriptorBase, kBase);
    EXPECT_EQ(fragments.leases[0].resolution().registrationOffset, 100u);
    EXPECT_EQ(fragments.leases[1].resolution().descriptorBase, kBase + 1024);
    EXPECT_EQ(fragments.leases[1].resolution().registrationOffset, 0u);
    EXPECT_EQ(fragments.leases[2].resolution().descriptorBase, kBase + 2048);

    auto deregistration =
        std::async(std::launch::async, [&]() { return manager.deregisterMemory(registration); });
    EXPECT_EQ(deregistration.wait_for(std::chrono::milliseconds(10)),
              std::future_status::timeout);
    fragments.leases.clear();
    EXPECT_TRUE(deregistration.get());
    EXPECT_EQ(manager.rangeCount(), 0u);
}

TEST(RegisteredMemoryManagerTest, FragmentResolutionRejectsGapsWithoutRetainingLeases) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(1024, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration first;
    LogicalMemoryRegistration second;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram, first));
    ASSERT_TRUE(manager.registerMemory(kBase + 2048,
                                       1024,
                                       RegisteredMemoryType::Dram, second));

    auto fragments = manager.resolveAndAcquireFragments(
        kBase + 512, 2048, RegisteredMemoryType::Dram);
    EXPECT_FALSE(fragments.valid());
    EXPECT_EQ(fragments.status, RangeResolveStatus::NotFound);
    EXPECT_TRUE(fragments.leases.empty());
    EXPECT_TRUE(manager.deregisterMemory(first));
    EXPECT_TRUE(manager.deregisterMemory(second));
}

TEST(RegisteredMemoryManagerTest, SameNumericDramAndVramRangesHaveIndependentLifetimes) {
    DescriptorMock mock;
    RegisteredMemoryManager manager(1024, mock.acquireCallback(), mock.releaseCallback());
    LogicalMemoryRegistration dram;
    LogicalMemoryRegistration vram;

    ASSERT_TRUE(
        manager.registerMemory(kBase, 1024, RegisteredMemoryType::Dram, dram));
    ASSERT_TRUE(
        manager.registerMemory(kBase, 1024, RegisteredMemoryType::Vram, vram));
    ASSERT_EQ(mock.acquisitions.size(), 2u);
    EXPECT_EQ(manager.rangeCount(), 2u);

    auto dram_lease = manager.resolveAndAcquire(kBase, 1, RegisteredMemoryType::Dram);
    auto vram_lease = manager.resolveAndAcquire(kBase, 1, RegisteredMemoryType::Vram);
    ASSERT_TRUE(dram_lease.valid());
    ASSERT_TRUE(vram_lease.valid());
    EXPECT_EQ(dram_lease.resolution().memoryType, RegisteredMemoryType::Dram);
    EXPECT_EQ(vram_lease.resolution().memoryType, RegisteredMemoryType::Vram);

    dram_lease.reset();
    ASSERT_TRUE(manager.deregisterMemory(dram));
    EXPECT_EQ(manager.resolve(kBase, 1, RegisteredMemoryType::Dram).status,
              RangeResolveStatus::NotFound);
    EXPECT_EQ(manager.resolve(kBase, 1, RegisteredMemoryType::Vram).status,
              RangeResolveStatus::Resolved);
    EXPECT_EQ(mock.releases.size(), 1u);

    vram_lease.reset();
    ASSERT_TRUE(manager.deregisterMemory(vram));
    EXPECT_EQ(mock.releases.size(), 2u);
    EXPECT_EQ(manager.rangeCount(), 0u);
}

} // namespace
} // namespace gtest::obj
