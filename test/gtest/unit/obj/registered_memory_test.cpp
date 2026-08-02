/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/registered_memory.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
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
    ASSERT_EQ(registry.insert({kBase, 1024, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::Inserted);

    const auto at_base = registry.resolve(kBase, 32);
    EXPECT_EQ(at_base.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(at_base.descriptorBase, kBase);
    EXPECT_EQ(at_base.registrationOffset, 0u);

    const auto interior = registry.resolve(kBase + 127, 64);
    EXPECT_EQ(interior.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(interior.descriptorBase, kBase);
    EXPECT_EQ(interior.registrationOffset, 127u);
    EXPECT_EQ(interior.registeredLength, 1024u);
    EXPECT_EQ(interior.memoryType, RegisteredMemoryType::Dram);

    const auto final_byte = registry.resolve(kBase + 1023, 1);
    EXPECT_EQ(final_byte.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(final_byte.registrationOffset, 1023u);
}

TEST(RegisteredMemoryRegistryTest, RejectsUnregisteredAndPartiallyContainedRequests) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.resolve(kBase - 1, 1).status, RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase + 1024, 1).status, RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase - 1, 2).status, RangeResolveStatus::NotFound);
    EXPECT_EQ(registry.resolve(kBase + 1023, 2).status, RangeResolveStatus::NotFound);
}

TEST(RegisteredMemoryRegistryTest, ReportsRequestAcrossAdjacentRanges) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::Inserted);
    ASSERT_EQ(registry.insert({kBase + 1024, 512, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.resolve(kBase + 1023, 2).status,
              RangeResolveStatus::CrossRegistration);
    const auto second = registry.resolve(kBase + 1024, 1);
    EXPECT_EQ(second.status, RangeResolveStatus::Resolved);
    EXPECT_EQ(second.descriptorBase, kBase + 1024);
}

TEST(RegisteredMemoryRegistryTest, DistinguishesGapFromCrossRegistration) {
    RegisteredMemoryRegistry registry;
    ASSERT_EQ(registry.insert({kBase, 1024, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::Inserted);
    ASSERT_EQ(registry.insert({kBase + 1025, 512, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.resolve(kBase + 1023, 3).status, RangeResolveStatus::NotFound);
}

TEST(RegisteredMemoryRegistryTest, RejectsZeroLengthAndAddressOverflow) {
    RegisteredMemoryRegistry registry;
    EXPECT_EQ(registry.insert({kBase, 0, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::InvalidRange);
    EXPECT_EQ(registry.resolve(kBase, 0).status, RangeResolveStatus::ZeroLength);

    const uintptr_t max_address = std::numeric_limits<uintptr_t>::max();
    EXPECT_EQ(registry.insert({max_address, 2, RegisteredMemoryType::Dram}, 1),
              RangeInsertResult::InvalidRange);
    EXPECT_EQ(registry.resolve(max_address, 2).status, RangeResolveStatus::AddressOverflow);

    ASSERT_EQ(registry.insert({max_address, 1, RegisteredMemoryType::Dram}, 2),
              RangeInsertResult::Inserted);
    EXPECT_EQ(registry.resolve(max_address, 1).status, RangeResolveStatus::Resolved);
}

TEST(RegisteredMemoryRegistryTest, ReferenceCountsDuplicatesAndRejectsOverlaps) {
    RegisteredMemoryRegistry registry;
    const RegisteredMemoryRange range{kBase, 1024, RegisteredMemoryType::Dram};
    ASSERT_EQ(registry.insert(range, 11), RangeInsertResult::Inserted);
    EXPECT_EQ(registry.insert(range, 12), RangeInsertResult::Duplicate);
    EXPECT_EQ(registry.resolve(kBase, 1).ownerCount, 2u);

    EXPECT_EQ(registry.insert({kBase, 1024, RegisteredMemoryType::Vram}, 13),
              RangeInsertResult::Overlap);
    EXPECT_EQ(registry.insert({kBase + 1, 1024, RegisteredMemoryType::Dram}, 13),
              RangeInsertResult::Overlap);
    EXPECT_EQ(registry.insert({kBase - 1, 2, RegisteredMemoryType::Dram}, 13),
              RangeInsertResult::Overlap);
    EXPECT_EQ(registry.insert({kBase + 1024, 1, RegisteredMemoryType::Dram}, 13),
              RangeInsertResult::Inserted);

    EXPECT_EQ(registry.remove(range, 11), RangeRemoveResult::Retained);
    EXPECT_EQ(registry.resolve(kBase, 1).ownerCount, 1u);
    EXPECT_EQ(registry.remove(range, 12), RangeRemoveResult::Removed);
    EXPECT_EQ(registry.resolve(kBase, 1).status, RangeResolveStatus::NotFound);
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
    std::vector<std::pair<uintptr_t, size_t>> acquisitions;
    std::vector<uintptr_t> releases;
    std::mutex mutex;
};

constexpr size_t kCuObjMaxMemoryRegistrationSize = 4ULL * 1024 * 1024 * 1024;

TEST(RegisteredMemoryManagerTest, KeepsRegistrationsAtOrBelowLimitInOneChunk) {
    for (const size_t length :
         {kCuObjMaxMemoryRegistrationSize - 1, kCuObjMaxMemoryRegistrationSize}) {
        RegisteredMemoryManager manager(kCuObjMaxMemoryRegistrationSize);
        DescriptorMock mock;
        LogicalMemoryRegistration registration;

        ASSERT_TRUE(manager.registerMemory(kBase,
                                           length,
                                           RegisteredMemoryType::Dram,
                                           mock.acquireCallback(),
                                           mock.releaseCallback(),
                                           registration));
        ASSERT_EQ(registration.chunks.size(), 1u);
        EXPECT_EQ(registration.chunks[0].length, length);
        ASSERT_EQ(mock.acquisitions.size(), 1u);
        EXPECT_LE(mock.acquisitions[0].second, kCuObjMaxMemoryRegistrationSize);
    }
}

TEST(RegisteredMemoryManagerTest, SplitsMaximumPlusOneAndMultipleChunks) {
    RegisteredMemoryManager manager(kCuObjMaxMemoryRegistrationSize);
    DescriptorMock mock;
    LogicalMemoryRegistration registration;
    const size_t length = 2 * kCuObjMaxMemoryRegistrationSize + 17;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       length,
                                       RegisteredMemoryType::Vram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       registration));
    ASSERT_EQ(registration.chunks.size(), 3u);
    EXPECT_EQ(registration.chunks[0].descriptorBase, kBase);
    EXPECT_EQ(registration.chunks[0].length, kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[1].descriptorBase,
              kBase + kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[1].length, kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[2].descriptorBase,
              kBase + 2 * kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[2].length, 17u);
    EXPECT_EQ(registration.chunks[2].memoryType, RegisteredMemoryType::Vram);
    for (const auto &acquisition : mock.acquisitions) {
        EXPECT_LE(acquisition.second, kCuObjMaxMemoryRegistrationSize);
    }
}

TEST(RegisteredMemoryManagerTest, MaximumPlusOneMakesTwoChunks) {
    RegisteredMemoryManager manager(kCuObjMaxMemoryRegistrationSize);
    DescriptorMock mock;
    LogicalMemoryRegistration registration;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       kCuObjMaxMemoryRegistrationSize + 1,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       registration));
    ASSERT_EQ(registration.chunks.size(), 2u);
    EXPECT_EQ(registration.chunks[0].length, kCuObjMaxMemoryRegistrationSize);
    EXPECT_EQ(registration.chunks[1].length, 1u);
}

TEST(RegisteredMemoryManagerTest, RollsBackEveryAcquiredChunkOnFailure) {
    constexpr size_t kChunkSize = 1024;
    RegisteredMemoryManager manager(kChunkSize);
    DescriptorMock mock;
    mock.failAcquisition = 3;
    LogicalMemoryRegistration registration;

    EXPECT_FALSE(manager.registerMemory(kBase,
                                        3 * kChunkSize,
                                        RegisteredMemoryType::Dram,
                                        mock.acquireCallback(),
                                        mock.releaseCallback(),
                                        registration));
    EXPECT_FALSE(registration.valid());
    EXPECT_EQ(manager.rangeCount(), 0u);
    ASSERT_EQ(mock.releases.size(), 2u);
    EXPECT_EQ(mock.releases[0], kBase + kChunkSize);
    EXPECT_EQ(mock.releases[1], kBase);
    EXPECT_EQ(manager.resolve(kBase, 1).status, RangeResolveStatus::NotFound);
}

TEST(RegisteredMemoryManagerTest, DeregistrationReleasesAllAndOnlyOwnedChunks) {
    constexpr size_t kChunkSize = 1024;
    RegisteredMemoryManager manager(kChunkSize);
    DescriptorMock mock;
    LogicalMemoryRegistration registration;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       2 * kChunkSize + 1,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       registration));
    EXPECT_TRUE(manager.deregisterMemory(registration, mock.releaseCallback()));
    EXPECT_FALSE(registration.valid());
    EXPECT_EQ(manager.rangeCount(), 0u);
    EXPECT_EQ(mock.releases,
              (std::vector<uintptr_t>{kBase, kBase + kChunkSize, kBase + 2 * kChunkSize}));
    EXPECT_FALSE(manager.deregisterMemory(registration, mock.releaseCallback()));
    EXPECT_EQ(mock.releases.size(), 3u);
}

TEST(RegisteredMemoryManagerTest, ExactDuplicatesShareDescriptorsUntilLastOwnerLeaves) {
    RegisteredMemoryManager manager(1024);
    DescriptorMock mock;
    LogicalMemoryRegistration first;
    LogicalMemoryRegistration second;

    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       first));
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       second));
    EXPECT_EQ(mock.acquisitions.size(), 1u);
    EXPECT_EQ(manager.resolve(kBase, 1).ownerCount, 2u);

    EXPECT_TRUE(manager.deregisterMemory(first, mock.releaseCallback()));
    EXPECT_TRUE(mock.releases.empty());
    EXPECT_EQ(manager.resolve(kBase, 1).ownerCount, 1u);
    EXPECT_TRUE(manager.deregisterMemory(second, mock.releaseCallback()));
    EXPECT_EQ(mock.releases, (std::vector<uintptr_t>{kBase}));
}

TEST(RegisteredMemoryManagerTest, ReleaseFailureIsReportedWithoutRetry) {
    RegisteredMemoryManager manager(1024);
    DescriptorMock mock;
    LogicalMemoryRegistration registration;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       2048,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       registration));
    mock.failRelease = 1;

    EXPECT_FALSE(manager.deregisterMemory(registration, mock.releaseCallback()));
    EXPECT_FALSE(registration.valid());
    EXPECT_EQ(manager.rangeCount(), 0u);
    EXPECT_EQ(mock.releases.size(), 2u);
}

TEST(RegisteredMemoryManagerTest, DeregistrationWaitsForTransferLeaseWithoutHoldingRegistryLock) {
    RegisteredMemoryManager manager(1024);
    DescriptorMock mock;
    LogicalMemoryRegistration registration;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       registration));

    auto lease = manager.resolveAndAcquire(kBase + 64, 128);
    ASSERT_TRUE(lease.valid());
    EXPECT_EQ(lease.resolution().descriptorBase, kBase);
    EXPECT_EQ(lease.resolution().registrationOffset, 64u);

    std::atomic<bool> release_called{false};
    auto deregistration = std::async(std::launch::async, [&]() {
        return manager.deregisterMemory(registration, [&](uintptr_t base) {
            release_called.store(true);
            return mock.release(base);
        });
    });

    for (int attempt = 0; attempt < 100 &&
                          manager.resolve(kBase, 1).status != RangeResolveStatus::NotFound;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(manager.resolve(kBase, 1).status, RangeResolveStatus::NotFound);
    EXPECT_EQ(deregistration.wait_for(std::chrono::milliseconds(10)),
              std::future_status::timeout);
    EXPECT_FALSE(release_called.load());

    // A non-overlapping registration can proceed while deregistration waits,
    // proving that no registry lock is held across the in-flight lifetime.
    LogicalMemoryRegistration adjacent;
    EXPECT_TRUE(manager.registerMemory(kBase + 2048,
                                       64,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       adjacent));

    lease.reset();
    EXPECT_TRUE(deregistration.get());
    EXPECT_TRUE(release_called.load());
    EXPECT_TRUE(manager.deregisterMemory(adjacent, mock.releaseCallback()));
}

TEST(RegisteredMemoryManagerTest, ConcurrentLookupDuplicateAndAdjacentRegistrationAreSafe) {
    RegisteredMemoryManager manager(1024);
    DescriptorMock mock;
    LogicalMemoryRegistration persistent;
    ASSERT_TRUE(manager.registerMemory(kBase,
                                       1024,
                                       RegisteredMemoryType::Dram,
                                       mock.acquireCallback(),
                                       mock.releaseCallback(),
                                       persistent));

    std::atomic<bool> failed{false};
    std::thread lookup_thread([&]() {
        for (int i = 0; i < 500; ++i) {
            auto lease = manager.resolveAndAcquire(kBase + (i % 512), 1);
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
                                        RegisteredMemoryType::Dram,
                                        mock.acquireCallback(),
                                        mock.releaseCallback(),
                                        duplicate) ||
                !manager.deregisterMemory(duplicate, mock.releaseCallback())) {
                failed.store(true);
            }
        }
    });
    std::thread adjacent_thread([&]() {
        for (int i = 0; i < 100; ++i) {
            LogicalMemoryRegistration adjacent;
            if (!manager.registerMemory(kBase + 1024,
                                        1024,
                                        RegisteredMemoryType::Dram,
                                        mock.acquireCallback(),
                                        mock.releaseCallback(),
                                        adjacent) ||
                !manager.deregisterMemory(adjacent, mock.releaseCallback())) {
                failed.store(true);
            }
        }
    });

    lookup_thread.join();
    duplicate_thread.join();
    adjacent_thread.join();
    EXPECT_FALSE(failed.load());
    EXPECT_TRUE(manager.deregisterMemory(persistent, mock.releaseCallback()));
    EXPECT_EQ(manager.rangeCount(), 0u);
}

} // namespace
} // namespace gtest::obj
