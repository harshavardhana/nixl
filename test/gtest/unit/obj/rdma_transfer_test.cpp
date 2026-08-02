/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/rdma.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace gtest::obj {
namespace {

using namespace nixl_obj_rdma;

class MockMemoryProvider final : public RdmaMemoryProvider {
public:
    struct TokenCall {
        uintptr_t base;
        size_t size;
        size_t offset;
        cuObjOpType_t operation;
    };

    MockMemoryProvider(uintptr_t base, size_t length, size_t chunk_size)
        : manager_(chunk_size) {
        EXPECT_TRUE(manager_.registerMemory(
            base,
            length,
            RegisteredMemoryType::Dram,
            [](uintptr_t, size_t) { return true; },
            [](uintptr_t) { return true; },
            registration_));
    }

    ~MockMemoryProvider() override {
        if (registration_.valid()) {
            manager_.deregisterMemory(registration_, [](uintptr_t) { return true; });
        }
    }

    RegisteredMemoryLease
    acquireBuffer(const void *ptr, size_t size) const override {
        return manager_.resolveAndAcquire(reinterpret_cast<uintptr_t>(ptr), size);
    }

    char *
    getToken(void *ptr, size_t size, size_t offset, cuObjOpType_t operation) override {
        tokenCalls.push_back({reinterpret_cast<uintptr_t>(ptr), size, offset, operation});
        if (mintFailures != 0) {
            --mintFailures;
            return nullptr;
        }
        return reinterpret_cast<char *>(static_cast<uintptr_t>(tokenCalls.size() + 1));
    }

    void
    putToken(char *token) override {
        releasedTokens.push_back(token);
    }

    mutable RegisteredMemoryManager manager_;
    LogicalMemoryRegistration registration_;
    size_t mintFailures = 0;
    std::vector<TokenCall> tokenCalls;
    std::vector<char *> releasedTokens;
};

class MockControlPlane final : public RdmaControlPlane {
public:
    struct PutCall {
        uint64_t address;
        uint64_t size;
    };
    struct GetCall {
        uint64_t address;
        uint64_t size;
        uint64_t objectOffset;
    };

    ssize_t
    rdmaPut(S3RdmaClientCtx &, const char *, uint64_t address, uint64_t size) override {
        puts.push_back({address, size});
        if (failures != 0) {
            --failures;
            return rdma_error;
        }
        return static_cast<ssize_t>(size);
    }

    ssize_t
    rdmaGet(S3RdmaClientCtx &,
            const char *,
            uint64_t address,
            uint64_t size,
            uint64_t object_offset) override {
        gets.push_back({address, size, object_offset});
        if (failures != 0) {
            --failures;
            return rdma_error;
        }
        return static_cast<ssize_t>(size);
    }

    size_t failures = 0;
    std::vector<PutCall> puts;
    std::vector<GetCall> gets;
};

constexpr uintptr_t kBase = 0x100000;

TEST(StandardRdmaTransferTest, PutAtRegistrationBaseUsesZeroMemoryOffset) {
    MockMemoryProvider memory(kBase, 4096, 4096);
    MockControlPlane control_plane;
    S3RdmaClientCtx context;

    EXPECT_EQ(rdmaPutWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase),
                               512),
              512);
    ASSERT_EQ(memory.tokenCalls.size(), 1u);
    EXPECT_EQ(memory.tokenCalls[0].base, kBase);
    EXPECT_EQ(memory.tokenCalls[0].size, 512u);
    EXPECT_EQ(memory.tokenCalls[0].offset, 0u);
    EXPECT_EQ(memory.tokenCalls[0].operation, CUOBJ_PUT);
    ASSERT_EQ(control_plane.puts.size(), 1u);
    EXPECT_EQ(control_plane.puts[0].address, kBase);
    EXPECT_EQ(control_plane.puts[0].size, 512u);
    EXPECT_EQ(memory.releasedTokens.size(), 1u);
}

TEST(StandardRdmaTransferTest, InteriorPutUsesBaseRelativeOffsetAndRequestedAddress) {
    MockMemoryProvider memory(kBase, 4096, 4096);
    MockControlPlane control_plane;
    S3RdmaClientCtx context;
    constexpr size_t kMemoryOffset = 321;

    EXPECT_EQ(rdmaPutWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase + kMemoryOffset),
                               256),
              256);
    ASSERT_EQ(memory.tokenCalls.size(), 1u);
    EXPECT_EQ(memory.tokenCalls[0].base, kBase);
    EXPECT_EQ(memory.tokenCalls[0].size, 256u);
    EXPECT_EQ(memory.tokenCalls[0].offset, kMemoryOffset);
    EXPECT_EQ(memory.tokenCalls[0].operation, CUOBJ_PUT);
    ASSERT_EQ(control_plane.puts.size(), 1u);
    EXPECT_EQ(control_plane.puts[0].address, kBase + kMemoryOffset);
}

TEST(StandardRdmaTransferTest, GetKeepsObjectAndMemoryOffsetsIndependent) {
    MockMemoryProvider memory(kBase, 4096, 4096);
    MockControlPlane control_plane;
    S3RdmaClientCtx context;
    constexpr size_t kMemoryOffset = 73;
    constexpr size_t kObjectOffset = 9001;

    EXPECT_EQ(rdmaGetWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase + kMemoryOffset),
                               128,
                               kObjectOffset),
              128);
    ASSERT_EQ(memory.tokenCalls.size(), 1u);
    EXPECT_EQ(memory.tokenCalls[0].base, kBase);
    EXPECT_EQ(memory.tokenCalls[0].offset, kMemoryOffset);
    EXPECT_EQ(memory.tokenCalls[0].operation, CUOBJ_GET);
    ASSERT_EQ(control_plane.gets.size(), 1u);
    EXPECT_EQ(control_plane.gets[0].address, kBase + kMemoryOffset);
    EXPECT_EQ(control_plane.gets[0].size, 128u);
    EXPECT_EQ(control_plane.gets[0].objectOffset, kObjectOffset);
}

TEST(StandardRdmaTransferTest, RejectsUnregisteredOverrunAndCrossChunkBeforeControlPlane) {
    MockMemoryProvider memory(kBase, 2048, 1024);
    S3RdmaClientCtx context;

    for (const auto &[address, size] :
         std::vector<std::pair<uintptr_t, size_t>>{{kBase - 1, 1},
                                                   {kBase + 2047, 2},
                                                   {kBase + 1023, 2}}) {
        MockControlPlane control_plane;
        const size_t token_calls_before = memory.tokenCalls.size();
        EXPECT_EQ(rdmaPutWithRetry(memory,
                                   control_plane,
                                   context,
                                   reinterpret_cast<void *>(address),
                                   size),
                  rdma_error);
        EXPECT_EQ(memory.tokenCalls.size(), token_calls_before);
        EXPECT_TRUE(control_plane.puts.empty());
        EXPECT_TRUE(control_plane.gets.empty());
    }
}

TEST(StandardRdmaTransferTest, RetryReusesResolutionAndReleasesEveryMintedToken) {
    MockMemoryProvider memory(kBase, 4096, 4096);
    MockControlPlane control_plane;
    control_plane.failures = 1;
    S3RdmaClientCtx context;

    EXPECT_EQ(rdmaGetWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase + 64),
                               512,
                               7),
              512);
    ASSERT_EQ(memory.tokenCalls.size(), 2u);
    EXPECT_EQ(memory.tokenCalls[0].base, kBase);
    EXPECT_EQ(memory.tokenCalls[1].base, kBase);
    EXPECT_EQ(memory.tokenCalls[0].offset, 64u);
    EXPECT_EQ(memory.tokenCalls[1].offset, 64u);
    EXPECT_EQ(memory.releasedTokens.size(), 2u);
    ASSERT_EQ(control_plane.gets.size(), 2u);
    EXPECT_EQ(control_plane.gets[0].objectOffset, 7u);
    EXPECT_EQ(control_plane.gets[1].objectOffset, 7u);
}

TEST(StandardRdmaTransferTest, TokenMintFailureRetriesWithoutControlPlaneOrReleaseImbalance) {
    MockMemoryProvider memory(kBase, 4096, 4096);
    memory.mintFailures = 1;
    MockControlPlane control_plane;
    S3RdmaClientCtx context;

    EXPECT_EQ(rdmaPutWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase + 8),
                               64),
              64);
    EXPECT_EQ(memory.tokenCalls.size(), 2u);
    EXPECT_EQ(control_plane.puts.size(), 1u);
    EXPECT_EQ(memory.releasedTokens.size(), 1u);
}

} // namespace
} // namespace gtest::obj
