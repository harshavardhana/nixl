/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3/rdma.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
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
        : manager_(
              chunk_size, [](uintptr_t, size_t) { return true; }, [](uintptr_t) { return true; }) {
        EXPECT_TRUE(manager_.registerMemory(base, length, RegisteredMemoryType::Dram, registration_));
    }

    ~MockMemoryProvider() override {
        if (registration_.valid()) {
            manager_.deregisterMemory(registration_);
        }
    }

    RegisteredMemoryFragments
    acquireBuffers(const void *ptr,
                   size_t size,
                   RegisteredMemoryType memory_type) const override {
        return manager_.resolveAndAcquireFragments(
            reinterpret_cast<uintptr_t>(ptr), size, memory_type);
    }

    RegisteredMemoryLease
    acquireBuffer(const void *ptr,
                  size_t size,
                  RegisteredMemoryType memory_type) const override {
        return manager_.resolveAndAcquire(
            reinterpret_cast<uintptr_t>(ptr), size, memory_type);
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
        uint32_t partNumber;
    };
    struct GetCall {
        uint64_t address;
        uint64_t size;
        uint64_t objectOffset;
    };

    ssize_t
    rdmaPut(S3RdmaClientCtx &ctx, const char *, uint64_t address, uint64_t size) override {
        puts.push_back({address, size, ctx.part_number});
        if (failedPart != 0 && ctx.part_number == failedPart) {
            return rdma_error;
        }
        if (failures != 0) {
            --failures;
            return rdma_error;
        }
        if (!ctx.upload_id.empty()) {
            ctx.etag = "etag-" + std::to_string(ctx.part_number);
        }
        if (copyData) {
            const auto *data = reinterpret_cast<const uint8_t *>(address);
            if (ctx.upload_id.empty()) {
                object.assign(data, data + size);
            } else {
                multipartData.insert(multipartData.end(), data, data + size);
            }
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
        if (copyData) {
            if (object_offset > object.size() || size > object.size() - object_offset) {
                return rdma_error;
            }
            std::memcpy(reinterpret_cast<void *>(address), object.data() + object_offset, size);
        }
        return static_cast<ssize_t>(size);
    }

    bool
    beginMultipartUpload(S3RdmaClientCtx &ctx) override {
        ++multipartBegins;
        multipartData.clear();
        ctx.upload_id = "upload-id";
        return beginSucceeds;
    }

    bool
    completeMultipartUpload(S3RdmaClientCtx &ctx,
                            const std::vector<RdmaMultipartPart> &parts) override {
        ++multipartCompletes;
        completedParts = parts;
        if (completeSucceeds) {
            if (copyData) {
                object = multipartData;
            }
            ctx.upload_id.clear();
        }
        return completeSucceeds;
    }

    void
    abortMultipartUpload(S3RdmaClientCtx &ctx) override {
        ++multipartAborts;
        multipartData.clear();
        ctx.upload_id.clear();
    }

    bool beginSucceeds = true;
    bool completeSucceeds = true;
    bool copyData = false;
    uint32_t failedPart = 0;
    size_t multipartBegins = 0;
    size_t multipartCompletes = 0;
    size_t multipartAborts = 0;
    std::vector<RdmaMultipartPart> completedParts;
    std::vector<uint8_t> object;
    std::vector<uint8_t> multipartData;
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
                               512,
                               RegisteredMemoryType::Dram),
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
                               256,
                               RegisteredMemoryType::Dram),
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
                               kObjectOffset,
                               RegisteredMemoryType::Dram),
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

TEST(StandardRdmaTransferTest, RejectsUnregisteredAndOverrunBeforeControlPlane) {
    MockMemoryProvider memory(kBase, 2048, 1024);
    S3RdmaClientCtx context;

    for (const auto &[address, size] :
         std::vector<std::pair<uintptr_t, size_t>>{{kBase - 1, 1}, {kBase + 2047, 2}}) {
        MockControlPlane control_plane;
        const size_t token_calls_before = memory.tokenCalls.size();
        EXPECT_EQ(rdmaPutWithRetry(memory,
                                   control_plane,
                                   context,
                                   reinterpret_cast<void *>(address),
                                   size,
                                   RegisteredMemoryType::Dram),
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
                               7,
                               RegisteredMemoryType::Dram),
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
                               64,
                               RegisteredMemoryType::Dram),
              64);
    EXPECT_EQ(memory.tokenCalls.size(), 2u);
    EXPECT_EQ(control_plane.puts.size(), 1u);
    EXPECT_EQ(memory.releasedTokens.size(), 1u);
}

TEST(StandardRdmaTransferTest, FragmentedGetUsesOrderedDescriptorsAndObjectOffsets) {
    constexpr size_t kChunk = 1024;
    MockMemoryProvider memory(kBase, 3 * kChunk, kChunk);
    MockControlPlane control_plane;
    S3RdmaClientCtx context;

    EXPECT_EQ(rdmaGetWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase + 100),
                               2500,
                               77,
                               RegisteredMemoryType::Dram),
              2500);
    ASSERT_EQ(memory.tokenCalls.size(), 3u);
    EXPECT_EQ(memory.tokenCalls[0].base, kBase);
    EXPECT_EQ(memory.tokenCalls[0].size, 924u);
    EXPECT_EQ(memory.tokenCalls[0].offset, 100u);
    EXPECT_EQ(memory.tokenCalls[1].base, kBase + kChunk);
    EXPECT_EQ(memory.tokenCalls[1].size, kChunk);
    EXPECT_EQ(memory.tokenCalls[1].offset, 0u);
    EXPECT_EQ(memory.tokenCalls[2].base, kBase + 2 * kChunk);
    EXPECT_EQ(memory.tokenCalls[2].size, 552u);
    ASSERT_EQ(control_plane.gets.size(), 3u);
    EXPECT_EQ(control_plane.gets[0].objectOffset, 77u);
    EXPECT_EQ(control_plane.gets[1].objectOffset, 1001u);
    EXPECT_EQ(control_plane.gets[2].objectOffset, 2025u);
    EXPECT_EQ(memory.releasedTokens.size(), 3u);
}

TEST(StandardRdmaTransferTest, FragmentedPutUsesMultipartAndCompletesAtomically) {
    constexpr size_t kChunk = 6 * 1024 * 1024;
    constexpr size_t kStartOffset = 1024 * 1024;
    constexpr size_t kTail = 4096;
    constexpr size_t kSize = 2 * kChunk + kTail - kStartOffset;
    MockMemoryProvider memory(kBase, 2 * kChunk + kTail, kChunk);
    MockControlPlane control_plane;
    S3RdmaClientCtx context;
    context.object = "multipart-object";

    EXPECT_EQ(rdmaPutWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase + kStartOffset),
                               kSize,
                               RegisteredMemoryType::Dram),
              static_cast<ssize_t>(kSize));
    EXPECT_EQ(control_plane.multipartBegins, 1u);
    EXPECT_EQ(control_plane.multipartCompletes, 1u);
    EXPECT_EQ(control_plane.multipartAborts, 0u);
    ASSERT_EQ(control_plane.puts.size(), 3u);
    EXPECT_EQ(control_plane.puts[0].size, 5 * 1024 * 1024u);
    EXPECT_EQ(control_plane.puts[0].partNumber, 1u);
    EXPECT_EQ(control_plane.puts[1].size, kChunk);
    EXPECT_EQ(control_plane.puts[1].partNumber, 2u);
    EXPECT_EQ(control_plane.puts[2].size, kTail);
    EXPECT_EQ(control_plane.puts[2].partNumber, 3u);
    ASSERT_EQ(control_plane.completedParts.size(), 3u);
    EXPECT_EQ(control_plane.completedParts[0].etag, "etag-1");
    EXPECT_EQ(control_plane.completedParts[2].etag, "etag-3");
    ASSERT_EQ(memory.tokenCalls.size(), 3u);
    EXPECT_EQ(memory.tokenCalls[0].base, kBase);
    EXPECT_EQ(memory.tokenCalls[0].offset, kStartOffset);
    EXPECT_EQ(memory.tokenCalls[1].base, kBase + kChunk);
    EXPECT_EQ(memory.tokenCalls[2].base, kBase + 2 * kChunk);
    EXPECT_EQ(memory.releasedTokens.size(), 3u);
}

TEST(StandardRdmaTransferTest, FragmentedPutFailureAtEveryPartAbortsAndReleasesTokens) {
    constexpr size_t kChunk = 6 * 1024 * 1024;
    for (uint32_t failed_part = 1; failed_part <= 3; ++failed_part) {
        SCOPED_TRACE(failed_part);
        MockMemoryProvider memory(kBase, 2 * kChunk + 1, kChunk);
        MockControlPlane control_plane;
        control_plane.failedPart = failed_part;
        S3RdmaClientCtx context;
        context.object = "failed-multipart-object";

        EXPECT_EQ(rdmaPutWithRetry(memory,
                                   control_plane,
                                   context,
                                   reinterpret_cast<void *>(kBase),
                                   2 * kChunk + 1,
                                   RegisteredMemoryType::Dram),
                  rdma_error);
        EXPECT_EQ(control_plane.multipartBegins, 1u);
        EXPECT_EQ(control_plane.multipartCompletes, 0u);
        EXPECT_EQ(control_plane.multipartAborts, 1u);
        ASSERT_EQ(control_plane.puts.size(), failed_part + 1u);
        EXPECT_EQ(control_plane.puts[control_plane.puts.size() - 2].partNumber, failed_part);
        EXPECT_EQ(control_plane.puts.back().partNumber, failed_part);
        EXPECT_EQ(memory.releasedTokens.size(), failed_part + 1u);
    }
}

TEST(StandardRdmaTransferTest, MultipartCompletionFailureAbortsAndReportsError) {
    constexpr size_t kChunk = 6 * 1024 * 1024;
    MockMemoryProvider memory(kBase, kChunk + 1, kChunk);
    MockControlPlane control_plane;
    control_plane.completeSucceeds = false;
    S3RdmaClientCtx context;
    context.object = "failed-completion-object";

    EXPECT_EQ(rdmaPutWithRetry(
                  memory,
                  control_plane,
                  context,
                  reinterpret_cast<void *>(kBase),
                  kChunk + 1,
                  RegisteredMemoryType::Dram),
              rdma_error);
    EXPECT_EQ(control_plane.multipartBegins, 1u);
    EXPECT_EQ(control_plane.multipartCompletes, 1u);
    EXPECT_EQ(control_plane.multipartAborts, 1u);
    EXPECT_EQ(memory.releasedTokens.size(), 2u);
}

TEST(StandardRdmaTransferTest, FragmentedGetCopiesExactBytesAcrossThreeChunks) {
    constexpr size_t kChunk = 1024;
    constexpr size_t kRegistrationSize = 3 * kChunk;
    constexpr size_t kMemoryOffset = 100;
    constexpr size_t kObjectOffset = 77;
    constexpr size_t kTransferSize = 2500;
    std::vector<uint8_t> destination(kRegistrationSize, 0xCD);
    MockMemoryProvider memory(reinterpret_cast<uintptr_t>(destination.data()),
                              destination.size(),
                              kChunk);
    MockControlPlane control_plane;
    control_plane.copyData = true;
    control_plane.object.resize(kObjectOffset + kTransferSize + 19);
    for (size_t i = 0; i < control_plane.object.size(); ++i) {
        control_plane.object[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
    }
    S3RdmaClientCtx context;

    EXPECT_EQ(rdmaGetWithRetry(memory,
                               control_plane,
                               context,
                               destination.data() + kMemoryOffset,
                               kTransferSize,
                               kObjectOffset,
                               RegisteredMemoryType::Dram),
              static_cast<ssize_t>(kTransferSize));
    ASSERT_EQ(control_plane.gets.size(), 3u);
    EXPECT_EQ(std::memcmp(destination.data() + kMemoryOffset,
                          control_plane.object.data() + kObjectOffset,
                          kTransferSize),
              0);
    EXPECT_EQ(destination[kMemoryOffset - 1], 0xCD);
    EXPECT_EQ(destination[kMemoryOffset + kTransferSize], 0xCD);
}

TEST(StandardRdmaTransferTest, FragmentedPutAssemblesExactObjectAcrossThreeParts) {
    constexpr size_t kChunk = 6 * 1024 * 1024;
    constexpr size_t kTransferSize = 2 * kChunk + 257;
    std::vector<uint8_t> source(kTransferSize);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>((i * 17 + 11) & 0xFF);
    }
    MockMemoryProvider memory(
        reinterpret_cast<uintptr_t>(source.data()), source.size(), kChunk);
    MockControlPlane control_plane;
    control_plane.copyData = true;
    S3RdmaClientCtx context;
    context.object = "byte-exact-multipart-object";

    EXPECT_EQ(rdmaPutWithRetry(
                  memory,
                  control_plane,
                  context,
                  source.data(),
                  source.size(),
                  RegisteredMemoryType::Dram),
              static_cast<ssize_t>(source.size()));
    ASSERT_EQ(control_plane.puts.size(), 3u);
    ASSERT_EQ(control_plane.object.size(), source.size());
    EXPECT_EQ(std::memcmp(control_plane.object.data(), source.data(), source.size()), 0);
    EXPECT_EQ(control_plane.multipartCompletes, 1u);
    EXPECT_EQ(control_plane.multipartAborts, 0u);
}

TEST(StandardRdmaTransferTest, FragmentedPutRejectsUndersizedNonFinalS3Part) {
    constexpr size_t kChunk = 6 * 1024 * 1024;
    MockMemoryProvider memory(kBase, 2 * kChunk, kChunk);
    MockControlPlane control_plane;
    S3RdmaClientCtx context;

    EXPECT_EQ(rdmaPutWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase + kChunk - 1024),
                               2048,
                               RegisteredMemoryType::Dram),
              rdma_error);
    EXPECT_EQ(control_plane.multipartBegins, 0u);
    EXPECT_TRUE(control_plane.puts.empty());
    EXPECT_TRUE(memory.tokenCalls.empty());
}

TEST(StandardRdmaTransferTest, RejectsARegisteredAddressFromTheWrongMemoryType) {
    MockMemoryProvider memory(kBase, 4096, 4096);
    MockControlPlane control_plane;
    S3RdmaClientCtx context;

    EXPECT_EQ(rdmaPutWithRetry(memory,
                               control_plane,
                               context,
                               reinterpret_cast<void *>(kBase),
                               512,
                               RegisteredMemoryType::Vram),
              rdma_error);
    EXPECT_TRUE(memory.tokenCalls.empty());
    EXPECT_TRUE(control_plane.puts.empty());
}

} // namespace
} // namespace gtest::obj
