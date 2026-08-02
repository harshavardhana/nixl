/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "obj_test_base.h"
#include "s3_accel/dell/engine_impl.h"
#include "s3_accel/dell/rdma_interface.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <vector>

namespace gtest::obj {
namespace {

class RecordingDellS3Client final : public mockS3Client, public iDellS3RdmaClient {
public:
    struct Call {
        uintptr_t address;
        size_t size;
        size_t objectOffset;
    };

    void
    putObjectRdmaAsync(std::string_view,
                       uintptr_t address,
                       size_t size,
                       size_t object_offset,
                       std::string_view,
                       put_object_callback_t callback) override {
        puts.push_back({address, size, object_offset});
        getPendingCallbacks().push_back([callback]() { callback(true); });
    }

    void
    getObjectRdmaAsync(std::string_view,
                       uintptr_t address,
                       size_t size,
                       size_t object_offset,
                       std::string_view,
                       get_object_callback_t callback) override {
        gets.push_back({address, size, object_offset});
        getPendingCallbacks().push_back([callback]() { callback(true); });
    }

    std::vector<Call> puts;
    std::vector<Call> gets;
};

class RecordingDellCuObjClient final : public iDellCuObjClient {
public:
    struct RegistrationCall {
        uintptr_t base;
        size_t size;
    };
    struct TransferCall {
        uintptr_t base;
        size_t size;
        size_t memoryOffset;
    };

    bool
    isConnected() const override {
        return true;
    }

    cuObjErr_t
    getDescriptor(void *ptr, size_t size) override {
        registrations.push_back({reinterpret_cast<uintptr_t>(ptr), size});
        if (failRegistration != 0 && registrations.size() == failRegistration) {
            return static_cast<cuObjErr_t>(-1);
        }
        return CU_OBJ_SUCCESS;
    }

    cuObjErr_t
    putDescriptor(void *ptr) override {
        releases.push_back(reinterpret_cast<uintptr_t>(ptr));
        releaseCount.fetch_add(1);
        return CU_OBJ_SUCCESS;
    }

    ssize_t
    putObject(void *, void *ptr, size_t size, size_t memory_offset) override {
        puts.push_back({reinterpret_cast<uintptr_t>(ptr), size, memory_offset});
        return 0;
    }

    ssize_t
    getObject(void *, void *ptr, size_t size, size_t memory_offset) override {
        gets.push_back({reinterpret_cast<uintptr_t>(ptr), size, memory_offset});
        return 0;
    }

    size_t failRegistration = 0;
    std::atomic<size_t> releaseCount{0};
    std::vector<RegistrationCall> registrations;
    std::vector<uintptr_t> releases;
    std::vector<TransferCall> puts;
    std::vector<TransferCall> gets;
};

class DellRegisteredTransferTest : public testing::Test {
protected:
    void
    SetUp() override {
        params_ = {{"accelerated", "true"}, {"type", "dell"}};
        init_.localAgent = "dell-range-test";
        init_.type = "OBJ";
        init_.customParams = &params_;
        init_.enableProgTh = false;
        init_.pthrDelay = 0;
        init_.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        s3_ = std::make_shared<RecordingDellS3Client>();
        cu_ = std::make_shared<RecordingDellCuObjClient>();
        engine_ = std::make_unique<S3DellObsObjEngineImpl>(&init_, s3_, cu_);
    }

    nixlBackendMD *
    registerObject(uint64_t dev_id = 2) {
        nixlBlobDesc object = {};
        object.devId = dev_id;
        object.metaInfo = "dell-range-object";
        nixlBackendMD *metadata = nullptr;
        EXPECT_EQ(engine_->registerMem(object, OBJ_SEG, metadata), NIXL_SUCCESS);
        return metadata;
    }

    nixl_status_t
    prepare(nixl_xfer_op_t operation,
            uintptr_t address,
            size_t size,
            size_t object_offset,
            uint64_t object_dev_id,
            nixlBackendReqH *&handle) {
        nixl_meta_dlist_t local(DRAM_SEG);
        nixl_meta_dlist_t remote(OBJ_SEG);
        local.addDesc(nixlMetaDesc(address, size, 1));
        remote.addDesc(nixlMetaDesc(object_offset, size, object_dev_id));
        return engine_->prepXfer(
            operation, local, remote, init_.localAgent, init_.localAgent, handle, nullptr);
    }

    nixlBackendInitParams init_;
    nixl_b_params_t params_;
    std::shared_ptr<RecordingDellS3Client> s3_;
    std::shared_ptr<RecordingDellCuObjClient> cu_;
    std::unique_ptr<S3DellObsObjEngineImpl> engine_;
};

constexpr uintptr_t kBase = 0x100000;

TEST_F(DellRegisteredTransferTest, InteriorPutUsesDescriptorBaseAndIndependentObjectOffset) {
    nixlBlobDesc memory = {};
    memory.addr = kBase;
    memory.len = 4096;
    nixlBackendMD *memory_metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(memory, DRAM_SEG, memory_metadata), NIXL_SUCCESS);
    nixlBackendMD *object_metadata = registerObject();
    ASSERT_NE(object_metadata, nullptr);

    nixlBackendReqH *handle = nullptr;
    constexpr size_t kMemoryOffset = 123;
    constexpr size_t kObjectOffset = 777;
    ASSERT_EQ(prepare(NIXL_WRITE,
                      kBase + kMemoryOffset,
                      256,
                      kObjectOffset,
                      2,
                      handle),
              NIXL_SUCCESS);
    ASSERT_EQ(cu_->puts.size(), 1u);
    EXPECT_EQ(cu_->puts[0].base, kBase);
    EXPECT_EQ(cu_->puts[0].size, 256u);
    EXPECT_EQ(cu_->puts[0].memoryOffset, kMemoryOffset);

    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    local.addDesc(nixlMetaDesc(kBase + kMemoryOffset, 256, 1));
    remote.addDesc(nixlMetaDesc(kObjectOffset, 256, 2));
    EXPECT_EQ(engine_->postXfer(
                  NIXL_WRITE, local, remote, init_.localAgent, handle, nullptr),
              NIXL_IN_PROG);
    ASSERT_EQ(s3_->puts.size(), 1u);
    EXPECT_EQ(s3_->puts[0].address, kBase + kMemoryOffset);
    EXPECT_EQ(s3_->puts[0].objectOffset, kObjectOffset);
    s3_->execAsync();
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_SUCCESS);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(memory_metadata), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(object_metadata), NIXL_SUCCESS);
}

TEST_F(DellRegisteredTransferTest, InteriorGetUsesDescriptorBaseAndMemoryOffset) {
    nixlBlobDesc memory = {};
    memory.addr = kBase;
    memory.len = 4096;
    nixlBackendMD *memory_metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(memory, VRAM_SEG, memory_metadata), NIXL_SUCCESS);
    nixlBackendMD *object_metadata = registerObject();

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(prepare(NIXL_READ, kBase + 511, 64, 9001, 2, handle), NIXL_SUCCESS);
    ASSERT_EQ(cu_->gets.size(), 1u);
    EXPECT_EQ(cu_->gets[0].base, kBase);
    EXPECT_EQ(cu_->gets[0].memoryOffset, 511u);

    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(memory_metadata), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(object_metadata), NIXL_SUCCESS);
}

TEST_F(DellRegisteredTransferTest, RejectsUnregisteredAndOverrunBeforeCuObjectTransfer) {
    nixlBlobDesc memory = {};
    memory.addr = kBase;
    memory.len = 1024;
    nixlBackendMD *memory_metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(memory, DRAM_SEG, memory_metadata), NIXL_SUCCESS);
    nixlBackendMD *object_metadata = registerObject();

    for (const auto &[address, size] :
         std::vector<std::pair<uintptr_t, size_t>>{{kBase - 1, 1}, {kBase + 1023, 2}}) {
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(prepare(NIXL_WRITE, address, size, 0, 2, handle), NIXL_ERR_BACKEND);
        EXPECT_EQ(handle, nullptr);
    }
    EXPECT_TRUE(cu_->puts.empty());

    EXPECT_EQ(engine_->deregisterMem(memory_metadata), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(object_metadata), NIXL_SUCCESS);
}

TEST_F(DellRegisteredTransferTest, ChunksLargeRegistrationAndRejectsCrossChunkTransfer) {
    constexpr size_t kMax = CUOBJ_MAX_MEMORY_REG_SIZE;
    nixlBlobDesc memory = {};
    memory.addr = kBase;
    memory.len = kMax + 16;
    nixlBackendMD *memory_metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(memory, DRAM_SEG, memory_metadata), NIXL_SUCCESS);
    ASSERT_EQ(cu_->registrations.size(), 2u);
    EXPECT_EQ(cu_->registrations[0].base, kBase);
    EXPECT_EQ(cu_->registrations[0].size, kMax);
    EXPECT_EQ(cu_->registrations[1].base, kBase + kMax);
    EXPECT_EQ(cu_->registrations[1].size, 16u);
    nixlBackendMD *object_metadata = registerObject();

    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(prepare(NIXL_WRITE, kBase + kMax - 1, 2, 0, 2, handle),
              NIXL_ERR_BACKEND);
    EXPECT_EQ(handle, nullptr);
    EXPECT_TRUE(cu_->puts.empty());

    EXPECT_EQ(engine_->deregisterMem(memory_metadata), NIXL_SUCCESS);
    EXPECT_EQ(cu_->releaseCount.load(), 2u);
    EXPECT_EQ(engine_->deregisterMem(object_metadata), NIXL_SUCCESS);
}

TEST_F(DellRegisteredTransferTest, ChunkRegistrationFailureRollsBackTransaction) {
    constexpr size_t kMax = CUOBJ_MAX_MEMORY_REG_SIZE;
    cu_->failRegistration = 2;
    nixlBlobDesc memory = {};
    memory.addr = kBase;
    memory.len = kMax + 1;
    nixlBackendMD *metadata = nullptr;

    EXPECT_EQ(engine_->registerMem(memory, DRAM_SEG, metadata), NIXL_ERR_BACKEND);
    EXPECT_EQ(metadata, nullptr);
    EXPECT_EQ(cu_->registrations.size(), 2u);
    ASSERT_EQ(cu_->releases.size(), 1u);
    EXPECT_EQ(cu_->releases[0], kBase);
}

TEST_F(DellRegisteredTransferTest, DeregistrationRacingInFlightTransferWaitsForCompletion) {
    nixlBlobDesc memory = {};
    memory.addr = kBase;
    memory.len = 4096;
    nixlBackendMD *memory_metadata = nullptr;
    ASSERT_EQ(engine_->registerMem(memory, DRAM_SEG, memory_metadata), NIXL_SUCCESS);
    nixlBackendMD *object_metadata = registerObject();

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(prepare(NIXL_WRITE, kBase + 32, 128, 0, 2, handle), NIXL_SUCCESS);
    nixl_meta_dlist_t local(DRAM_SEG);
    nixl_meta_dlist_t remote(OBJ_SEG);
    local.addDesc(nixlMetaDesc(kBase + 32, 128, 1));
    remote.addDesc(nixlMetaDesc(0, 128, 2));
    ASSERT_EQ(engine_->postXfer(
                  NIXL_WRITE, local, remote, init_.localAgent, handle, nullptr),
              NIXL_IN_PROG);

    auto deregistration = std::async(
        std::launch::async, [&]() { return engine_->deregisterMem(memory_metadata); });
    EXPECT_EQ(deregistration.wait_for(std::chrono::milliseconds(10)),
              std::future_status::timeout);
    EXPECT_EQ(cu_->releaseCount.load(), 0u);

    s3_->execAsync();
    EXPECT_EQ(deregistration.get(), NIXL_SUCCESS);
    EXPECT_EQ(cu_->releaseCount.load(), 1u);
    EXPECT_EQ(engine_->checkXfer(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(engine_->deregisterMem(object_metadata), NIXL_SUCCESS);
}

} // namespace
} // namespace gtest::obj
