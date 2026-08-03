/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "client.h"
#include "rdma_interface.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <memory>
#include <future>
#include <optional>
#include <vector>
#include <chrono>
#include <algorithm>
#include <utility>

#include "obj_engine_registry.h"

namespace {

objAccelEngineRegistrar reg_dell(
    "dell",
    [](const nixlBackendInitParams *p) { return std::make_unique<S3DellObsObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client> s3, std::shared_ptr<iS3Client>) {
        return std::make_unique<S3DellObsObjEngineImpl>(p, std::move(s3));
    });

/**
 * RDMA context structure for cuObject operations.
 */
typedef struct rdma_ctx {
    /// RDMA descriptor string
    std::string rdma_desc;
} rdma_ctx_t;

/**
 * Validate parameters for prepXfer operation.
 * Ensures operation type, memory types, and descriptor counts are valid.
 *
 * @param operation Transfer operation type
 * @param local Local memory descriptor list
 * @param remote Remote memory descriptor list
 * @param remote_agent Remote agent identifier
 * @param local_agent Local agent identifier
 * @return true if parameters are valid, false otherwise
 */
bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (remote_agent != local_agent) {
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);
    }

    if ((local.getType() != DRAM_SEG) && (local.getType() != VRAM_SEG)) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local memory type must be VRAM_SEG or DRAM_SEG, got %d", local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local and remote descriptor counts must match. Got %d local, %d remote",
            local.descCount(),
            remote.descCount());
        return false;
    }

    return true;
}

/**
 * Transfer request handle for ObjectScale operations.
 * Contains information needed for RDMA-accelerated S3 transfers.
 */
class obsObjTransferRequestH {
public:
    /// Memory address for transfer
    uintptr_t addr;
    /// Size of data to transfer
    size_t size;
    /// Offset within object
    size_t offset;
    /// RDMA descriptor for acceleration
    std::string rdma_desc;
    /// Object key in S3
    std::string obj_key;
    /// Pins the owning cuObject descriptor through transfer completion.
    nixl_obj_rdma::RegisteredMemoryLease memoryLease;
    /// RDMA context for async operations
    rdma_ctx_t ctx;

    /**
     * Default constructor - initializes all fields to zero/empty.
     */
    obsObjTransferRequestH() : addr(0), size(0), offset(0), rdma_desc(""), obj_key("") {}

    /**
     * Constructor with basic transfer parameters.
     *
     * @param a Memory address
     * @param s Size of transfer
     * @param off Offset within object
     */
    obsObjTransferRequestH(uintptr_t a, size_t s, size_t off)
        : addr(a),
          size(s),
          offset(off),
          rdma_desc(""),
          obj_key("") {}

    ~obsObjTransferRequestH() = default;
};

/**
 * Backend request handle for ObjectScale operations.
 * Manages multiple transfer requests and their completion status.
 */
class nixlObsObjBackendReqH : public nixlBackendReqH {
public:
    /// Vector of transfer requests
    std::vector<obsObjTransferRequestH> reqs_;
    /// Futures for tracking completion status
    std::vector<std::future<nixl_status_t>> statusFutures_;

    /**
     * Default constructor.
     */
    nixlObsObjBackendReqH() = default;

    /**
     * Destructor.
     */
    ~nixlObsObjBackendReqH() = default;

    /**
     * Get the overall status of all transfer requests.
     * Checks completion of all futures and returns the first error encountered,
     * or NIXL_SUCCESS if all complete successfully, or NIXL_IN_PROG if any are pending.
     *
     * @return Overall transfer status
     */
    nixl_status_t
    getOverallStatus() {
        bool has_pending = false;
        auto it = statusFutures_.begin();
        while (it != statusFutures_.end()) {
            if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                auto current_status = it->get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    return current_status;
                }
                it = statusFutures_.erase(it);
            } else {
                ++it;
                has_pending = true;
            }
        }
        if (has_pending) {
            return NIXL_IN_PROG;
        }
        return NIXL_SUCCESS;
    }
};

/**
 * Metadata for ObjectScale operations.
 * Contains information about registered memory regions.
 */
class nixlObsObjMetadata : public nixlBackendMD {
public:
    /**
     * Constructor for object segments.
     *
     * @param nixl_mem Memory type
     * @param dev_id Device ID
     * @param obj_key Object key in S3
     */
    nixlObsObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(std::move(obj_key)) {}

    /** Constructor for memory segments (DRAM/VRAM), owning all chunks. */
    nixlObsObjMetadata(nixl_mem_t nixl_mem,
                       nixl_obj_rdma::LogicalMemoryRegistration &&registration)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(0),
          objKey(""),
          rdmaRegistration(std::move(registration)) {}

    ~nixlObsObjMetadata() = default;

    /// NIXL memory type
    nixl_mem_t nixlMem;
    /// Device ID for object segments
    uint64_t devId;
    /// Object key in S3
    std::string objKey;
    /// Every cuObject chunk belonging to one logical NIXL registration.
    nixl_obj_rdma::LogicalMemoryRegistration rdmaRegistration;
};

/**
 * cuObject get callback function.
 * Extracts RDMA descriptor from cuFile RDMA info and stores it in context.
 *
 * @param handle cuObject handle
 * @param buf Buffer to fill (unused in this implementation)
 * @param size Size of data (unused in this implementation)
 * @param offset Offset within object (unused in this implementation)
 * @param infop RDMA info containing descriptor
 * @return 0 on success
 */
static ssize_t
objectGet(const void *handle,
          char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectGet: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = cuObjClient::getCtx(handle);
    if (ctx == nullptr) {
        NIXL_ERROR << "objectGet: context is null";
        return -EINVAL;
    }
    NIXL_DEBUG << "objectGet: handle=" << handle << ", buf=" << static_cast<const void *>(buf)
               << ", size=" << size << ", offset=" << offset << ", infop=" << infop;
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

/**
 * cuObject put callback function.
 * Extracts RDMA descriptor from cuFile RDMA info and stores it in context.
 *
 * @param handle cuObject handle
 * @param buf Buffer containing data (unused in this implementation)
 * @param size Size of data (unused in this implementation)
 * @param offset Offset within object (unused in this implementation)
 * @param infop RDMA info containing descriptor
 * @return 0 on success
 */
static ssize_t
objectPut(const void *handle,
          const char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectPut: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = cuObjClient::getCtx(handle);
    if (ctx == nullptr) {
        NIXL_ERROR << "objectPut: context is null";
        return -EINVAL;
    }
    NIXL_DEBUG << "objectPut: handle=" << handle << ", buf=" << static_cast<const void *>(buf)
               << ", size=" << size << ", offset=" << offset << ", infop=" << infop;
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

/// cuObject I/O operations for ObjectScale
CUObjIOOps obs_ops = {.get = objectGet, .put = objectPut};

class DellCuObjClient final : public iDellCuObjClient {
public:
    DellCuObjClient()
        : client_(std::make_shared<cuObjClient>(obs_ops, CUOBJ_PROTO_RDMA_DC_V1)) {}

    bool
    isConnected() const override {
        return client_->isConnected();
    }

    cuObjErr_t
    getDescriptor(void *ptr, size_t size) override {
        return client_->cuMemObjGetDescriptor(ptr, size);
    }

    cuObjErr_t
    putDescriptor(void *ptr) override {
        return client_->cuMemObjPutDescriptor(ptr);
    }

    ssize_t
    putObject(void *ctx, void *ptr, size_t size, size_t memory_offset) override {
        return client_->cuObjPut(ctx, ptr, size, memory_offset);
    }

    ssize_t
    getObject(void *ctx, void *ptr, size_t size, size_t memory_offset) override {
        return client_->cuObjGet(ctx, ptr, size, memory_offset);
    }

private:
    std::shared_ptr<cuObjClient> client_;
};

} // namespace

/**
 * Constructor for S3DellObsObjEngineImpl.
 * Initializes the S3 Dell ObjectScale engine with both S3 client and cuObject client.
 *
 * @param init_params Backend initialization parameters containing custom S3 configuration
 */
S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3AccelObjEngineImpl(init_params) {
    // Disable SDK response checksum validation by default for RDMA. RDMA GET
    // responses have an empty HTTP body (Content-Length: 0) with data delivered
    // out-of-band, so SDK body-checksum validation always fails when the server
    // returns checksum headers. Transport integrity is ensured by RoCEv2 iCRC.
    // emplace() preserves any user-provided override.
    nixl_b_params_t *params_to_use = init_params->customParams;
    nixl_b_params_t local_params;
    if (!init_params->customParams) {
        local_params["resp_checksum"] = "required";
        params_to_use = &local_params;
    } else {
        init_params->customParams->emplace("resp_checksum", "required");
    }
    s3Client_ = std::make_shared<awsS3DellObsClient>(params_to_use, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Dell ObjectScale client";

    cuClient_ = std::make_shared<DellCuObjClient>();
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient failed to connect.";
        return;
    }
}

/**
 * Constructor for S3DellObsObjEngineImpl with injected S3 client.
 * Used primarily for testing with mock S3 clients.
 *
 * @param init_params Backend initialization parameters
 * @param s3_client Pre-configured S3 client (can be mock for testing)
 */
S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                                               std::shared_ptr<iS3Client> s3_client)
    : S3AccelObjEngineImpl(init_params, s3_client) {
    // See primary constructor for rationale.
    nixl_b_params_t *params_to_use = init_params->customParams;
    nixl_b_params_t local_params;
    if (!init_params->customParams) {
        local_params["resp_checksum"] = "required";
        params_to_use = &local_params;
    } else {
        init_params->customParams->emplace("resp_checksum", "required");
    }
    // Use the injected client if provided, otherwise create a new one
    if (s3_client) {
        s3Client_ = s3_client; // Use the injected mock client for testing
    } else {
        s3Client_ = std::make_shared<awsS3DellObsClient>(params_to_use, executor_);
    }

    NIXL_INFO << "Object storage backend initialized with S3 Dell ObjectScale client";

    cuClient_ = std::make_shared<DellCuObjClient>();
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient failed to connect.";
        return;
    }
}

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(
    const nixlBackendInitParams *init_params,
    std::shared_ptr<iS3Client> s3_client,
    std::shared_ptr<iDellCuObjClient> cu_client)
    : S3AccelObjEngineImpl(init_params, s3_client),
      s3Client_(std::move(s3_client)),
      cuClient_(std::move(cu_client)) {
    nixl_b_params_t *params_to_use = init_params->customParams;
    nixl_b_params_t local_params;
    if (!init_params->customParams) {
        local_params["resp_checksum"] = "required";
        params_to_use = &local_params;
    } else {
        init_params->customParams->emplace("resp_checksum", "required");
    }
    if (!s3Client_) {
        s3Client_ = std::make_shared<awsS3DellObsClient>(params_to_use, executor_);
    }
    if (!cuClient_) {
        cuClient_ = std::make_shared<DellCuObjClient>();
    }
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient failed to connect.";
    }
}

bool
S3DellObsObjEngineImpl::acquireDescriptor(uintptr_t descriptor_base, size_t registered_length) {
    const std::lock_guard<std::mutex> lock(cuClientMutex_);
    const cuObjErr_t status =
        cuClient_->getDescriptor(reinterpret_cast<void *>(descriptor_base), registered_length);
    if (status != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjGetDescriptor failed with status: " << status;
        return false;
    }
    return true;
}

bool
S3DellObsObjEngineImpl::releaseDescriptor(uintptr_t descriptor_base) {
    const std::lock_guard<std::mutex> lock(cuClientMutex_);
    const cuObjErr_t status = cuClient_->putDescriptor(reinterpret_cast<void *>(descriptor_base));
    if (status != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjPutDescriptor failed with status: " << status;
        return false;
    }
    return true;
}

/**
 * Register memory with the backend for RDMA operations.
 * Supports OBJ_SEG, DRAM_SEG, and VRAM_SEG memory types.
 * For OBJ_SEG, creates object metadata and maps device ID to object key.
 * For DRAM_SEG/VRAM_SEG, obtains cuObject descriptor for the memory region.
 *
 * @param mem Memory blob descriptor containing address, length, device ID, and metadata
 * @param nixl_mem Memory type (OBJ_SEG, DRAM_SEG, or VRAM_SEG)
 * @param out Output backend metadata handle
 * @return NIXL_SUCCESS on success, NIXL_ERR_BACKEND on cuObject failure, NIXL_ERR_NOT_SUPPORTED for
 * unsupported memory types
 */
nixl_status_t
S3DellObsObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                    const nixl_mem_t &nixl_mem,
                                    nixlBackendMD *&out) {
    out = nullptr;
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient is not connected.";
        return NIXL_ERR_BACKEND;
    }

    const auto supported_mems = {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        auto obj_md = std::make_unique<nixlObsObjMetadata>(
            nixl_mem, mem.devId, mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        {
            const std::lock_guard<std::mutex> lock(objectMapMutex_);
            devIdToObjKey_[mem.devId] = obj_md->objKey;
        }
        out = obj_md.release();
        return NIXL_SUCCESS;
    }

    NIXL_DEBUG << "registerMem: addr=" << mem.addr << ", len=" << mem.len
               << ", nixl_mem=" << nixl_mem;
    nixl_obj_rdma::LogicalMemoryRegistration registration;
    const auto memory_type = nixl_mem == VRAM_SEG
                                 ? nixl_obj_rdma::RegisteredMemoryType::Vram
                                 : nixl_obj_rdma::RegisteredMemoryType::Dram;

    const bool registered =
        registeredMemory_.registerMemory(mem.addr, mem.len, memory_type, registration);
    if (!registered) {
        NIXL_ERROR << "Transactional cuObject memory registration failed";
        return NIXL_ERR_BACKEND;
    }

    try {
        out = std::make_unique<nixlObsObjMetadata>(nixl_mem, std::move(registration)).release();
    }
    catch (...) {
        registeredMemory_.deregisterMemory(registration);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

/**
 * Deregister memory from the backend.
 * Cleans up cuObject descriptors and removes device ID mappings.
 *
 * @param meta Backend metadata handle to deregister
 * @return NIXL_SUCCESS on success, NIXL_ERR_BACKEND on cuObject failure
 */
nixl_status_t
S3DellObsObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    std::unique_ptr<nixlObsObjMetadata> md(static_cast<nixlObsObjMetadata *>(meta));
    if (!md) {
        return NIXL_SUCCESS;
    }

    if (md->nixlMem == OBJ_SEG) {
        const std::lock_guard<std::mutex> lock(objectMapMutex_);
        devIdToObjKey_.erase(md->devId);
        return NIXL_SUCCESS;
    }

    const bool released = registeredMemory_.deregisterMemory(md->rdmaRegistration);
    if (!released) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

/**
 * Prepare a transfer operation between local and remote memory.
 * Validates parameters, sets up RDMA contexts using cuObject, and creates
 * transfer request handles for subsequent execution.
 *
 * @param operation Transfer operation (NIXL_READ or NIXL_WRITE)
 * @param local Local memory descriptor list (must be DRAM_SEG or VRAM_SEG)
 * @param remote Remote memory descriptor list (must be OBJ_SEG)
 * @param remote_agent Remote agent identifier
 * @param local_agent Local agent identifier
 * @param handle Output transfer request handle
 * @param opt_args Optional backend arguments (unused)
 * @return NIXL_SUCCESS on success, NIXL_ERR_BACKEND on cuObject failure, NIXL_ERR_INVALID_PARAM on
 * validation failure
 */
nixl_status_t
S3DellObsObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 const std::string &remote_agent,
                                 const std::string &local_agent,
                                 nixlBackendReqH *&handle,
                                 const nixl_opt_b_args_t *opt_args) const {

    handle = nullptr;
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient is not connected.";
        return NIXL_ERR_BACKEND;
    }

    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlObsObjBackendReqH>();

    for (int i = 0; i < local.descCount(); ++i) {
        obsObjTransferRequestH req(local[i].addr, local[i].len, remote[i].addr);

        // Validate devId-to-object-key mapping before cuObj operations
        {
            const std::lock_guard<std::mutex> lock(objectMapMutex_);
            auto obj_key_search = devIdToObjKey_.find(remote[i].devId);
            if (obj_key_search == devIdToObjKey_.end()) {
                NIXL_ERROR << "The object segment key " << remote[i].devId
                           << " is not registered with the backend";
                return NIXL_ERR_INVALID_PARAM;
            }
            req.obj_key = obj_key_search->second;
        }

        // Preserve the existing zero-length prep behavior. The Dell S3 client
        // rejects the request when posted, as it did before range resolution.
        if (req.size != 0) {
            const auto memory_type = local.getType() == VRAM_SEG
                                         ? nixl_obj_rdma::RegisteredMemoryType::Vram
                                         : nixl_obj_rdma::RegisteredMemoryType::Dram;
            req.memoryLease =
                registeredMemory_.resolveAndAcquire(req.addr, req.size, memory_type);
            if (!req.memoryLease.valid()) {
                NIXL_ERROR << "Dell RDMA range is not contained in one registered descriptor: "
                           << "status="
                           << static_cast<int>(req.memoryLease.resolution().status)
                           << " ptr=" << reinterpret_cast<void *>(req.addr)
                           << " size=" << req.size;
                return NIXL_ERR_BACKEND;
            }

            const auto &resolution = req.memoryLease.resolution();
            void *descriptor_base = reinterpret_cast<void *>(resolution.descriptorBase);
            ssize_t cuda_status = 0;
            {
                const std::lock_guard<std::mutex> lock(cuClientMutex_);
                if (operation == NIXL_WRITE) {
                    cuda_status = cuClient_->putObject(
                        &req.ctx, descriptor_base, req.size, resolution.registrationOffset);
                } else {
                    cuda_status = cuClient_->getObject(
                        &req.ctx, descriptor_base, req.size, resolution.registrationOffset);
                }
            }
            if (cuda_status < 0) {
                NIXL_ERROR << (operation == NIXL_WRITE ? "cuObjPut" : "cuObjGet")
                           << " failed with status: " << cuda_status;
                return NIXL_ERR_BACKEND;
            }
        }
        req.rdma_desc = req.ctx.rdma_desc;

        req_h->reqs_.push_back(std::move(req));
    }

    handle = req_h.release();

    return NIXL_SUCCESS;
}

/**
 * Post a transfer operation for execution.
 * Initiates asynchronous S3 operations with RDMA descriptors obtained
 * from the preparation phase. Uses futures/promises to bridge callback
 * and polling interfaces.
 *
 * @param operation Transfer operation (NIXL_READ or NIXL_WRITE)
 * @param local Local memory descriptor list
 * @param remote Remote memory descriptor list
 * @param remote_agent Remote agent identifier
 * @param handle Transfer request handle from prepXfer
 * @param opt_args Optional backend arguments (unused)
 * @return NIXL_IN_PROG if operation started successfully
 */
nixl_status_t
S3DellObsObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 const std::string &remote_agent,
                                 nixlBackendReqH *&handle,
                                 const nixl_opt_b_args_t *opt_args) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);

    for (auto &req : req_h->reqs_) {

        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        auto memory_lease = std::make_shared<nixl_obj_rdma::RegisteredMemoryLease>(
            std::move(req.memoryLease));
        req_h->statusFutures_.push_back(status_promise->get_future());

        // S3 client interface signals completion via a callback, but NIXL API polls request handle
        // for the status code. Use future/promise pair to bridge the gap.
        // Cast to RDMA-capable client to access RDMA methods
        auto rdmaClient = dynamic_cast<iDellS3RdmaClient *>(s3Client_.get());
        if (!rdmaClient) {
            NIXL_ERROR << "Dell RDMA operations require iDellS3RdmaClient";
            status_promise->set_value(NIXL_ERR_BACKEND);
            return NIXL_IN_PROG;
        }

        if (operation == NIXL_WRITE) {
            rdmaClient->putObjectRdmaAsync(req.obj_key,
                                           req.addr,
                                           req.size,
                                           req.offset,
                                           req.rdma_desc,
                                           [status_promise, memory_lease](bool success) {
                                               memory_lease->reset();
                                               status_promise->set_value(
                                                   success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                                           });
        } else {
            rdmaClient->getObjectRdmaAsync(req.obj_key,
                                           req.addr,
                                           req.size,
                                           req.offset,
                                           req.rdma_desc,
                                           [status_promise, memory_lease](bool success) {
                                               memory_lease->reset();
                                               status_promise->set_value(
                                                   success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                                           });
        }
    }

    return NIXL_IN_PROG;
}

/**
 * Check the status of an ongoing transfer operation.
 * Polls the futures associated with the transfer request to determine
 * completion status.
 *
 * @param handle Transfer request handle to check
 * @return NIXL_SUCCESS if completed, NIXL_IN_PROG if ongoing, error code on failure
 */
nixl_status_t
S3DellObsObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

/**
 * Release a transfer request handle.
 * Cleans up resources associated with the transfer request.
 *
 * @param handle Transfer request handle to release
 * @return NIXL_SUCCESS on success
 */
nixl_status_t
S3DellObsObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}

/**
 * Get the S3 client instance.
 *
 * @return Pointer to the S3 client interface
 */
iS3Client *
S3DellObsObjEngineImpl::getClient() const {
    return s3Client_.get();
}
