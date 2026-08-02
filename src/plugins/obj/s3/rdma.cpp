/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rdma.h"

#ifdef HAVE_CUOBJ_CLIENT

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <sstream>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/http/URI.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>

#include "object/s3/utils.h"
#include "object/s3/aws_sdk_init.h"
#include "common/nixl_log.h"

namespace nixl_obj_rdma {

namespace {
    // S3 ETag values are returned wrapped in double quotes; strip them.
    std::string
    stripQuotes(const std::string &s) {
        size_t b = 0, e = s.size();
        if (e >= 2 && s.front() == '"' && s.back() == '"') {
            b = 1;
            e = s.size() - 1;
        }
        return s.substr(b, e - b);
    }
} // namespace

// ---------------------------------------------------------------------------
// SharedCuObjClient
// ---------------------------------------------------------------------------

SharedCuObjClient::SharedCuObjClient() {
    try {
        // The token-based flow does not use the get/put callbacks, so empty ops
        // suffice (matches the reference SDKs' availability probe).
        client_ = std::make_unique<cuObjClient>(ops_, CUOBJ_PROTO_RDMA_DC_V1);
        connected_ = client_ && client_->isConnected();
        if (connected_) {
            NIXL_INFO << "S3 RDMA fabric connected (cuObject)";
        } else {
            NIXL_INFO << "S3 RDMA fabric not connected; transfers use HTTP";
        }
    }
    catch (const std::exception &e) {
        NIXL_WARN << "cuObjClient init failed: " << e.what() << "; transfers use HTTP";
        connected_ = false;
    }
}

SharedCuObjClient *
SharedCuObjClient::instance() {
    static SharedCuObjClient inst;
    return inst.connected_ ? &inst : nullptr;
}

bool
SharedCuObjClient::registerBuffer(void *ptr,
                                  size_t size,
                                  RegisteredMemoryType memory_type,
                                  LogicalMemoryRegistration &registration) {
    return registeredMemory_.registerMemory(
        reinterpret_cast<uintptr_t>(ptr), size, memory_type, registration);
}

bool
SharedCuObjClient::acquireDescriptor(uintptr_t descriptor_base, size_t registered_length) {
    const std::lock_guard<std::mutex> lock(mutex_);
    void *descriptor_ptr = reinterpret_cast<void *>(descriptor_base);
    const cuObjErr_t rc = client_->cuMemObjGetDescriptor(descriptor_ptr, registered_length);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjGetDescriptor failed rc=" << rc << " ptr=" << descriptor_ptr
                   << " size=" << registered_length;
        return false;
    }
    NIXL_DEBUG << "cuMemObjGetDescriptor OK ptr=" << descriptor_ptr
               << " size=" << registered_length;
    return true;
}

bool
SharedCuObjClient::releaseDescriptor(uintptr_t descriptor_base) {
    const std::lock_guard<std::mutex> lock(mutex_);
    void *descriptor_ptr = reinterpret_cast<void *>(descriptor_base);
    const cuObjErr_t rc = client_->cuMemObjPutDescriptor(descriptor_ptr);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_WARN << "cuMemObjPutDescriptor failed for ptr " << descriptor_ptr << " rc=" << rc;
        return false;
    }
    return true;
}


bool
SharedCuObjClient::deregisterBuffer(LogicalMemoryRegistration &registration) {
    return registeredMemory_.deregisterMemory(registration);
}

RegisteredMemoryLease
SharedCuObjClient::acquireBuffer(const void *ptr,
                                 size_t size,
                                 RegisteredMemoryType memory_type) const {
    return registeredMemory_.resolveAndAcquire(
        reinterpret_cast<uintptr_t>(ptr), size, memory_type);
}

RegisteredMemoryFragments
SharedCuObjClient::acquireBuffers(const void *ptr,
                                  size_t size,
                                  RegisteredMemoryType memory_type) const {
    return registeredMemory_.resolveAndAcquireFragments(
        reinterpret_cast<uintptr_t>(ptr), size, memory_type);
}

bool
SharedCuObjClient::isDeviceMemory(const void *ptr) const {
    return cuObjClient::getMemoryType(ptr) == CUOBJ_MEMORY_CUDA_DEVICE;
}

char *
SharedCuObjClient::getToken(void *ptr, size_t size, size_t offset, cuObjOpType_t op) {
    const std::lock_guard<std::mutex> lock(mutex_);
    char *token = nullptr;
    cuObjErr_t rc = client_->cuMemObjGetRDMAToken(ptr, size, offset, op, &token);
    if (rc != CU_OBJ_SUCCESS || token == nullptr) {
        NIXL_ERROR << "cuMemObjGetRDMAToken failed rc=" << rc << " ptr=" << ptr << " size=" << size
                   << " op=" << op << " token=" << static_cast<void *>(token);
        return nullptr;
    }
    return token;
}

void
SharedCuObjClient::putToken(char *token) {
    if (token == nullptr) {
        return;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    client_->cuMemObjPutRDMAToken(token);
}

// ---------------------------------------------------------------------------
// S3RdmaControlPlane
//
// === UNVERIFIED SEAM ===
// Everything in Impl touches the AWS SDK low-level HTTP/signing layer and could
// not be compiled in the authoring environment (no aws-sdk-cpp present). The
// surrounding protocol logic (rdmaPut/rdmaGet below) is SDK-agnostic and unit
// tested via rdma_protocol.h. A reviewer with the SDK should focus validation
// here: SigV4 UNSIGNED-PAYLOAD signing, URI construction (path vs virtual
// addressing), and response-header retrieval.
// ---------------------------------------------------------------------------

struct S3RdmaControlPlane::Impl {
    Aws::String scheme; // "http" / "https"
    Aws::String host; // endpoint host (no port; GetAuthority strips it)
    unsigned port = 0; // explicit port (0 => scheme default)
    Aws::String region;
    bool virtual_addressing = false;
    std::shared_ptr<Aws::Http::HttpClient> http;
    std::shared_ptr<Aws::S3::S3Client> s3;
    Aws::String access_key;
    Aws::String secret_key;
    Aws::String session_token;

    // SigV4-sign the request with payload hash "UNSIGNED-PAYLOAD". We sign
    // manually (rather than via the SDK's AWSAuthV4Signer) because that signer
    // hashes the empty body over plain HTTP — the S3 RDMA server only skips
    // content-sha256 validation when the header is exactly UNSIGNED-PAYLOAD, and
    // the data here travels out-of-band over RDMA. Mirrors minio-cpp/rs SignV4S3.
    // All non-signed headers (host, x-amz-rdma-token, content-*, checksum) must
    // already be set on the request before calling this.
    void
    signV4(Aws::Http::HttpRequest &req) const {
        using Aws::Utils::HashingUtils;
        using Aws::Utils::StringUtils;
        const Aws::String service = "s3";
        const Aws::String payload_hash = "UNSIGNED-PAYLOAD";

        Aws::Utils::DateTime now = Aws::Utils::DateTime::Now();
        const Aws::String amz_date = now.ToGmtString("%Y%m%dT%H%M%SZ");
        const Aws::String date_stamp = now.ToGmtString("%Y%m%d");

        // Host header (with port) must be signed and match what is sent.
        Aws::String host = req.GetUri().GetAuthority();
        const unsigned p = req.GetUri().GetPort();
        if (p != 0 && p != 80 && p != 443) {
            host += ":" + std::to_string(p);
        }
        req.SetHeaderValue("host", host);
        req.SetHeaderValue("x-amz-date", amz_date);
        req.SetHeaderValue("x-amz-content-sha256", payload_hash);
        if (!session_token.empty()) {
            req.SetHeaderValue("x-amz-security-token", session_token);
        }

        // Canonical headers: lowercase name, trimmed value, sorted by name.
        std::map<Aws::String, Aws::String> hdrs;
        for (const auto &h : req.GetHeaders()) {
            hdrs[StringUtils::ToLower(h.first.c_str())] = StringUtils::Trim(h.second.c_str());
        }
        Aws::String canonical_headers, signed_headers;
        for (const auto &kv : hdrs) {
            canonical_headers += kv.first + ":" + kv.second + "\n";
            if (!signed_headers.empty()) {
                signed_headers += ";";
            }
            signed_headers += kv.first;
        }

        // Canonical query string: sorted, RFC3986-encoded key=value.
        const auto qp = req.GetUri().GetQueryStringParameters();
        std::map<Aws::String, Aws::String> q(qp.begin(), qp.end());
        Aws::String canonical_query;
        for (const auto &kv : q) {
            if (!canonical_query.empty()) {
                canonical_query += "&";
            }
            canonical_query += StringUtils::URLEncode(kv.first.c_str()) + "=" +
                StringUtils::URLEncode(kv.second.c_str());
        }

        Aws::String canonical_uri = req.GetUri().GetURLEncodedPathRFC3986();
        if (canonical_uri.empty()) {
            canonical_uri = "/";
        }

        const Aws::String method =
            Aws::Http::HttpMethodMapper::GetNameForHttpMethod(req.GetMethod());
        const Aws::String canonical_request = method + "\n" + canonical_uri + "\n" +
            canonical_query + "\n" + canonical_headers + "\n" + signed_headers + "\n" +
            payload_hash;

        const Aws::String scope = date_stamp + "/" + region + "/" + service + "/aws4_request";
        const Aws::String cr_hash =
            HashingUtils::HexEncode(HashingUtils::CalculateSHA256(canonical_request));
        const Aws::String string_to_sign =
            "AWS4-HMAC-SHA256\n" + amz_date + "\n" + scope + "\n" + cr_hash;

        auto hmac = [](const Aws::Utils::ByteBuffer &key, const Aws::String &data) {
            return HashingUtils::CalculateSHA256HMAC(
                Aws::Utils::ByteBuffer(reinterpret_cast<const unsigned char *>(data.c_str()),
                                       data.size()),
                key);
        };
        const Aws::String k_secret_str = "AWS4" + secret_key;
        Aws::Utils::ByteBuffer k_secret(
            reinterpret_cast<const unsigned char *>(k_secret_str.c_str()), k_secret_str.size());
        Aws::Utils::ByteBuffer k_date = hmac(k_secret, date_stamp);
        Aws::Utils::ByteBuffer k_region = hmac(k_date, region);
        Aws::Utils::ByteBuffer k_service = hmac(k_region, service);
        Aws::Utils::ByteBuffer k_signing = hmac(k_service, "aws4_request");
        const Aws::String signature = HashingUtils::HexEncode(hmac(k_signing, string_to_sign));

        req.SetHeaderValue("authorization",
                           "AWS4-HMAC-SHA256 Credential=" + access_key + "/" + scope +
                               ", SignedHeaders=" + signed_headers + ", Signature=" + signature);
    }

    // Build the request URI for a given object key, applying path-style or
    // virtual-hosted-style addressing.
    Aws::Http::URI
    buildUri(const std::string &bucket, const std::string &key) const {
        Aws::Http::URI uri;
        uri.SetScheme(scheme == "http" ? Aws::Http::Scheme::HTTP : Aws::Http::Scheme::HTTPS);
        if (virtual_addressing) {
            uri.SetAuthority(Aws::String(bucket.c_str()) + "." + host);
            uri.SetPath("/" + Aws::String(key.c_str()));
        } else {
            uri.SetAuthority(host);
            uri.SetPath("/" + Aws::String(bucket.c_str()) + "/" + Aws::String(key.c_str()));
        }
        // GetAuthority() drops the port, so set it explicitly — otherwise the
        // request goes to the scheme default (80/443) and fails to connect.
        if (port != 0) {
            uri.SetPort(static_cast<uint16_t>(port));
        }
        return uri;
    }
};

S3RdmaControlPlane::S3RdmaControlPlane(nixl_b_params_t *custom_params) : impl_(new Impl()) {
    try {
        nixl_s3_utils::initAWSSDK();

        Aws::Client::ClientConfiguration config;
        nixl_s3_utils::configureClientCommon(config, custom_params);

        impl_->region = config.region.empty() ? Aws::String("us-east-1") : config.region;
        impl_->scheme = (config.scheme == Aws::Http::Scheme::HTTP) ? "http" : "https";
        impl_->virtual_addressing = nixl_s3_utils::getUseVirtualAddressing(custom_params);

        // Endpoint authority: explicit override (AIStor / S3-compatible) or the
        // default AWS S3 regional host.
        if (!config.endpointOverride.empty()) {
            Aws::Http::URI ep(config.endpointOverride);
            impl_->scheme = (ep.GetScheme() == Aws::Http::Scheme::HTTP) ? "http" : impl_->scheme;
            impl_->host = ep.GetAuthority();
            impl_->port = ep.GetPort();
        } else {
            impl_->host = "s3." + impl_->region + ".amazonaws.com";
            impl_->port = (impl_->scheme == "http") ? 80 : 443;
        }

        // Resolve credentials once (explicit params, else the default chain) and
        // store them for manual SigV4 signing (see Impl::signV4).
        Aws::Auth::AWSCredentials creds;
        auto explicit_creds = nixl_s3_utils::createAWSCredentials(custom_params);
        if (explicit_creds.has_value()) {
            creds = explicit_creds.value();
        } else {
            creds = Aws::Auth::DefaultAWSCredentialsProviderChain().GetAWSCredentials();
        }
        impl_->access_key = creds.GetAWSAccessKeyId();
        impl_->secret_key = creds.GetAWSSecretKey();
        impl_->session_token = creds.GetSessionToken();

        config.connectTimeoutMs = rdma_connect_timeout_secs * 1000;
        config.requestTimeoutMs = rdma_timeout_secs * 1000;
        impl_->http = Aws::Http::CreateHttpClient(config);
        impl_->s3 = std::make_shared<Aws::S3::S3Client>(
            creds,
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            impl_->virtual_addressing);

        valid_ = impl_->http != nullptr && impl_->s3 != nullptr && !impl_->access_key.empty();
    }
    catch (const std::exception &e) {
        NIXL_WARN << "S3 RDMA control plane init failed: " << e.what();
        valid_ = false;
    }
}

S3RdmaControlPlane::~S3RdmaControlPlane() = default;

bool
S3RdmaControlPlane::beginMultipartUpload(S3RdmaClientCtx &ctx) {
    if (!valid_ || ctx.bucket.empty() || ctx.object.empty()) {
        return false;
    }

    Aws::S3::Model::CreateMultipartUploadRequest request;
    request.WithBucket(ctx.bucket.c_str()).WithKey(ctx.object.c_str());
    const auto outcome = impl_->s3->CreateMultipartUpload(request);
    if (!outcome.IsSuccess()) {
        NIXL_ERROR << "CreateMultipartUpload failed for key=" << ctx.object << ": "
                   << outcome.GetError().GetExceptionName() << ": "
                   << outcome.GetError().GetMessage();
        return false;
    }

    ctx.upload_id = outcome.GetResult().GetUploadId().c_str();
    return !ctx.upload_id.empty();
}

bool
S3RdmaControlPlane::completeMultipartUpload(
    S3RdmaClientCtx &ctx, const std::vector<RdmaMultipartPart> &parts) {
    if (!valid_ || ctx.upload_id.empty() || parts.empty()) {
        return false;
    }

    Aws::S3::Model::CompletedMultipartUpload completed;
    for (const auto &part : parts) {
        if (part.partNumber == 0 || part.partNumber > 10000 || part.etag.empty()) {
            return false;
        }
        Aws::S3::Model::CompletedPart completed_part;
        completed_part.WithPartNumber(static_cast<int>(part.partNumber))
            .WithETag(part.etag.c_str());
        completed.AddParts(std::move(completed_part));
    }

    Aws::S3::Model::CompleteMultipartUploadRequest request;
    request.WithBucket(ctx.bucket.c_str())
        .WithKey(ctx.object.c_str())
        .WithUploadId(ctx.upload_id.c_str())
        .WithMultipartUpload(std::move(completed));
    const auto outcome = impl_->s3->CompleteMultipartUpload(request);
    if (!outcome.IsSuccess()) {
        NIXL_ERROR << "CompleteMultipartUpload failed for key=" << ctx.object << ": "
                   << outcome.GetError().GetExceptionName() << ": "
                   << outcome.GetError().GetMessage();
        return false;
    }

    ctx.etag = stripQuotes(outcome.GetResult().GetETag().c_str());
    ctx.upload_id.clear();
    ctx.part_number = 0;
    return true;
}

void
S3RdmaControlPlane::abortMultipartUpload(S3RdmaClientCtx &ctx) {
    if (!valid_ || ctx.upload_id.empty()) {
        return;
    }

    Aws::S3::Model::AbortMultipartUploadRequest request;
    request.WithBucket(ctx.bucket.c_str())
        .WithKey(ctx.object.c_str())
        .WithUploadId(ctx.upload_id.c_str());
    const auto outcome = impl_->s3->AbortMultipartUpload(request);
    if (!outcome.IsSuccess()) {
        NIXL_WARN << "AbortMultipartUpload failed for key=" << ctx.object << ": "
                  << outcome.GetError().GetExceptionName() << ": "
                  << outcome.GetError().GetMessage();
    }
    ctx.upload_id.clear();
    ctx.part_number = 0;
}

ssize_t
S3RdmaControlPlane::rdmaPut(S3RdmaClientCtx &ctx,
                            const char *token,
                            uint64_t buf_addr,
                            uint64_t size) {
    try {
        Aws::Http::URI uri = impl_->buildUri(ctx.bucket, ctx.object);
        if (!ctx.upload_id.empty()) {
            if (ctx.part_number == 0 || ctx.part_number > 10000) {
                NIXL_ERROR << "rdmaPut: invalid partNumber " << ctx.part_number
                           << " (expected 1..10000) for key=" << ctx.object;
                return rdma_error;
            }
            uri.AddQueryStringParameter("uploadId", ctx.upload_id.c_str());
            uri.AddQueryStringParameter("partNumber", std::to_string(ctx.part_number).c_str());
        }

        auto req =
            Aws::Http::CreateHttpRequest(uri,
                                         Aws::Http::HttpMethod::HTTP_PUT,
                                         Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
        req->SetHeaderValue("x-amz-content-sha256", unsigned_payload);
        req->SetHeaderValue(amz_rdma_token, formatRdmaToken(token, buf_addr, size).c_str());
        req->SetHeaderValue("content-type", "application/octet-stream");
        req->SetContentLength("0");
        if (!ctx.checksum_crc64nvme.empty()) {
            req->SetHeaderValue("x-amz-checksum-crc64nvme", ctx.checksum_crc64nvme.c_str());
        }

        impl_->signV4(*req); // manual SigV4 with UNSIGNED-PAYLOAD

        auto resp = impl_->http->MakeRequest(req);
        if (!resp) {
            NIXL_ERROR << "rdmaPut: MakeRequest returned null for key=" << ctx.object;
            return rdma_error;
        }

        const int http_status = static_cast<int>(resp->GetResponseCode());
        const std::string etag =
            resp->HasHeader("etag") ? stripQuotes(resp->GetHeader("etag").c_str()) : "";

        // Success: the server completed the RDMA_READ and returns a standard
        // HTTP 200 + ETag (the object payload moved out-of-band, so the HTTP body
        // is empty). Matches minio-cpp/minio-rs rdmaPut.
        if (http_status == 200 && !etag.empty()) {
            ctx.etag = etag;
            if (resp->HasHeader("x-amz-checksum-crc64nvme")) {
                ctx.checksum_crc64nvme = resp->GetHeader("x-amz-checksum-crc64nvme").c_str();
            }
            return static_cast<ssize_t>(size);
        }

        // Otherwise inspect the RDMA reply marker: 501 (or absent) => declined.
        const std::string reply = resp->HasHeader(amz_rdma_reply) ?
            std::string(resp->GetHeader(amz_rdma_reply).c_str()) :
            "";
        const int reply_code = parseRdmaReply(reply);
        std::ostringstream body;
        body << resp->GetResponseBody().rdbuf();
        if (reply_code == static_cast<int>(rdma_not_supported)) {
            NIXL_ERROR << "rdmaPut declined: http=" << http_status << " x-amz-rdma-reply='" << reply
                       << "' url=" << uri.GetURIString() << " body=" << body.str().substr(0, 400)
                       << " key=" << ctx.object;
            return rdma_not_supported;
        }
        NIXL_ERROR << "rdmaPut failed: http=" << http_status << " x-amz-rdma-reply='" << reply
                   << "' reply_code=" << reply_code << " key=" << ctx.object
                   << " body=" << body.str().substr(0, 400);
        return rdma_error;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "rdmaPut failed: " << e.what();
        return rdma_error;
    }
}

ssize_t
S3RdmaControlPlane::rdmaGet(S3RdmaClientCtx &ctx,
                            const char *token,
                            uint64_t buf_addr,
                            uint64_t size,
                            uint64_t offset) {
    try {
        Aws::Http::URI uri = impl_->buildUri(ctx.bucket, ctx.object);
        auto req =
            Aws::Http::CreateHttpRequest(uri,
                                         Aws::Http::HttpMethod::HTTP_GET,
                                         Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
        req->SetHeaderValue("x-amz-content-sha256", unsigned_payload);
        req->SetHeaderValue(amz_rdma_token, formatRdmaToken(token, buf_addr, size).c_str());
        // Byte-range fetch when reading a slice of the object (server replies 206).
        if (size != 0) {
            req->SetHeaderValue(
                "range",
                ("bytes=" + std::to_string(offset) + "-" + std::to_string(offset + size - 1))
                    .c_str());
        }

        impl_->signV4(*req);

        auto resp = impl_->http->MakeRequest(req);
        if (!resp) {
            NIXL_ERROR << "rdmaGet: MakeRequest returned null for key=" << ctx.object;
            return rdma_error;
        }

        // GET is inherently fail-safe: a non-RDMA server omits x-amz-rdma-reply,
        // which parseRdmaReply maps to "declined" (caller errors under
        // accelerated=true).
        const int http_status = static_cast<int>(resp->GetResponseCode());
        const std::string reply = resp->HasHeader(amz_rdma_reply) ?
            std::string(resp->GetHeader(amz_rdma_reply).c_str()) :
            "";
        const int reply_code = parseRdmaReply(reply);
        if (reply_code == static_cast<int>(rdma_not_supported)) {
            return rdma_not_supported;
        }
        if (reply_code != rdma_reply_success && reply_code != rdma_reply_partial_content) {
            NIXL_ERROR << "rdmaGet failed: http=" << http_status << " x-amz-rdma-reply='" << reply
                       << "' reply_code=" << reply_code << " key=" << ctx.object;
            return rdma_error;
        }

        if (resp->HasHeader("etag")) {
            ctx.etag = stripQuotes(resp->GetHeader("etag").c_str());
        }

        // Trust the server's reported transferred byte count (can be < requested
        // for ranged/partial GETs).
        if (resp->HasHeader(amz_rdma_bytes_transferred)) {
            try {
                const long long n = std::stoll(resp->GetHeader(amz_rdma_bytes_transferred).c_str());
                return n < 0 ? rdma_error : static_cast<ssize_t>(n);
            }
            catch (const std::exception &) {
                return rdma_error;
            }
        }
        return static_cast<ssize_t>(size);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "rdmaGet failed: " << e.what();
        return rdma_error;
    }
}

// ---------------------------------------------------------------------------
// Retry wrappers (token lifecycle + one transient retry). A token-mint failure
// is itself transient (cuObject NIC selection / registration hiccup), so it is
// retried rather than aborting on the first attempt.
// ---------------------------------------------------------------------------

namespace {

constexpr size_t s3_min_multipart_part_size = 5 * 1024 * 1024;

size_t
fragmentSize(const RegisteredMemoryResolution &resolution, size_t remaining) {
    return std::min(remaining, resolution.registeredLength - resolution.registrationOffset);
}

ssize_t
transferFragmentWithRetry(RdmaMemoryProvider &rdma,
                          RdmaControlPlane &cp,
                          S3RdmaClientCtx &ctx,
                          const RegisteredMemoryResolution &resolution,
                          size_t fragment_size,
                          cuObjOpType_t operation,
                          size_t object_offset) {
    void *descriptor_base = reinterpret_cast<void *>(resolution.descriptorBase);
    const uint64_t data_address =
        static_cast<uint64_t>(resolution.descriptorBase + resolution.registrationOffset);
    ssize_t ret = rdma_error;
    for (int attempt = 0; attempt < rdma_max_attempts; ++attempt) {
        char *token =
            rdma.getToken(descriptor_base, fragment_size, resolution.registrationOffset, operation);
        if (token == nullptr) {
            continue;
        }
        try {
            ret = operation == CUOBJ_PUT
                      ? cp.rdmaPut(ctx, token, data_address, fragment_size)
                      : cp.rdmaGet(ctx, token, data_address, fragment_size, object_offset);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "RDMA control-plane exception: " << e.what();
            ret = rdma_error;
        }
        catch (...) {
            NIXL_ERROR << "RDMA control-plane exception";
            ret = rdma_error;
        }
        rdma.putToken(token);
        if (ret == static_cast<ssize_t>(fragment_size) || ret == rdma_not_supported) {
            break;
        }
        if (ret > 0) {
            NIXL_WARN << "RDMA fragment transferred " << ret << " of " << fragment_size
                      << " bytes; retrying the complete fragment";
            ret = rdma_error;
        }
    }
    return ret;
}

void
abortMultipartNoThrow(RdmaControlPlane &cp, S3RdmaClientCtx &ctx) {
    try {
        cp.abortMultipartUpload(ctx);
    }
    catch (const std::exception &e) {
        NIXL_WARN << "AbortMultipartUpload threw: " << e.what();
    }
    catch (...) {
        NIXL_WARN << "AbortMultipartUpload threw";
    }
}

} // namespace

ssize_t
rdmaPutWithRetry(RdmaMemoryProvider &rdma,
                 RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 size_t size,
                 RegisteredMemoryType memory_type) {
    RegisteredMemoryFragments fragments = rdma.acquireBuffers(buf, size, memory_type);
    if (!fragments.valid()) {
        NIXL_ERROR << "RDMA PUT range is not fully registered: status="
                   << static_cast<int>(fragments.status) << " ptr=" << buf << " size=" << size;
        return rdma_error;
    }
    if (size > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
        NIXL_ERROR << "RDMA PUT size exceeds ssize_t result range: " << size;
        return rdma_error;
    }

    if (fragments.leases.size() == 1) {
        return transferFragmentWithRetry(rdma,
                                         cp,
                                         ctx,
                                         fragments.leases.front().resolution(),
                                         size,
                                         CUOBJ_PUT,
                                         0);
    }
    if (fragments.leases.size() > 10000) {
        NIXL_ERROR << "RDMA multipart PUT needs " << fragments.leases.size()
                   << " parts, exceeding the S3 limit of 10000";
        return rdma_error;
    }

    size_t remaining = size;
    for (size_t i = 0; i + 1 < fragments.leases.size(); ++i) {
        const size_t fragment_size = fragmentSize(fragments.leases[i].resolution(), remaining);
        if (fragment_size < s3_min_multipart_part_size) {
            NIXL_ERROR << "RDMA multipart PUT fragment " << (i + 1) << " is " << fragment_size
                       << " bytes; non-final S3 parts require at least "
                       << s3_min_multipart_part_size << " bytes";
            return rdma_error;
        }
        remaining -= fragment_size;
    }

    try {
        if (!cp.beginMultipartUpload(ctx)) {
            if (!ctx.upload_id.empty()) {
                abortMultipartNoThrow(cp, ctx);
            }
            NIXL_ERROR << "RDMA multipart PUT could not be initiated for key=" << ctx.object;
            return rdma_error;
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "CreateMultipartUpload threw: " << e.what();
        if (!ctx.upload_id.empty()) {
            abortMultipartNoThrow(cp, ctx);
        }
        return rdma_error;
    }
    catch (...) {
        NIXL_ERROR << "CreateMultipartUpload threw";
        if (!ctx.upload_id.empty()) {
            abortMultipartNoThrow(cp, ctx);
        }
        return rdma_error;
    }

    std::vector<RdmaMultipartPart> completed_parts;
    completed_parts.reserve(fragments.leases.size());
    size_t bytes_done = 0;
    for (size_t i = 0; i < fragments.leases.size(); ++i) {
        const auto &resolution = fragments.leases[i].resolution();
        const size_t fragment_size = fragmentSize(resolution, size - bytes_done);
        ctx.part_number = static_cast<uint32_t>(i + 1);
        ctx.etag.clear();
        const ssize_t ret = transferFragmentWithRetry(
            rdma, cp, ctx, resolution, fragment_size, CUOBJ_PUT, 0);
        if (ret != static_cast<ssize_t>(fragment_size) || ctx.etag.empty()) {
            NIXL_ERROR << "RDMA multipart PUT failed at part " << ctx.part_number
                       << " after " << bytes_done << " bytes";
            abortMultipartNoThrow(cp, ctx);
            return ret < 0 ? ret : rdma_error;
        }
        completed_parts.push_back({ctx.part_number, ctx.etag});
        bytes_done += fragment_size;
    }

    try {
        if (!cp.completeMultipartUpload(ctx, completed_parts)) {
            NIXL_ERROR << "RDMA multipart PUT completion failed for key=" << ctx.object;
            abortMultipartNoThrow(cp, ctx);
            return rdma_error;
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "CompleteMultipartUpload threw: " << e.what();
        abortMultipartNoThrow(cp, ctx);
        return rdma_error;
    }
    catch (...) {
        NIXL_ERROR << "CompleteMultipartUpload threw";
        abortMultipartNoThrow(cp, ctx);
        return rdma_error;
    }
    return static_cast<ssize_t>(bytes_done);
}

ssize_t
rdmaGetWithRetry(RdmaMemoryProvider &rdma,
                 RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 size_t size,
                 size_t offset,
                 RegisteredMemoryType memory_type) {
    RegisteredMemoryFragments fragments = rdma.acquireBuffers(buf, size, memory_type);
    if (!fragments.valid()) {
        NIXL_ERROR << "RDMA GET range is not fully registered: status="
                   << static_cast<int>(fragments.status) << " ptr=" << buf << " size=" << size;
        return rdma_error;
    }
    if (size > static_cast<size_t>(std::numeric_limits<ssize_t>::max()) ||
        (size != 0 && offset > std::numeric_limits<size_t>::max() - (size - 1))) {
        NIXL_ERROR << "RDMA GET size or object range overflows";
        return rdma_error;
    }

    size_t bytes_done = 0;
    for (const auto &lease : fragments.leases) {
        const auto &resolution = lease.resolution();
        const size_t fragment_size = fragmentSize(resolution, size - bytes_done);
        const ssize_t ret = transferFragmentWithRetry(rdma,
                                                      cp,
                                                      ctx,
                                                      resolution,
                                                      fragment_size,
                                                      CUOBJ_GET,
                                                      offset + bytes_done);
        if (ret != static_cast<ssize_t>(fragment_size)) {
            NIXL_ERROR << "RDMA fragmented GET failed after " << bytes_done << " bytes";
            return ret < 0 ? ret : rdma_error;
        }
        bytes_done += fragment_size;
    }
    return static_cast<ssize_t>(bytes_done);
}

} // namespace nixl_obj_rdma

#endif // HAVE_CUOBJ_CLIENT
