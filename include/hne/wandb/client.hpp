#pragma once

#ifdef HNE_WANDB

#include <nlohmann/json.hpp>
#include <mutex>
#include <string>

namespace hne::wandb {

// Minimal HTTPS client wrapping libcurl. Not thread-safe on a single instance;
// the caller must serialize requests (WandbCallback does this via a single
// uploader thread).
class HttpClient {
public:
    HttpClient();
    ~HttpClient();

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    struct Response {
        long status_code = 0;
        std::string body;
        std::string error;  // libcurl transport error, empty on HTTP success
        bool ok() const { return error.empty() && status_code >= 200 && status_code < 300; }
    };

    // Set the Basic-auth credentials used for every subsequent request.
    // `api_key` is the raw W&B API key; the header is built as
    //   Authorization: Basic base64("api:" + api_key)
    void set_api_key(const std::string& api_key);

    // POST application/json body to `url`. Returns the response synchronously.
    Response post_json(const std::string& url, const std::string& json_body);

    // POST a multipart form with a single "file" part at `file_path`.
    Response post_file(const std::string& url, const std::string& file_path);

private:
    void* curl_ = nullptr;           // CURL*
    std::string auth_header_;        // cached "Authorization: Basic ..."
    std::mutex mu_;                  // guards curl handle reuse
};

// Process-wide libcurl init/cleanup refcount. Construct one of these for the
// lifetime of any HttpClient; multiple concurrent instances are safe.
class CurlGlobal {
public:
    CurlGlobal();
    ~CurlGlobal();
    CurlGlobal(const CurlGlobal&) = delete;
    CurlGlobal& operator=(const CurlGlobal&) = delete;
};

// Base64-encode a byte string. Public for testability.
std::string base64_encode(const std::string& input);

} // namespace hne::wandb

#endif // HNE_WANDB
