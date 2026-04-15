#ifdef HNE_WANDB

#include <hne/wandb/client.hpp>

#include <curl/curl.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>

namespace hne::wandb {

// ── CurlGlobal ──────────────────────────────────────────────────────────────
//
// libcurl requires process-wide init/cleanup. We refcount so multiple
// independent callbacks can coexist without double-init.
namespace {
std::mutex g_curl_init_mu;
int g_curl_init_refcount = 0;
}

CurlGlobal::CurlGlobal() {
    std::lock_guard<std::mutex> lock(g_curl_init_mu);
    if (g_curl_init_refcount++ == 0) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
}

CurlGlobal::~CurlGlobal() {
    std::lock_guard<std::mutex> lock(g_curl_init_mu);
    if (--g_curl_init_refcount == 0) {
        curl_global_cleanup();
    }
}

// ── base64 ──────────────────────────────────────────────────────────────────

std::string base64_encode(const std::string& input) {
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((input.size() + 2) / 3) * 4);

    std::size_t i = 0;
    while (i + 3 <= input.size()) {
        unsigned a = static_cast<unsigned char>(input[i]);
        unsigned b = static_cast<unsigned char>(input[i + 1]);
        unsigned c = static_cast<unsigned char>(input[i + 2]);
        out.push_back(table[(a >> 2) & 0x3F]);
        out.push_back(table[((a << 4) | (b >> 4)) & 0x3F]);
        out.push_back(table[((b << 2) | (c >> 6)) & 0x3F]);
        out.push_back(table[c & 0x3F]);
        i += 3;
    }
    if (i < input.size()) {
        unsigned a = static_cast<unsigned char>(input[i]);
        unsigned b = (i + 1 < input.size())
                        ? static_cast<unsigned char>(input[i + 1]) : 0;
        out.push_back(table[(a >> 2) & 0x3F]);
        out.push_back(table[((a << 4) | (b >> 4)) & 0x3F]);
        if (i + 1 < input.size()) {
            out.push_back(table[(b << 2) & 0x3F]);
        } else {
            out.push_back('=');
        }
        out.push_back('=');
    }
    return out;
}

// ── HttpClient ──────────────────────────────────────────────────────────────

namespace {
std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* ud) {
    auto* out = static_cast<std::string*>(ud);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}
}

HttpClient::HttpClient() {
    curl_ = curl_easy_init();
}

HttpClient::~HttpClient() {
    if (curl_) {
        curl_easy_cleanup(static_cast<CURL*>(curl_));
    }
}

void HttpClient::set_api_key(const std::string& api_key) {
    std::lock_guard<std::mutex> lock(mu_);
    if (api_key.empty()) {
        auth_header_.clear();
        return;
    }
    auth_header_ = "Authorization: Basic " + base64_encode("api:" + api_key);
}

HttpClient::Response HttpClient::post_json(const std::string& url,
                                           const std::string& json_body) {
    Response r;
    std::lock_guard<std::mutex> lock(mu_);
    if (!curl_) {
        r.error = "curl_easy_init() failed";
        return r;
    }
    auto* h = static_cast<CURL*>(curl_);
    curl_easy_reset(h);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "User-Agent: hne-wandb/0.1");
    if (!auth_header_.empty()) {
        headers = curl_slist_append(headers, auth_header_.c_str());
    }

    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE, static_cast<long>(json_body.size()));
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &r.body);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode rc = curl_easy_perform(h);
    if (rc != CURLE_OK) {
        r.error = curl_easy_strerror(rc);
    } else {
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &r.status_code);
    }

    curl_slist_free_all(headers);
    return r;
}

HttpClient::Response HttpClient::post_file(const std::string& url,
                                           const std::string& file_path) {
    Response r;
    std::lock_guard<std::mutex> lock(mu_);
    if (!curl_) {
        r.error = "curl_easy_init() failed";
        return r;
    }
    auto* h = static_cast<CURL*>(curl_);
    curl_easy_reset(h);

    curl_mime* mime = curl_mime_init(h);
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    if (curl_mime_filedata(part, file_path.c_str()) != CURLE_OK) {
        curl_mime_free(mime);
        r.error = "failed to open file: " + file_path;
        return r;
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "User-Agent: hne-wandb/0.1");
    if (!auth_header_.empty()) {
        headers = curl_slist_append(headers, auth_header_.c_str());
    }

    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &r.body);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode rc = curl_easy_perform(h);
    if (rc != CURLE_OK) {
        r.error = curl_easy_strerror(rc);
    } else {
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &r.status_code);
    }

    curl_mime_free(mime);
    curl_slist_free_all(headers);
    return r;
}

} // namespace hne::wandb

#endif // HNE_WANDB
