#ifdef HNE_WANDB

#include <hne/wandb/callback.hpp>
#include <hne/wandb/client.hpp>
#include <hne/training/trainer.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <sstream>
#include <utility>

namespace hne::wandb {

namespace {

std::string random_run_id() {
    // 8 hex chars — cheap client-side identifier. If the server returns its
    // own run id we replace this with the server's value.
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream os;
    os << std::hex << (dist(rng) & 0xFFFFFFFFULL);
    return os.str();
}

// W&B stores run config as {key: {value: X, desc: null}, ...}. Flatten an
// arbitrary JSON object one level deep; nested objects are preserved as-is
// inside the `value` field, matching the Python SDK.
nlohmann::json flatten_config(const nlohmann::json& hp) {
    nlohmann::json out = nlohmann::json::object();
    if (!hp.is_object()) {
        return out;
    }
    for (auto it = hp.begin(); it != hp.end(); ++it) {
        out[it.key()] = {{"value", it.value()}, {"desc", nullptr}};
    }
    return out;
}

std::string resolve_api_key(const RunConfig& cfg) {
    if (!cfg.api_key.empty()) return cfg.api_key;
    if (const char* env = std::getenv("WANDB_API_KEY"); env && *env) {
        return env;
    }
    return {};
}

} // namespace

WandbCallback::WandbCallback(RunConfig cfg)
    : cfg_(std::move(cfg)),
      curl_init_(std::make_unique<CurlGlobal>()),
      http_(std::make_unique<HttpClient>()) {
    const auto key = resolve_api_key(cfg_);
    if (key.empty()) {
        set_error("No W&B API key provided (set WANDB_API_KEY or the panel field).");
    } else {
        http_->set_api_key(key);
    }
    uploader_ = std::thread(&WandbCallback::uploader_loop, this);
}

WandbCallback::~WandbCallback() {
    {
        std::lock_guard<std::mutex> lock(q_mu_);
        stop_ = true;
    }
    q_cv_.notify_all();
    if (uploader_.joinable()) {
        uploader_.join();
    }
}

bool WandbCallback::is_running() const {
    return run_live_.load();
}

std::string WandbCallback::run_url() const {
    // run_url_ is written only by on_training_start (uploader thread does not
    // touch it); the atomic run_live_ acts as a publication fence.
    return run_live_.load() ? run_url_ : std::string{};
}

std::string WandbCallback::last_error() const {
    std::lock_guard<std::mutex> lock(err_mu_);
    return last_error_;
}

void WandbCallback::set_error(std::string msg) {
    std::lock_guard<std::mutex> lock(err_mu_);
    last_error_ = std::move(msg);
}

// ── Trainer hooks ───────────────────────────────────────────────────────────
//
// These run on the training thread; they must be cheap and must never throw.

void WandbCallback::on_training_start(const Trainer& /*trainer*/) {
    start_run();
}

void WandbCallback::on_update(const Trainer& /*trainer*/,
                               const TrainingMetrics& m) {
    nlohmann::json row = {
        {"iteration", m.iteration},
        {"total_timesteps", m.total_timesteps},
        {"mean_episode_reward", m.mean_episode_reward},
        {"mean_episode_length", m.mean_episode_length},
    };
    for (const auto& [k, v] : m.algorithm_metrics) {
        row[k] = v;
    }
    const int64_t s = (m.total_timesteps > 0)
                          ? m.total_timesteps : step_.fetch_add(1) + 1;
    row["_step"] = s;

    Row r;
    r.kind = Row::History;
    r.history = std::move(row);
    {
        std::lock_guard<std::mutex> lock(q_mu_);
        queue_.push_back(std::move(r));
    }
    q_cv_.notify_one();
}

void WandbCallback::on_evaluation(const Trainer& /*trainer*/,
                                   float mean_reward, float mean_length) {
    nlohmann::json row = {
        {"eval/mean_reward", mean_reward},
        {"eval/mean_length", mean_length},
        {"_step", step_.fetch_add(1) + 1},
    };
    Row r;
    r.kind = Row::History;
    r.history = std::move(row);
    {
        std::lock_guard<std::mutex> lock(q_mu_);
        queue_.push_back(std::move(r));
    }
    q_cv_.notify_one();
}

void WandbCallback::on_checkpoint(const Trainer& /*trainer*/,
                                   const std::string& path) {
    if (!cfg_.upload_checkpoints) return;
    Row r;
    r.kind = Row::Checkpoint;
    r.file_path = path;
    {
        std::lock_guard<std::mutex> lock(q_mu_);
        queue_.push_back(std::move(r));
    }
    q_cv_.notify_one();
}

void WandbCallback::on_training_end(const Trainer& /*trainer*/) {
    // Wait for the queue to drain — bounded by uploader's batch deadline. We
    // don't stop the uploader here (the destructor will); this just gives the
    // final rows a chance to ship before destruction.
    using namespace std::chrono_literals;
    for (int i = 0; i < 60; ++i) {
        {
            std::lock_guard<std::mutex> lock(q_mu_);
            if (queue_.empty()) break;
        }
        std::this_thread::sleep_for(100ms);
    }
    run_live_ = false;
}

// ── Run lifecycle ───────────────────────────────────────────────────────────

void WandbCallback::start_run() {
    if (resolve_api_key(cfg_).empty()) {
        return;  // error already surfaced in constructor
    }
    run_id_ = random_run_id();
    resolved_entity_ = cfg_.entity;

    // W&B upsertBucket GraphQL mutation — creates-or-gets the run. We send a
    // minimal shape that the real server accepts; if the endpoint rejects us
    // we record the error but still go live in "best-effort" mode so the
    // training loop is never blocked.
    nlohmann::json variables = {
        {"name", run_id_},
        {"project", cfg_.project},
        {"entity", cfg_.entity.empty() ? nullptr : nlohmann::json(cfg_.entity)},
        {"displayName", cfg_.run_name.empty() ? run_id_ : cfg_.run_name},
        {"config", flatten_config(cfg_.hyperparameters).dump()},
    };
    nlohmann::json gql = {
        {"query",
         "mutation UpsertBucket($name: String!, $project: String, "
         "$entity: String, $displayName: String, $config: JSONString) { "
         "upsertBucket(input: {name: $name, project: $project, entity: $entity, "
         "displayName: $displayName, config: $config}) { "
         "bucket { id name displayName } } }"},
        {"variables", variables},
    };

    const auto url = cfg_.base_url + "/graphql";
    auto resp = http_->post_json(url, gql.dump());
    if (!resp.ok()) {
        set_error("upsertBucket failed: " +
                  (resp.error.empty()
                       ? ("HTTP " + std::to_string(resp.status_code))
                       : resp.error));
        // Fall through — run is still considered live so history rows queue up;
        // file_stream posts will also fail but that's surfaced separately.
    } else {
        try {
            auto j = nlohmann::json::parse(resp.body);
            if (j.contains("data") && j["data"].contains("upsertBucket") &&
                j["data"]["upsertBucket"].contains("bucket")) {
                const auto& b = j["data"]["upsertBucket"]["bucket"];
                if (b.contains("name") && b["name"].is_string()) {
                    run_id_ = b["name"].get<std::string>();
                }
            }
        } catch (...) {
            // Non-JSON response body — leave client-side run_id in place.
        }
    }

    // Build the user-visible URL. api.wandb.ai → app.wandb.ai for the browser.
    std::string app_url = cfg_.base_url;
    const std::string api_host = "api.wandb.ai";
    if (auto pos = app_url.find(api_host); pos != std::string::npos) {
        app_url.replace(pos, api_host.size(), "wandb.ai");
    }
    run_url_ = app_url + "/" +
               (cfg_.entity.empty() ? "me" : cfg_.entity) + "/" +
               cfg_.project + "/runs/" + run_id_;

    run_live_ = true;
}

// ── Uploader thread ─────────────────────────────────────────────────────────

void WandbCallback::uploader_loop() {
    using namespace std::chrono_literals;
    constexpr std::size_t kMaxBatch = 50;
    constexpr auto kFlushInterval = 2s;

    while (true) {
        std::deque<Row> batch;
        {
            std::unique_lock<std::mutex> lock(q_mu_);
            q_cv_.wait_for(lock, kFlushInterval, [this] {
                return stop_ || !queue_.empty();
            });

            if (stop_ && queue_.empty()) {
                return;
            }

            const auto take = std::min(kMaxBatch, queue_.size());
            for (std::size_t i = 0; i < take; ++i) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }

        if (batch.empty()) continue;
        if (!run_live_.load()) {
            // Run never came up (bad creds, server down). Drop rows to avoid
            // unbounded memory growth; error is already surfaced.
            continue;
        }
        flush_batch(batch);
    }
}

void WandbCallback::flush_batch(std::deque<Row>& batch) {
    // Split into history rows and checkpoints; both flow through the same
    // /files/{entity}/{project}/{run}/file_stream endpoint.
    nlohmann::json files = nlohmann::json::object();
    nlohmann::json history_lines = nlohmann::json::array();
    std::vector<std::string> checkpoint_paths;

    for (auto& row : batch) {
        switch (row.kind) {
            case Row::History:
                history_lines.push_back(row.history.dump());
                break;
            case Row::Checkpoint:
                checkpoint_paths.push_back(std::move(row.file_path));
                break;
        }
    }

    if (!history_lines.empty()) {
        files["wandb-history.jsonl"] = {
            {"offset", step_.load() - static_cast<int64_t>(history_lines.size())},
            {"content", history_lines},
        };
        nlohmann::json body = {{"files", files}};

        const std::string url = cfg_.base_url + "/files/" +
                                (resolved_entity_.empty() ? "me" : resolved_entity_) +
                                "/" + cfg_.project + "/" + run_id_ + "/file_stream";
        auto resp = http_->post_json(url, body.dump());
        if (!resp.ok()) {
            set_error("file_stream failed: " +
                      (resp.error.empty()
                           ? ("HTTP " + std::to_string(resp.status_code))
                           : resp.error));
        }
    }

    // Checkpoints go as separate multipart uploads (may be large).
    for (const auto& path : checkpoint_paths) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            set_error("checkpoint file missing: " + path);
            continue;
        }
        const auto filename = std::filesystem::path(path).filename().string();
        const std::string url = cfg_.base_url + "/files/" +
                                (resolved_entity_.empty() ? "me" : resolved_entity_) +
                                "/" + cfg_.project + "/" + run_id_ + "/" + filename;
        auto resp = http_->post_file(url, path);
        if (!resp.ok()) {
            set_error("checkpoint upload failed: " +
                      (resp.error.empty()
                           ? ("HTTP " + std::to_string(resp.status_code))
                           : resp.error));
        }
    }
}

} // namespace hne::wandb

#endif // HNE_WANDB
