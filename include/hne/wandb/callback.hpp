#pragma once

#ifdef HNE_WANDB

#include "../training/callbacks.hpp"
#include "../training/metrics.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace hne::wandb {

class HttpClient;
class CurlGlobal;

struct RunConfig {
    std::string entity;          // W&B entity (user or team); blank → server default
    std::string project;         // W&B project name
    std::string run_name;        // display name for the run
    std::string api_key;         // overrides $WANDB_API_KEY when non-empty
    std::string base_url = "https://api.wandb.ai";
    nlohmann::json hyperparameters;  // stored on the run's config blob
    bool upload_checkpoints = false;
};

// Implements ITrainerCallback; streams metrics/evals (and optionally
// checkpoints) to Weights & Biases over HTTPS on a background uploader thread.
// Never blocks the training thread on network I/O; all HTTP failures are
// surfaced via last_error() and never propagated as exceptions.
class WandbCallback : public ITrainerCallback {
public:
    explicit WandbCallback(RunConfig cfg);
    ~WandbCallback() override;

    WandbCallback(const WandbCallback&) = delete;
    WandbCallback& operator=(const WandbCallback&) = delete;

    // ── ITrainerCallback ──────────────────────────────────────────────────
    void on_training_start(const Trainer& trainer) override;
    void on_update(const Trainer& trainer, const TrainingMetrics& metrics) override;
    void on_evaluation(const Trainer& trainer, float mean_reward,
                       float mean_length) override;
    void on_checkpoint(const Trainer& trainer, const std::string& path) override;
    void on_training_end(const Trainer& trainer) override;

    // ── Status (for the ImGui panel) ──────────────────────────────────────
    bool is_running() const;
    std::string run_url() const;
    std::string last_error() const;

private:
    struct Row {
        enum Kind { History, Checkpoint };
        Kind kind = History;
        nlohmann::json history;  // set when kind == History
        std::string file_path;   // set when kind == Checkpoint
    };

    void start_run();
    void uploader_loop();
    void flush_batch(std::deque<Row>& batch);
    void set_error(std::string msg);

    RunConfig cfg_;
    std::unique_ptr<CurlGlobal> curl_init_;
    std::unique_ptr<HttpClient> http_;

    // Run identity — populated after on_training_start succeeds.
    std::string run_id_;
    std::string resolved_entity_;
    std::string run_url_;
    std::atomic<bool> run_live_{false};

    // Uploader queue.
    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::deque<Row> queue_;
    std::atomic<bool> stop_{false};
    std::thread uploader_;

    // Monotonic step counter for the history stream (`_step` field mirrors
    // TrainingMetrics::total_timesteps when available, otherwise a local
    // counter).
    std::atomic<int64_t> step_{0};

    mutable std::mutex err_mu_;
    std::string last_error_;
};

} // namespace hne::wandb

#endif // HNE_WANDB
