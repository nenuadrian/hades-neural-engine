#pragma once

#ifdef HNE_TRAINING

#include "callbacks.hpp"
#include "metrics.hpp"
#include "trainer_config.hpp"
#include "../core/algorithm.hpp"
#include "../core/environment.hpp"
#include "../core/policy.hpp"
#include "../core/replay_buffer.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace hne {

class Trainer {
public:
    enum class State : uint8_t {
        Idle,
        Running,
        Paused,
        Finished,
        Error
    };

    explicit Trainer(const TrainerConfig& config);
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    void set_environment_factory(VectorizedEnv::EnvFactory factory);
    void set_policy(std::shared_ptr<IPolicy> policy);
    void set_algorithm(std::unique_ptr<IAlgorithm> algorithm);
    void add_callback(std::shared_ptr<ITrainerCallback> callback);

    void train_async();
    void train();

    void request_stop();
    void pause();
    void resume();

    [[nodiscard]] State state() const;
    [[nodiscard]] int32_t current_iteration() const;
    [[nodiscard]] int64_t total_timesteps() const;
    [[nodiscard]] TrainingMetrics latest_metrics() const;

    struct EvalResult {
        float total_reward = 0.0f;
        int32_t episode_length = 0;
        nlohmann::json info;
    };

    EvalResult evaluate(IEnvironment& eval_env, bool deterministic = true,
                        int32_t max_steps = 10000);

    bool save_checkpoint(const std::string& path) const;
    bool load_checkpoint(const std::string& path);
    bool export_policy(const std::string& path) const;

private:
    void training_loop(std::stop_token stop_token);
    void collect_rollout(std::stop_token& stop_token);
    void train_on_rollout();
    void run_evaluation();

    TrainerConfig config_;
    std::shared_ptr<IPolicy> policy_;
    std::unique_ptr<IAlgorithm> algorithm_;
    std::unique_ptr<VectorizedEnv> envs_;
    std::unique_ptr<RolloutBuffer> buffer_;
    VectorizedEnv::EnvFactory env_factory_;
    std::vector<std::shared_ptr<ITrainerCallback>> callbacks_;

    std::jthread training_thread_;
    std::stop_source stop_source_;
    std::atomic<State> state_{State::Idle};
    std::atomic<bool> pause_requested_{false};
    std::atomic<int32_t> current_iteration_{0};
    std::atomic<int64_t> total_timesteps_{0};

    mutable std::mutex metrics_mutex_;
    TrainingMetrics latest_metrics_;

    mutable std::mutex policy_mutex_;

    // Episode tracking for metrics
    std::vector<float> episode_rewards_;
    std::vector<float> episode_lengths_;
    std::vector<float> current_ep_rewards_; // per-env running reward
};

} // namespace hne

#endif // HNE_TRAINING
