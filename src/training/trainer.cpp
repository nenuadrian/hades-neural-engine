#ifdef HNE_TRAINING

#include <hne/training/trainer.hpp>
#include <hne/training/checkpoint.hpp>
#include <hne/algorithms/ppo.hpp>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>

namespace hne {

// ── MLPPolicy implementation (lives here, linked into hne_training) ────────

MLPPolicy::MLPPolicy(const Config& config)
    : action_space_(config.action_space) {

    continuous_ = std::holds_alternative<BoxSpace>(config.action_space);

    // Build shared trunk
    shared_ = torch::nn::Sequential();
    int32_t in_size = config.obs_size;
    for (size_t i = 0; i < config.hidden_sizes.size(); i++) {
        shared_->push_back(torch::nn::Linear(in_size, config.hidden_sizes[i]));
        shared_->push_back(torch::nn::Tanh());
        in_size = config.hidden_sizes[i];
    }
    register_module("shared", shared_);

    // Actor head
    int32_t act_size;
    if (continuous_) {
        auto& box = std::get<BoxSpace>(config.action_space);
        act_size = std::accumulate(box.shape.begin(), box.shape.end(),
                                    int32_t{1}, std::multiplies<>());
    } else if (std::holds_alternative<DiscreteSpace>(config.action_space)) {
        act_size = std::get<DiscreteSpace>(config.action_space).n;
    } else {
        auto& mds = std::get<MultiDiscreteSpace>(config.action_space);
        act_size = std::accumulate(mds.nvec.begin(), mds.nvec.end(), int32_t{0});
    }

    actor_head_ = torch::nn::Linear(in_size, act_size);
    register_module("actor_head", actor_head_);

    // Critic head (scalar value)
    critic_head_ = torch::nn::Linear(in_size, 1);
    register_module("critic_head", critic_head_);

    // Log std for continuous actions (learnable parameter)
    if (continuous_) {
        log_std_ = register_parameter("log_std",
            torch::full({act_size}, config.init_log_std));
    }
}

PolicyOutput MLPPolicy::forward(torch::Tensor observation) {
    auto features = shared_->forward(observation);
    auto action_out = actor_head_->forward(features);
    auto value = critic_head_->forward(features);

    PolicyOutput out;
    out.action_logits = action_out;
    out.value = value;
    if (continuous_) {
        out.log_std = log_std_.expand_as(action_out);
    }
    return out;
}

// ── Trainer implementation ─────────────────────────────────────────────────

Trainer::Trainer(const TrainerConfig& config) : config_(config) {}

Trainer::~Trainer() {
    request_stop();
    if (training_thread_.joinable()) {
        training_thread_.join();
    }
}

void Trainer::set_environment_factory(VectorizedEnv::EnvFactory factory) {
    env_factory_ = std::move(factory);
}

void Trainer::set_policy(std::shared_ptr<IPolicy> policy) {
    policy_ = std::move(policy);
}

void Trainer::set_algorithm(std::unique_ptr<IAlgorithm> algorithm) {
    algorithm_ = std::move(algorithm);
}

void Trainer::add_callback(std::shared_ptr<ITrainerCallback> callback) {
    callbacks_.push_back(std::move(callback));
}

void Trainer::train_async() {
    if (state_ != State::Idle && state_ != State::Finished && state_ != State::Error) {
        return;
    }
    stop_requested_ = false;
    if (training_thread_.joinable()) {
        training_thread_.join();
    }
    training_thread_ = std::thread([this]() {
        training_loop();
    });
}

void Trainer::train() {
    stop_requested_ = false;
    training_loop();
}

void Trainer::request_stop() {
    stop_requested_ = true;
}

void Trainer::pause() {
    pause_requested_ = true;
    state_ = State::Paused;
}

void Trainer::resume() {
    pause_requested_ = false;
    state_ = State::Running;
}

Trainer::State Trainer::state() const { return state_; }
int32_t Trainer::current_iteration() const { return current_iteration_; }
int64_t Trainer::total_timesteps() const { return total_timesteps_; }

TrainingMetrics Trainer::latest_metrics() const {
    std::lock_guard lock(metrics_mutex_);
    return latest_metrics_;
}

void Trainer::training_loop() {
    try {
        state_ = State::Running;

        // Create environments
        envs_ = std::make_unique<VectorizedEnv>(env_factory_, config_.num_envs);
        auto obs_space = envs_->observation_space();
        auto act_space = envs_->action_space();
        int32_t obs_size = flat_size(obs_space);

        // Create default policy if not set
        if (!policy_) {
            policy_ = std::make_shared<MLPPolicy>(MLPPolicy::Config{
                .obs_size = obs_size,
                .action_space = act_space,
                .hidden_sizes = config_.hidden_sizes,
            });
        }

        // Create default algorithm if not set
        if (!algorithm_) {
            algorithm_ = std::make_unique<PPO>(config_.ppo);
        }

        // Create rollout buffer
        buffer_ = std::make_unique<RolloutBuffer>(RolloutBuffer::Config{
            .buffer_size = config_.rollout_length,
            .num_envs = config_.num_envs,
            .obs_size = obs_size,
            .action_space = act_space,
            .gamma = config_.gamma,
            .gae_lambda = config_.gae_lambda,
        });

        // Initialize episode tracking
        current_ep_rewards_.resize(config_.num_envs, 0.0f);

        // Notify callbacks
        for (auto& cb : callbacks_) cb->on_training_start(*this);

        // Reset all environments
        auto observations = envs_->reset_all(config_.seed);

        // Training loop
        while (!stop_requested_ &&
               total_timesteps_ < config_.total_timesteps) {

            // Pause handling
            while (pause_requested_ && !stop_requested_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (stop_requested_) break;

            for (auto& cb : callbacks_) cb->on_rollout_start(*this);
            buffer_->reset();

            // Collect rollout
            for (int32_t step = 0; step < config_.rollout_length; step++) {
                if (stop_requested_) break;

                // Stack observations into a batch tensor
                std::vector<float> obs_batch;
                obs_batch.reserve(config_.num_envs * obs_size);
                for (auto& obs : observations) {
                    obs_batch.insert(obs_batch.end(), obs.data.begin(), obs.data.end());
                }

                torch::Tensor obs_tensor;
                {
                    std::lock_guard lock(policy_mutex_);
                    obs_tensor = torch::from_blob(
                        obs_batch.data(),
                        {config_.num_envs, obs_size},
                        torch::kFloat32
                    ).clone();
                }

                // Forward pass (no grad)
                torch::Tensor action_tensor, log_prob_tensor, value_tensor;
                {
                    torch::NoGradGuard no_grad;
                    std::lock_guard lock(policy_mutex_);
                    auto out = policy_->forward(obs_tensor);
                    value_tensor = out.value.squeeze(-1);

                    bool continuous = std::holds_alternative<BoxSpace>(act_space);
                    if (continuous) {
                        auto mean = out.action_logits;
                        auto log_std = out.log_std;
                        auto std_val = log_std.exp();
                        auto noise = torch::randn_like(mean);
                        action_tensor = mean + std_val * noise;

                        auto diff = action_tensor - mean;
                        log_prob_tensor = (-0.5f * (diff / std_val).pow(2)
                                           - log_std
                                           - 0.5f * std::log(2.0f * M_PI)).sum(1);
                    } else {
                        auto probs = torch::softmax(out.action_logits, 1);
                        action_tensor = probs.multinomial(1).squeeze(1);
                        auto log_probs_all = torch::log_softmax(out.action_logits, 1);
                        log_prob_tensor = log_probs_all.gather(
                            1, action_tensor.unsqueeze(1)).squeeze(1);
                    }
                }

                // Convert actions to hne::Action and step environments
                std::vector<Action> actions;
                bool continuous = std::holds_alternative<BoxSpace>(act_space);
                for (int32_t e = 0; e < config_.num_envs; e++) {
                    if (continuous) {
                        int32_t act_dim = action_tensor.size(1);
                        std::vector<float> act_vec(act_dim);
                        for (int32_t d = 0; d < act_dim; d++) {
                            act_vec[d] = action_tensor[e][d].item<float>();
                        }
                        actions.push_back(Action::continuous(std::move(act_vec)));
                    } else {
                        actions.push_back(Action::discrete(
                            action_tensor[e].item<int32_t>()));
                    }
                }

                auto results = envs_->step_all(actions);

                // Store transitions
                for (int32_t e = 0; e < config_.num_envs; e++) {
                    buffer_->add(
                        observations[e],
                        actions[e],
                        results[e].reward,
                        results[e].terminated || results[e].truncated,
                        value_tensor[e].item<float>(),
                        log_prob_tensor[e].item<float>()
                    );

                    // Episode tracking
                    current_ep_rewards_[e] += results[e].reward;
                    if (results[e].terminated || results[e].truncated) {
                        episode_rewards_.push_back(current_ep_rewards_[e]);
                        episode_lengths_.push_back(
                            static_cast<float>(step + 1)); // approximate
                        current_ep_rewards_[e] = 0.0f;
                    }

                    // Auto-reset already handled by VectorizedEnv
                    observations[e] = results[e].observation;
                }

                total_timesteps_ += config_.num_envs;
                for (auto& cb : callbacks_) cb->on_step(*this, total_timesteps_);
            }

            if (stop_requested_) break;

            // Compute last values for GAE bootstrap
            std::vector<float> last_values(config_.num_envs);
            {
                std::vector<float> obs_batch;
                obs_batch.reserve(config_.num_envs * obs_size);
                for (auto& obs : observations) {
                    obs_batch.insert(obs_batch.end(), obs.data.begin(), obs.data.end());
                }

                torch::NoGradGuard no_grad;
                std::lock_guard lock(policy_mutex_);
                auto obs_t = torch::from_blob(
                    obs_batch.data(),
                    {config_.num_envs, obs_size},
                    torch::kFloat32
                ).clone();
                auto out = policy_->forward(obs_t);
                for (int32_t e = 0; e < config_.num_envs; e++) {
                    last_values[e] = out.value[e][0].item<float>();
                }
            }

            buffer_->compute_returns_and_advantages(last_values);

            for (auto& cb : callbacks_) cb->on_rollout_end(*this);

            // PPO update
            AlgorithmMetrics algo_metrics;
            {
                std::lock_guard lock(policy_mutex_);
                algo_metrics = algorithm_->update(*policy_, *buffer_);
            }

            current_iteration_++;

            // Update metrics
            {
                std::lock_guard lock(metrics_mutex_);
                latest_metrics_.iteration = current_iteration_;
                latest_metrics_.total_timesteps = total_timesteps_;
                latest_metrics_.algorithm_metrics = algo_metrics.scalars;

                if (!episode_rewards_.empty()) {
                    latest_metrics_.mean_episode_reward =
                        std::accumulate(episode_rewards_.begin(),
                                        episode_rewards_.end(), 0.0f) /
                        episode_rewards_.size();
                    latest_metrics_.mean_episode_length =
                        std::accumulate(episode_lengths_.begin(),
                                        episode_lengths_.end(), 0.0f) /
                        episode_lengths_.size();
                    episode_rewards_.clear();
                    episode_lengths_.clear();
                }
            }

            for (auto& cb : callbacks_) cb->on_update(*this, latest_metrics_);

            // Evaluation
            if (config_.eval_interval > 0 &&
                current_iteration_ % config_.eval_interval == 0) {
                run_evaluation();
            }

            // Checkpointing
            if (config_.checkpoint_interval > 0 &&
                current_iteration_ % config_.checkpoint_interval == 0) {
                std::filesystem::create_directories(config_.checkpoint_dir);
                auto path = std::format("{}/checkpoint_{}.pt",
                                        config_.checkpoint_dir, current_iteration_.load());
                save_checkpoint(path);
                for (auto& cb : callbacks_) cb->on_checkpoint(*this, path);
            }
        }

        for (auto& cb : callbacks_) cb->on_training_end(*this);
        state_ = State::Finished;

    } catch (const std::exception& e) {
        std::cerr << "[HNE] Training error: " << e.what() << std::endl;
        state_ = State::Error;
    }
}

void Trainer::run_evaluation() {
    if (!env_factory_) return;

    float total_reward = 0.0f;
    float total_length = 0.0f;

    for (int32_t ep = 0; ep < config_.eval_episodes; ep++) {
        auto eval_env = env_factory_();
        auto result = evaluate(*eval_env, true);
        total_reward += result.total_reward;
        total_length += result.episode_length;
    }

    float mean_reward = total_reward / config_.eval_episodes;
    float mean_length = total_length / config_.eval_episodes;

    for (auto& cb : callbacks_) {
        cb->on_evaluation(*this, mean_reward, mean_length);
    }
}

Trainer::EvalResult Trainer::evaluate(IEnvironment& eval_env, bool deterministic,
                                       int32_t max_steps) {
    auto obs = eval_env.reset();
    int32_t obs_size = obs.numel();
    float total_reward = 0.0f;
    int32_t steps = 0;

    for (int32_t step = 0; step < max_steps; step++) {
        torch::Tensor obs_t;
        torch::Tensor action_t;
        bool continuous = std::holds_alternative<BoxSpace>(eval_env.action_space());

        {
            torch::NoGradGuard no_grad;
            std::lock_guard lock(policy_mutex_);
            obs_t = torch::from_blob(obs.data.data(), {1, obs_size},
                                      torch::kFloat32).clone();
            auto out = policy_->forward(obs_t);

            if (continuous) {
                if (deterministic) {
                    action_t = out.action_logits;
                } else {
                    auto std_val = out.log_std.exp();
                    action_t = out.action_logits + std_val * torch::randn_like(out.action_logits);
                }
            } else {
                if (deterministic) {
                    action_t = out.action_logits.argmax(1);
                } else {
                    auto probs = torch::softmax(out.action_logits, 1);
                    action_t = probs.multinomial(1).squeeze(1);
                }
            }
        }

        Action action;
        if (continuous) {
            int32_t act_dim = action_t.size(1);
            std::vector<float> act_vec(act_dim);
            for (int32_t d = 0; d < act_dim; d++) {
                act_vec[d] = action_t[0][d].item<float>();
            }
            action = Action::continuous(std::move(act_vec));
        } else {
            action = Action::discrete(action_t[0].item<int32_t>());
        }

        auto result = eval_env.step(action);
        total_reward += result.reward;
        obs = result.observation;
        steps++;

        if (result.terminated || result.truncated) break;
    }

    return EvalResult{
        .total_reward = total_reward,
        .episode_length = steps,
    };
}

bool Trainer::save_checkpoint(const std::string& path) const {
    if (!policy_ || !algorithm_) return false;
    auto* ppo = dynamic_cast<PPO*>(algorithm_.get());
    if (!ppo || !ppo->optimizer()) return false;

    std::lock_guard lock(policy_mutex_);
    return checkpoint::save(path, *policy_, *ppo->optimizer(),
        CheckpointData{
            .iteration = current_iteration_.load(),
            .total_timesteps = total_timesteps_.load(),
            .config = config_,
        });
}

bool Trainer::load_checkpoint(const std::string& path) {
    if (!policy_ || !algorithm_) return false;
    auto* ppo = dynamic_cast<PPO*>(algorithm_.get());
    if (!ppo || !ppo->optimizer()) return false;

    CheckpointData data;
    if (!checkpoint::load(path, *policy_, *ppo->optimizer(), data)) return false;

    current_iteration_ = data.iteration;
    total_timesteps_ = data.total_timesteps;
    config_ = data.config;
    return true;
}

bool Trainer::export_policy(const std::string& path) const {
    if (!policy_ || !envs_) return false;
    std::lock_guard lock(policy_mutex_);
    return checkpoint::export_policy(
        path, *policy_,
        envs_->observation_space(),
        envs_->action_space(),
        flat_size(envs_->observation_space()));
}

} // namespace hne

#endif // HNE_TRAINING
