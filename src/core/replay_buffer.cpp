#ifdef HNE_TRAINING

#include <hne/core/replay_buffer.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace hne {

RolloutBuffer::RolloutBuffer(const Config& config) : config_(config) {
    continuous_ = std::holds_alternative<BoxSpace>(config_.action_space);
    if (continuous_) {
        auto& box = std::get<BoxSpace>(config_.action_space);
        act_dim_ = std::accumulate(box.shape.begin(), box.shape.end(),
                                    int32_t{1}, std::multiplies<>());
    } else {
        act_dim_ = 1;
    }

    int32_t total = config_.buffer_size * config_.num_envs;
    observations_.resize(total * config_.obs_size, 0.0f);
    actions_.resize(total * act_dim_, 0.0f);
    rewards_.resize(total, 0.0f);
    dones_.resize(total, 0.0f);
    values_.resize(total, 0.0f);
    log_probs_.resize(total, 0.0f);
    advantages_.resize(total, 0.0f);
    returns_.resize(total, 0.0f);
}

void RolloutBuffer::add(const Tensor& obs, const Action& action,
                         float reward, bool done, float value, float log_prob) {
    if (is_full()) {
        throw std::runtime_error("RolloutBuffer is full");
    }

    int32_t idx = pos_;

    // Observation
    std::copy(obs.data.begin(), obs.data.end(),
              observations_.begin() + idx * config_.obs_size);

    // Action
    if (continuous_) {
        const auto& vals = action.as_continuous();
        std::copy(vals.begin(), vals.end(),
                  actions_.begin() + idx * act_dim_);
    } else {
        actions_[idx] = static_cast<float>(action.as_discrete());
    }

    rewards_[idx] = reward;
    dones_[idx] = done ? 1.0f : 0.0f;
    values_[idx] = value;
    log_probs_[idx] = log_prob;

    pos_++;
}

void RolloutBuffer::compute_returns_and_advantages(const std::vector<float>& last_values) {
    int32_t total = pos_;
    int32_t num_envs = config_.num_envs;
    int32_t steps = total / num_envs;

    // GAE computation: traverse backwards
    // For a single-env buffer (num_envs=1), it's straightforward.
    // For multi-env, data is stored interleaved: [env0_step0, env0_step1, ..., env1_step0, ...]
    // Actually, we store sequentially: pos increments for each (env, step) pair.
    // With VectorizedEnv, the Trainer adds transitions one-at-a-time per env per step.
    // Layout: steps are grouped by timestep across envs.
    // Index: step * num_envs + env_idx

    for (int32_t env = 0; env < num_envs; env++) {
        float last_gae = 0.0f;
        float next_value = last_values[env];
        bool next_done = false;

        for (int32_t step = steps - 1; step >= 0; step--) {
            int32_t idx = step * num_envs + env;
            float delta = rewards_[idx]
                          + config_.gamma * next_value * (1.0f - dones_[idx])
                          - values_[idx];
            last_gae = delta
                       + config_.gamma * config_.gae_lambda * (1.0f - dones_[idx]) * last_gae;
            advantages_[idx] = last_gae;
            returns_[idx] = advantages_[idx] + values_[idx];

            next_value = values_[idx];
            next_done = (dones_[idx] > 0.5f);
        }
    }
}

std::vector<RolloutSample> RolloutBuffer::get_batches(int32_t mini_batch_size) const {
    int32_t total = pos_;

    // Convert flat vectors to tensors
    auto obs_t = torch::from_blob(
        const_cast<float*>(observations_.data()),
        {total, config_.obs_size}, torch::kFloat32).clone();

    torch::Tensor act_t;
    if (continuous_) {
        act_t = torch::from_blob(
            const_cast<float*>(actions_.data()),
            {total, act_dim_}, torch::kFloat32).clone();
    } else {
        // Discrete actions stored as float, convert to long for indexing
        act_t = torch::from_blob(
            const_cast<float*>(actions_.data()),
            {total}, torch::kFloat32).clone().to(torch::kLong);
    }

    auto lp_t = torch::from_blob(
        const_cast<float*>(log_probs_.data()),
        {total}, torch::kFloat32).clone();
    auto ret_t = torch::from_blob(
        const_cast<float*>(returns_.data()),
        {total}, torch::kFloat32).clone();
    auto adv_t = torch::from_blob(
        const_cast<float*>(advantages_.data()),
        {total}, torch::kFloat32).clone();
    auto val_t = torch::from_blob(
        const_cast<float*>(values_.data()),
        {total}, torch::kFloat32).clone();

    // Shuffle indices
    auto indices = torch::randperm(total, torch::kLong);

    // Split into mini-batches
    std::vector<RolloutSample> batches;
    for (int32_t start = 0; start < total; start += mini_batch_size) {
        int32_t end = std::min(start + mini_batch_size, total);
        auto idx = indices.slice(0, start, end);

        batches.push_back(RolloutSample{
            .observations = obs_t.index_select(0, idx),
            .actions = act_t.index_select(0, idx),
            .log_probs = lp_t.index_select(0, idx),
            .returns = ret_t.index_select(0, idx),
            .advantages = adv_t.index_select(0, idx),
            .values = val_t.index_select(0, idx),
        });
    }

    return batches;
}

void RolloutBuffer::reset() {
    pos_ = 0;
}

bool RolloutBuffer::is_full() const {
    return pos_ >= config_.buffer_size * config_.num_envs;
}

int32_t RolloutBuffer::size() const {
    return pos_;
}

int32_t RolloutBuffer::capacity() const {
    return config_.buffer_size * config_.num_envs;
}

} // namespace hne

#endif // HNE_TRAINING
