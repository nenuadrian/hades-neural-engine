#pragma once

#ifdef HNE_TRAINING

#include "types.hpp"
#include <torch/torch.h>
#include <vector>

namespace hne {

struct RolloutSample {
    torch::Tensor observations;  // [batch, obs_size]
    torch::Tensor actions;       // [batch] (discrete) or [batch, act_dim] (continuous)
    torch::Tensor log_probs;     // [batch]
    torch::Tensor returns;       // [batch]
    torch::Tensor advantages;    // [batch]
    torch::Tensor values;        // [batch]
};

class RolloutBuffer {
public:
    struct Config {
        int32_t buffer_size = 2048;
        int32_t num_envs = 1;
        int32_t obs_size = 0;
        SpaceSpec action_space;
        float gamma = 0.99f;
        float gae_lambda = 0.95f;
    };

    explicit RolloutBuffer(const Config& config);

    void add(const Tensor& obs, const Action& action,
             float reward, bool done, float value, float log_prob);

    void compute_returns_and_advantages(const std::vector<float>& last_values);

    [[nodiscard]] std::vector<RolloutSample> get_batches(int32_t mini_batch_size) const;

    void reset();

    [[nodiscard]] bool is_full() const;
    [[nodiscard]] int32_t size() const;
    [[nodiscard]] int32_t capacity() const;

private:
    Config config_;
    int32_t pos_ = 0;
    int32_t act_dim_ = 1;
    bool continuous_ = false;

    std::vector<float> observations_;
    std::vector<float> actions_;
    std::vector<float> rewards_;
    std::vector<float> dones_;
    std::vector<float> values_;
    std::vector<float> log_probs_;
    std::vector<float> advantages_;
    std::vector<float> returns_;
};

} // namespace hne

#endif // HNE_TRAINING
