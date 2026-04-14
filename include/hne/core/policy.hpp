#pragma once

#ifdef HNE_TRAINING

#include "types.hpp"
#include <torch/torch.h>
#include <vector>

namespace hne {

struct PolicyOutput {
    torch::Tensor action_logits;  // Discrete: [batch, n] | Continuous: mean [batch, dim]
    torch::Tensor value;          // [batch, 1]
    torch::Tensor log_std;        // Continuous only: [batch, dim] (unused for discrete)
};

class IPolicy : public torch::nn::Module {
public:
    ~IPolicy() override = default;
    virtual PolicyOutput forward(torch::Tensor observation) = 0;
};

// Default MLP actor-critic with shared trunk.
class MLPPolicy : public IPolicy {
public:
    struct Config {
        int32_t obs_size = 0;
        SpaceSpec action_space;
        std::vector<int32_t> hidden_sizes = {64, 64};
        float init_log_std = 0.0f;
    };

    explicit MLPPolicy(const Config& config);

    PolicyOutput forward(torch::Tensor observation) override;

private:
    torch::nn::Sequential shared_{nullptr};
    torch::nn::Linear actor_head_{nullptr};
    torch::nn::Linear critic_head_{nullptr};
    torch::Tensor log_std_;
    SpaceSpec action_space_;
    bool continuous_ = false;
};

} // namespace hne

#endif // HNE_TRAINING
