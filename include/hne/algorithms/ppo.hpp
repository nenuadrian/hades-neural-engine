#pragma once

#include <nlohmann/json.hpp>
#include <cstdint>

namespace hne {

struct PPOConfig {
    float clip_epsilon = 0.2f;
    float value_loss_coeff = 0.5f;
    float entropy_coeff = 0.01f;
    float max_grad_norm = 0.5f;
    float learning_rate = 3e-4f;
    int32_t num_epochs = 10;
    int32_t mini_batch_size = 64;
    bool normalize_advantages = true;
    float target_kl = -1.0f; // Disabled by default
};

void to_json(nlohmann::json& j, const PPOConfig& c);
void from_json(const nlohmann::json& j, PPOConfig& c);

} // namespace hne

#ifdef HNE_TRAINING

#include "../core/algorithm.hpp"
#include "../core/replay_buffer.hpp"
#include <memory>
#include <torch/torch.h>

namespace hne {

class PPO : public IAlgorithm {
public:
    explicit PPO(const PPOConfig& config = {});

    AlgorithmMetrics update(IPolicy& policy, const RolloutBuffer& buffer) override;
    [[nodiscard]] std::string name() const override { return "PPO"; }

    [[nodiscard]] const PPOConfig& config() const { return config_; }

    torch::optim::Adam* optimizer() { return optimizer_.get(); }

private:
    PPOConfig config_;
    std::unique_ptr<torch::optim::Adam> optimizer_;

    void ensure_optimizer(IPolicy& policy);
};

} // namespace hne

#endif // HNE_TRAINING
