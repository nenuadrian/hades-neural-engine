#include <hne/algorithms/ppo.hpp>

namespace hne {

// ── PPOConfig JSON ─────────────────────────────────────────────────────────

void to_json(nlohmann::json& j, const PPOConfig& c) {
    j = {
        {"clip_epsilon", c.clip_epsilon},
        {"value_loss_coeff", c.value_loss_coeff},
        {"entropy_coeff", c.entropy_coeff},
        {"max_grad_norm", c.max_grad_norm},
        {"learning_rate", c.learning_rate},
        {"num_epochs", c.num_epochs},
        {"mini_batch_size", c.mini_batch_size},
        {"normalize_advantages", c.normalize_advantages},
        {"target_kl", c.target_kl},
    };
}

void from_json(const nlohmann::json& j, PPOConfig& c) {
    if (j.contains("clip_epsilon")) j.at("clip_epsilon").get_to(c.clip_epsilon);
    if (j.contains("value_loss_coeff")) j.at("value_loss_coeff").get_to(c.value_loss_coeff);
    if (j.contains("entropy_coeff")) j.at("entropy_coeff").get_to(c.entropy_coeff);
    if (j.contains("max_grad_norm")) j.at("max_grad_norm").get_to(c.max_grad_norm);
    if (j.contains("learning_rate")) j.at("learning_rate").get_to(c.learning_rate);
    if (j.contains("num_epochs")) j.at("num_epochs").get_to(c.num_epochs);
    if (j.contains("mini_batch_size")) j.at("mini_batch_size").get_to(c.mini_batch_size);
    if (j.contains("normalize_advantages")) j.at("normalize_advantages").get_to(c.normalize_advantages);
    if (j.contains("target_kl")) j.at("target_kl").get_to(c.target_kl);
}

} // namespace hne

#ifdef HNE_TRAINING

#include <cmath>
#include <stdexcept>

namespace hne {

PPO::PPO(const PPOConfig& config) : config_(config) {}

void PPO::ensure_optimizer(IPolicy& policy) {
    if (!optimizer_) {
        optimizer_ = std::make_unique<torch::optim::Adam>(
            policy.parameters(),
            torch::optim::AdamOptions(config_.learning_rate)
        );
    }
}

AlgorithmMetrics PPO::update(IPolicy& policy, const RolloutBuffer& buffer) {
    ensure_optimizer(policy);

    float total_policy_loss = 0.0f;
    float total_value_loss = 0.0f;
    float total_entropy = 0.0f;
    float total_approx_kl = 0.0f;
    float total_clip_fraction = 0.0f;
    int32_t n_updates = 0;
    bool early_stopped = false;

    for (int32_t epoch = 0; epoch < config_.num_epochs; epoch++) {
        auto batches = buffer.get_batches(config_.mini_batch_size);

        for (auto& batch : batches) {
            auto out = policy.forward(batch.observations);

            // Value estimate
            auto values = out.value.squeeze(-1);

            // Compute action log probabilities and entropy
            torch::Tensor new_log_probs;
            torch::Tensor entropy;

            // Check if action_logits represents discrete or continuous
            if (batch.actions.dtype() == torch::kLong) {
                // Discrete: action_logits are logits for categorical distribution
                auto log_probs_all = torch::log_softmax(out.action_logits, /*dim=*/1);
                new_log_probs = log_probs_all.gather(1, batch.actions.unsqueeze(1)).squeeze(1);

                auto probs = torch::softmax(out.action_logits, 1);
                entropy = -(probs * log_probs_all).sum(1);
            } else {
                // Continuous: action_logits are means, log_std is learnable
                auto mean = out.action_logits;
                auto log_std = out.log_std;
                auto std = log_std.exp();

                // Log probability of taken actions under Gaussian
                auto diff = batch.actions - mean;
                new_log_probs = (-0.5f * (diff / std).pow(2)
                                 - log_std
                                 - 0.5f * std::log(2.0f * M_PI)).sum(1);

                // Entropy of Gaussian: 0.5 * ln(2*pi*e*sigma^2) per dimension
                entropy = (log_std + 0.5f * std::log(2.0f * M_PI * M_E)).sum(1);
            }

            // Advantage normalization
            auto advantages = batch.advantages;
            if (config_.normalize_advantages && advantages.numel() > 1) {
                advantages = (advantages - advantages.mean()) /
                             (advantages.std() + 1e-8f);
            }

            // PPO clipped surrogate loss
            auto ratio = (new_log_probs - batch.log_probs).exp();
            auto surr1 = ratio * advantages;
            auto surr2 = torch::clamp(ratio, 1.0f - config_.clip_epsilon,
                                       1.0f + config_.clip_epsilon) * advantages;
            auto policy_loss = -torch::min(surr1, surr2).mean();

            // Value function loss
            auto value_loss = torch::mse_loss(values, batch.returns);

            // Entropy bonus
            auto entropy_loss = -entropy.mean();

            // Total loss
            auto loss = policy_loss
                        + config_.value_loss_coeff * value_loss
                        + config_.entropy_coeff * entropy_loss;

            // Gradient step
            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(policy.parameters(), config_.max_grad_norm);
            optimizer_->step();

            // Track metrics
            total_policy_loss += policy_loss.item<float>();
            total_value_loss += value_loss.item<float>();
            total_entropy += -entropy_loss.item<float>();
            n_updates++;

            // Approximate KL divergence
            auto approx_kl = ((ratio - 1.0f) - ratio.log()).mean().item<float>();
            total_approx_kl += approx_kl;

            // Clip fraction
            auto clip_frac = ((ratio - 1.0f).abs() > config_.clip_epsilon)
                             .to(torch::kFloat32).mean().item<float>();
            total_clip_fraction += clip_frac;
        }

        // Early stopping on KL divergence
        if (config_.target_kl > 0.0f && n_updates > 0) {
            float mean_kl = total_approx_kl / n_updates;
            if (mean_kl > config_.target_kl) {
                early_stopped = true;
                break;
            }
        }
    }

    float inv_n = (n_updates > 0) ? 1.0f / n_updates : 0.0f;
    return AlgorithmMetrics{
        .scalars = {
            {"policy_loss", total_policy_loss * inv_n},
            {"value_loss", total_value_loss * inv_n},
            {"entropy", total_entropy * inv_n},
            {"approx_kl", total_approx_kl * inv_n},
            {"clip_fraction", total_clip_fraction * inv_n},
            {"early_stopped", early_stopped ? 1.0f : 0.0f},
            {"n_updates", static_cast<float>(n_updates)},
        }
    };
}

} // namespace hne

#endif // HNE_TRAINING
