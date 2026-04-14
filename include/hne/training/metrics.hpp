#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace hne {

struct TrainingMetrics {
    int32_t iteration = 0;
    int64_t total_timesteps = 0;
    float mean_episode_reward = 0.0f;
    float mean_episode_length = 0.0f;
    std::map<std::string, float> algorithm_metrics; // policy_loss, value_loss, entropy, etc.
};

} // namespace hne
