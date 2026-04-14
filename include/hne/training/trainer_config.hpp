#pragma once

#include "../algorithms/ppo.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace hne {

struct TrainerConfig {
    // Environment
    int32_t num_envs = 8;

    // Rollout
    int32_t rollout_length = 2048;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;

    // Training budget
    int64_t total_timesteps = 1000000;
    int32_t eval_interval = 10;
    int32_t eval_episodes = 5;
    int32_t checkpoint_interval = 50;
    std::string checkpoint_dir = "checkpoints";

    // Policy network
    std::vector<int32_t> hidden_sizes = {64, 64};

    // PPO hyperparameters
    PPOConfig ppo;

    // Seed
    int32_t seed = 42;
};

void to_json(nlohmann::json& j, const TrainerConfig& c);
void from_json(const nlohmann::json& j, TrainerConfig& c);

TrainerConfig load_trainer_config(const std::string& path);
void save_trainer_config(const TrainerConfig& config, const std::string& path);

} // namespace hne
