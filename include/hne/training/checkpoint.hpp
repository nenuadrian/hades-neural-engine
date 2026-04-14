#pragma once

#ifdef HNE_TRAINING

#include "../core/policy.hpp"
#include "trainer_config.hpp"
#include <string>
#include <torch/torch.h>

namespace hne {

struct CheckpointData {
    int32_t iteration = 0;
    int64_t total_timesteps = 0;
    TrainerConfig config;
};

namespace checkpoint {

bool save(const std::string& path,
          const IPolicy& policy,
          const torch::optim::Adam& optimizer,
          const CheckpointData& data);

bool load(const std::string& path,
          IPolicy& policy,
          torch::optim::Adam& optimizer,
          CheckpointData& data);

bool export_policy(const std::string& path,
                   IPolicy& policy,
                   const SpaceSpec& obs_space,
                   const SpaceSpec& act_space,
                   int32_t obs_size);

} // namespace checkpoint

} // namespace hne

#endif // HNE_TRAINING
