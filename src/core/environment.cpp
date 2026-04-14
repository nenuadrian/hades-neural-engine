#include <hne/core/environment.hpp>

namespace hne {

VectorizedEnv::VectorizedEnv(EnvFactory factory, int32_t num_envs) {
    envs_.reserve(num_envs);
    for (int32_t i = 0; i < num_envs; i++) {
        envs_.push_back(factory());
    }
    obs_space_ = envs_[0]->observation_space();
    act_space_ = envs_[0]->action_space();
}

VectorizedEnv::~VectorizedEnv() = default;

int32_t VectorizedEnv::num_envs() const {
    return static_cast<int32_t>(envs_.size());
}

SpaceSpec VectorizedEnv::observation_space() const { return obs_space_; }
SpaceSpec VectorizedEnv::action_space() const { return act_space_; }

std::vector<Tensor> VectorizedEnv::reset_all(int32_t seed) {
    std::vector<Tensor> observations;
    observations.reserve(envs_.size());

    for (int32_t i = 0; i < static_cast<int32_t>(envs_.size()); i++) {
        observations.push_back(envs_[i]->reset(seed >= 0 ? seed + i : -1));
    }

    return observations;
}

std::vector<StepResult> VectorizedEnv::step_all(const std::vector<Action>& actions) {
    std::vector<StepResult> results;
    results.reserve(envs_.size());

    for (size_t i = 0; i < envs_.size(); i++) {
        auto result = envs_[i]->step(actions[i]);
        if (result.terminated || result.truncated) {
            result.observation = envs_[i]->reset();
        }
        results.push_back(std::move(result));
    }

    return results;
}

} // namespace hne
