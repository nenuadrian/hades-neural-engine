#pragma once

#include "../core/agent.hpp"
#include "../core/types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace hne {

class InferenceRuntime {
public:
    InferenceRuntime();
    ~InferenceRuntime();

    InferenceRuntime(const InferenceRuntime&) = delete;
    InferenceRuntime& operator=(const InferenceRuntime&) = delete;

    bool load(const std::string& path);

    [[nodiscard]] bool is_loaded() const;

    [[nodiscard]] Action evaluate(const Tensor& observation,
                                  bool deterministic = true);

    [[nodiscard]] std::vector<Action> evaluate_batch(
        const std::vector<Tensor>& observations,
        bool deterministic = true);

    [[nodiscard]] SpaceSpec observation_space() const;
    [[nodiscard]] SpaceSpec action_space() const;

    [[nodiscard]] std::unique_ptr<IAgent> create_agent(uint32_t agent_id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hne
