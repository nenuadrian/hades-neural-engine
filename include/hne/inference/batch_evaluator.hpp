#pragma once

#include "runtime.hpp"
#include "../core/types.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hne {

class BatchEvaluator {
public:
    explicit BatchEvaluator(InferenceRuntime& runtime);

    void submit(uint32_t agent_id, Tensor observation);

    void evaluate(bool deterministic = true);

    [[nodiscard]] Action get_action(uint32_t agent_id) const;

    void clear();

    [[nodiscard]] int32_t pending_count() const;

private:
    InferenceRuntime& runtime_;
    std::vector<uint32_t> agent_ids_;
    std::vector<Tensor> observations_;
    std::unordered_map<uint32_t, Action> results_;
};

} // namespace hne
