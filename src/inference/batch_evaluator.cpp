#include <hne/inference/batch_evaluator.hpp>
#include <stdexcept>

namespace hne {

BatchEvaluator::BatchEvaluator(InferenceRuntime& runtime)
    : runtime_(runtime) {}

void BatchEvaluator::submit(uint32_t agent_id, Tensor observation) {
    agent_ids_.push_back(agent_id);
    observations_.push_back(std::move(observation));
}

void BatchEvaluator::evaluate(bool deterministic) {
    if (observations_.empty()) return;

    auto actions = runtime_.evaluate_batch(observations_, deterministic);

    results_.clear();
    for (size_t i = 0; i < agent_ids_.size(); i++) {
        results_[agent_ids_[i]] = std::move(actions[i]);
    }

    agent_ids_.clear();
    observations_.clear();
}

Action BatchEvaluator::get_action(uint32_t agent_id) const {
    auto it = results_.find(agent_id);
    if (it == results_.end()) {
        throw std::runtime_error("No action for agent " + std::to_string(agent_id));
    }
    return it->second;
}

void BatchEvaluator::clear() {
    agent_ids_.clear();
    observations_.clear();
    results_.clear();
}

int32_t BatchEvaluator::pending_count() const {
    return static_cast<int32_t>(observations_.size());
}

} // namespace hne
