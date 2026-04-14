#include <hne/inference/runtime.hpp>

#include <torch/script.h>
#include <iostream>
#include <stdexcept>

namespace hne {

struct InferenceRuntime::Impl {
    torch::jit::script::Module model;
    bool loaded = false;
    SpaceSpec obs_space;
    SpaceSpec act_space;
    int32_t obs_size = 0;
    bool continuous = false;
};

InferenceRuntime::InferenceRuntime() : impl_(std::make_unique<Impl>()) {}
InferenceRuntime::~InferenceRuntime() = default;

bool InferenceRuntime::load(const std::string& path) {
    try {
        impl_->model = torch::jit::load(path);
        impl_->model.eval();

        // Try to read embedded metadata
        if (impl_->model.hasattr("hne_meta.json")) {
            // Extra files from save()
        }

        // Read extra files
        std::unordered_map<std::string, std::string> extra;
        extra["hne_meta.json"] = "";

        // Re-load with extra files
        impl_->model = torch::jit::load(path, torch::kCPU, extra);
        impl_->model.eval();

        auto& meta_str = extra["hne_meta.json"];
        if (!meta_str.empty()) {
            auto meta = nlohmann::json::parse(meta_str);
            from_json(meta["obs_space"], impl_->obs_space);
            from_json(meta["act_space"], impl_->act_space);
            impl_->obs_size = meta.value("obs_size", 0);
        }

        impl_->continuous = std::holds_alternative<BoxSpace>(impl_->act_space);
        impl_->loaded = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[HNE] Failed to load policy: " << e.what() << std::endl;
        impl_->loaded = false;
        return false;
    }
}

bool InferenceRuntime::is_loaded() const {
    return impl_->loaded;
}

Action InferenceRuntime::evaluate(const Tensor& observation, bool deterministic) {
    if (!impl_->loaded) {
        throw std::runtime_error("No policy loaded");
    }

    torch::NoGradGuard no_grad;

    auto obs_t = torch::from_blob(
        const_cast<float*>(observation.data.data()),
        {1, static_cast<int64_t>(observation.data.size())},
        torch::kFloat32
    ).clone();

    auto output = impl_->model.forward({obs_t});

    // The traced model returns a tuple: (action_logits, value) or
    // (action_logits, value, log_std) depending on the policy
    torch::Tensor action_logits;
    if (output.isTuple()) {
        auto elements = output.toTuple()->elements();
        action_logits = elements[0].toTensor();
    } else {
        action_logits = output.toTensor();
    }

    if (impl_->continuous) {
        torch::Tensor action;
        if (deterministic) {
            action = action_logits; // mean
        } else {
            // We don't have log_std from traced model easily,
            // so default to deterministic for inference
            action = action_logits;
        }
        int32_t act_dim = action.size(1);
        std::vector<float> act_vec(act_dim);
        for (int32_t d = 0; d < act_dim; d++) {
            act_vec[d] = action[0][d].item<float>();
        }
        return Action::continuous(std::move(act_vec));
    } else {
        int32_t action_id;
        if (deterministic) {
            action_id = action_logits.argmax(1)[0].item<int32_t>();
        } else {
            auto probs = torch::softmax(action_logits, 1);
            action_id = probs.multinomial(1)[0][0].item<int32_t>();
        }
        return Action::discrete(action_id);
    }
}

std::vector<Action> InferenceRuntime::evaluate_batch(
    const std::vector<Tensor>& observations, bool deterministic) {

    if (!impl_->loaded) {
        throw std::runtime_error("No policy loaded");
    }

    if (observations.empty()) return {};

    torch::NoGradGuard no_grad;

    int32_t batch_size = static_cast<int32_t>(observations.size());
    int32_t obs_dim = static_cast<int32_t>(observations[0].data.size());

    // Stack observations into single tensor
    std::vector<float> flat;
    flat.reserve(batch_size * obs_dim);
    for (auto& obs : observations) {
        flat.insert(flat.end(), obs.data.begin(), obs.data.end());
    }

    auto obs_t = torch::from_blob(flat.data(), {batch_size, obs_dim},
                                   torch::kFloat32).clone();

    auto output = impl_->model.forward({obs_t});

    torch::Tensor action_logits;
    if (output.isTuple()) {
        auto elements = output.toTuple()->elements();
        action_logits = elements[0].toTensor();
    } else {
        action_logits = output.toTensor();
    }

    std::vector<Action> actions;
    actions.reserve(batch_size);

    if (impl_->continuous) {
        int32_t act_dim = action_logits.size(1);
        for (int32_t b = 0; b < batch_size; b++) {
            std::vector<float> act_vec(act_dim);
            for (int32_t d = 0; d < act_dim; d++) {
                act_vec[d] = action_logits[b][d].item<float>();
            }
            actions.push_back(Action::continuous(std::move(act_vec)));
        }
    } else {
        torch::Tensor action_ids;
        if (deterministic) {
            action_ids = action_logits.argmax(1);
        } else {
            auto probs = torch::softmax(action_logits, 1);
            action_ids = probs.multinomial(1).squeeze(1);
        }
        for (int32_t b = 0; b < batch_size; b++) {
            actions.push_back(Action::discrete(action_ids[b].item<int32_t>()));
        }
    }

    return actions;
}

SpaceSpec InferenceRuntime::observation_space() const {
    return impl_->obs_space;
}

SpaceSpec InferenceRuntime::action_space() const {
    return impl_->act_space;
}

// ── Agent wrapper ──────────────────────────────────────────────────────────

namespace {

class RuntimeAgent : public IAgent {
public:
    RuntimeAgent(InferenceRuntime& runtime, uint32_t agent_id)
        : runtime_(runtime), agent_id_(agent_id) {}

    Action act(const Tensor& observation, bool deterministic) override {
        return runtime_.evaluate(observation, deterministic);
    }

    uint32_t id() const override { return agent_id_; }

private:
    InferenceRuntime& runtime_;
    uint32_t agent_id_;
};

} // anonymous namespace

std::unique_ptr<IAgent> InferenceRuntime::create_agent(uint32_t agent_id) {
    return std::make_unique<RuntimeAgent>(*this, agent_id);
}

} // namespace hne
