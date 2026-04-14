#include <hne/inference/runtime.hpp>

#include <torch/torch.h>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace hne {

namespace {

class NativeInferencePolicy : public torch::nn::Module {
public:
    NativeInferencePolicy(int32_t obs_size,
                          const std::vector<int32_t>& hidden_sizes,
                          const SpaceSpec& act_space)
        : continuous_(std::holds_alternative<BoxSpace>(act_space)) {

        shared_ = torch::nn::Sequential();
        int32_t in_size = obs_size;
        for (int32_t hidden_size : hidden_sizes) {
            shared_->push_back(torch::nn::Linear(in_size, hidden_size));
            shared_->push_back(torch::nn::Tanh());
            in_size = hidden_size;
        }
        register_module("shared", shared_);

        int32_t act_size = 0;
        if (continuous_) {
            const auto& box = std::get<BoxSpace>(act_space);
            act_size = std::accumulate(box.shape.begin(), box.shape.end(),
                                       int32_t{1}, std::multiplies<>());
            log_std_ = register_parameter("log_std", torch::zeros({act_size}));
        } else if (std::holds_alternative<DiscreteSpace>(act_space)) {
            act_size = std::get<DiscreteSpace>(act_space).n;
        } else {
            const auto& mds = std::get<MultiDiscreteSpace>(act_space);
            act_size = std::accumulate(mds.nvec.begin(), mds.nvec.end(), int32_t{0});
        }

        actor_head_ = torch::nn::Linear(in_size, act_size);
        register_module("actor_head", actor_head_);
    }

    torch::Tensor forward(torch::Tensor observation) {
        auto features = shared_->forward(observation);
        return actor_head_->forward(features);
    }

private:
    torch::nn::Sequential shared_{nullptr};
    torch::nn::Linear actor_head_{nullptr};
    torch::Tensor log_std_;
    bool continuous_ = false;
};

} // namespace

struct InferenceRuntime::Impl {
    std::shared_ptr<NativeInferencePolicy> model;
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
        torch::serialize::InputArchive archive;
        archive.load_from(path);

        torch::Tensor meta_tensor;
        archive.read("meta.json", meta_tensor);
        std::string meta_str(static_cast<char*>(meta_tensor.data_ptr()),
                             meta_tensor.numel());
        auto meta = nlohmann::json::parse(meta_str);
        from_json(meta["obs_space"], impl_->obs_space);
        from_json(meta["act_space"], impl_->act_space);
        impl_->obs_size = meta.value("obs_size", 0);

        std::vector<int32_t> hidden_sizes;
        for (const auto& value : meta.value("hidden_sizes", nlohmann::json::array())) {
            hidden_sizes.push_back(value.get<int32_t>());
        }

        impl_->model = std::make_shared<NativeInferencePolicy>(
            impl_->obs_size, hidden_sizes, impl_->act_space);

        for (auto& p : impl_->model->named_parameters()) {
            torch::Tensor loaded;
            archive.read(std::string("policy.") + p.key(), loaded);
            p.value().data().copy_(loaded);
        }

        impl_->model->eval();

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

    auto action_logits = impl_->model->forward(obs_t);

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

    auto action_logits = impl_->model->forward(obs_t);

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
