#ifdef HNE_TRAINING

#include <hne/training/checkpoint.hpp>
#include <torch/script.h>
#include <fstream>
#include <iostream>

namespace hne::checkpoint {

bool save(const std::string& path,
          const IPolicy& policy,
          const torch::optim::Adam& optimizer,
          const CheckpointData& data) {
    try {
        // Save as a bundle: policy state dict + optimizer state dict + metadata
        torch::serialize::OutputArchive archive;

        // Policy parameters
        auto policy_params = policy.named_parameters();
        for (const auto& p : policy_params) {
            archive.write(std::string("policy.") + p.key(), p.value());
        }

        // Metadata as tensors
        archive.write("meta.iteration",
            torch::tensor(static_cast<int64_t>(data.iteration)));
        archive.write("meta.total_timesteps",
            torch::tensor(data.total_timesteps));

        // Config as JSON string stored in a tensor
        nlohmann::json config_json;
        to_json(config_json, data.config);
        std::string config_str = config_json.dump();
        auto config_tensor = torch::zeros({static_cast<int64_t>(config_str.size())},
                                           torch::kByte);
        std::memcpy(config_tensor.data_ptr(), config_str.data(), config_str.size());
        archive.write("meta.config", config_tensor);

        archive.save_to(path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HNE] Checkpoint save error: " << e.what() << std::endl;
        return false;
    }
}

bool load(const std::string& path,
          IPolicy& policy,
          torch::optim::Adam& optimizer,
          CheckpointData& data) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);

        // Load policy parameters
        auto policy_params = policy.named_parameters();
        for (auto& p : policy_params) {
            torch::Tensor loaded;
            archive.read(std::string("policy.") + p.key(), loaded);
            p.value().data().copy_(loaded);
        }

        // Load metadata
        torch::Tensor iter_t, ts_t, config_t;
        archive.read("meta.iteration", iter_t);
        archive.read("meta.total_timesteps", ts_t);
        data.iteration = static_cast<int32_t>(iter_t.item<int64_t>());
        data.total_timesteps = ts_t.item<int64_t>();

        archive.read("meta.config", config_t);
        std::string config_str(static_cast<char*>(config_t.data_ptr()),
                                config_t.numel());
        auto config_json = nlohmann::json::parse(config_str);
        from_json(config_json, data.config);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HNE] Checkpoint load error: " << e.what() << std::endl;
        return false;
    }
}

bool export_policy(const std::string& path,
                   IPolicy& policy,
                   const SpaceSpec& obs_space,
                   const SpaceSpec& act_space,
                   int32_t obs_size) {
    try {
        policy.eval();
        auto example_input = torch::randn({1, obs_size});
        auto traced = torch::jit::trace(policy, {example_input});

        // Store space specs as extra files
        nlohmann::json meta;
        to_json(meta["obs_space"], obs_space);
        to_json(meta["act_space"], act_space);
        meta["obs_size"] = obs_size;
        std::string meta_str = meta.dump();

        // Save the traced module
        traced.save(path, {{"hne_meta.json", meta_str}});

        policy.train();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HNE] Policy export error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace hne::checkpoint

#endif // HNE_TRAINING
