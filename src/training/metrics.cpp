#ifdef HNE_TRAINING

#include <hne/training/callbacks.hpp>
#include <hne/training/trainer.hpp>
#include <hne/training/trainer_config.hpp>
#include <format>
#include <fstream>
#include <iostream>

namespace hne {

// ── TrainerConfig JSON ─────────────────────────────────────────────────────

void to_json(nlohmann::json& j, const TrainerConfig& c) {
    j = {
        {"num_envs", c.num_envs},
        {"rollout_length", c.rollout_length},
        {"gamma", c.gamma},
        {"gae_lambda", c.gae_lambda},
        {"total_timesteps", c.total_timesteps},
        {"eval_interval", c.eval_interval},
        {"eval_episodes", c.eval_episodes},
        {"checkpoint_interval", c.checkpoint_interval},
        {"checkpoint_dir", c.checkpoint_dir},
        {"hidden_sizes", c.hidden_sizes},
        {"ppo", c.ppo},
        {"seed", c.seed},
    };
}

void from_json(const nlohmann::json& j, TrainerConfig& c) {
    if (j.contains("num_envs")) j.at("num_envs").get_to(c.num_envs);
    if (j.contains("rollout_length")) j.at("rollout_length").get_to(c.rollout_length);
    if (j.contains("gamma")) j.at("gamma").get_to(c.gamma);
    if (j.contains("gae_lambda")) j.at("gae_lambda").get_to(c.gae_lambda);
    if (j.contains("total_timesteps")) j.at("total_timesteps").get_to(c.total_timesteps);
    if (j.contains("eval_interval")) j.at("eval_interval").get_to(c.eval_interval);
    if (j.contains("eval_episodes")) j.at("eval_episodes").get_to(c.eval_episodes);
    if (j.contains("checkpoint_interval")) j.at("checkpoint_interval").get_to(c.checkpoint_interval);
    if (j.contains("checkpoint_dir")) j.at("checkpoint_dir").get_to(c.checkpoint_dir);
    if (j.contains("hidden_sizes")) j.at("hidden_sizes").get_to(c.hidden_sizes);
    if (j.contains("ppo")) j.at("ppo").get_to(c.ppo);
    if (j.contains("seed")) j.at("seed").get_to(c.seed);
}

TrainerConfig load_trainer_config(const std::string& path) {
    std::ifstream f(path);
    auto j = nlohmann::json::parse(f);
    TrainerConfig config;
    from_json(j, config);
    return config;
}

void save_trainer_config(const TrainerConfig& config, const std::string& path) {
    nlohmann::json j;
    to_json(j, config);
    std::ofstream f(path);
    f << j.dump(2);
}

// ── Callbacks ──────────────────────────────────────────────────────────────

void ConsoleLogCallback::on_update(const Trainer& trainer,
                                    const TrainingMetrics& metrics) {
    std::cout << std::format(
        "[HNE] iter={} timesteps={} reward={:.2f} ep_len={:.0f}",
        metrics.iteration, metrics.total_timesteps,
        metrics.mean_episode_reward, metrics.mean_episode_length);

    if (auto it = metrics.algorithm_metrics.find("policy_loss");
        it != metrics.algorithm_metrics.end()) {
        std::cout << std::format(
            " p_loss={:.4f} v_loss={:.4f} entropy={:.4f}",
            it->second,
            metrics.algorithm_metrics.at("value_loss"),
            metrics.algorithm_metrics.at("entropy"));
    }
    std::cout << std::endl;
}

void ConsoleLogCallback::on_evaluation(const Trainer& trainer,
                                        float mean_reward, float mean_length) {
    std::cout << std::format(
        "[HNE] EVAL iter={} mean_reward={:.2f} mean_length={:.0f}",
        trainer.current_iteration(), mean_reward, mean_length)
              << std::endl;
}

void LambdaCallback::on_update(const Trainer& trainer,
                                const TrainingMetrics& metrics) {
    if (on_update_fn) on_update_fn(metrics);
}

void LambdaCallback::on_evaluation(const Trainer& trainer,
                                    float mean_reward, float mean_length) {
    if (on_evaluation_fn) on_evaluation_fn(mean_reward, mean_length);
}

void LambdaCallback::on_checkpoint(const Trainer& trainer,
                                    const std::string& path) {
    if (on_checkpoint_fn) on_checkpoint_fn(path);
}

} // namespace hne

#endif // HNE_TRAINING
