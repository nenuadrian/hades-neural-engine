#include <hne/imgui/training_widgets.hpp>

// ImGui header provided by the consuming project
#include <imgui.h>

#include <algorithm>
#include <format>
#include <string>

namespace hne::imgui {

bool render_trainer_config_editor(TrainerConfig& config) {
    bool start = false;

    if (ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("Num Envs", &config.num_envs, 1, 64);
        ImGui::InputInt("Rollout Length", &config.rollout_length, 256, 1024);
    }

    if (ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_DefaultOpen)) {
        int total_ts = static_cast<int>(config.total_timesteps);
        if (ImGui::InputInt("Total Timesteps", &total_ts, 100000, 1000000)) {
            config.total_timesteps = total_ts;
        }
        ImGui::SliderFloat("Gamma", &config.gamma, 0.9f, 0.999f, "%.3f");
        ImGui::SliderFloat("GAE Lambda", &config.gae_lambda, 0.8f, 1.0f, "%.2f");
        ImGui::InputInt("Eval Interval", &config.eval_interval, 1, 10);
        ImGui::InputInt("Eval Episodes", &config.eval_episodes, 1, 5);
        ImGui::InputInt("Checkpoint Interval", &config.checkpoint_interval, 10, 50);
        ImGui::InputInt("Seed", &config.seed, 1, 10);
    }

    if (ImGui::CollapsingHeader("Policy Network", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Simple display of hidden sizes
        std::string sizes_str;
        for (size_t i = 0; i < config.hidden_sizes.size(); i++) {
            if (i > 0) sizes_str += ", ";
            sizes_str += std::to_string(config.hidden_sizes[i]);
        }
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s", sizes_str.c_str());
        if (ImGui::InputText("Hidden Sizes", buf, sizeof(buf))) {
            // Parse comma-separated ints
            config.hidden_sizes.clear();
            std::string s(buf);
            size_t pos = 0;
            while (pos < s.size()) {
                auto next = s.find(',', pos);
                if (next == std::string::npos) next = s.size();
                auto token = s.substr(pos, next - pos);
                // Trim
                auto start_idx = token.find_first_not_of(" ");
                if (start_idx != std::string::npos) {
                    try {
                        config.hidden_sizes.push_back(std::stoi(token.substr(start_idx)));
                    } catch (...) {}
                }
                pos = next + 1;
            }
            if (config.hidden_sizes.empty()) config.hidden_sizes = {64, 64};
        }
    }

    if (ImGui::CollapsingHeader("PPO", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Clip Epsilon", &config.ppo.clip_epsilon, 0.05f, 0.5f, "%.2f");
        ImGui::SliderFloat("Value Loss Coeff", &config.ppo.value_loss_coeff, 0.1f, 1.0f, "%.2f");
        ImGui::SliderFloat("Entropy Coeff", &config.ppo.entropy_coeff, 0.0f, 0.1f, "%.3f");
        ImGui::SliderFloat("Max Grad Norm", &config.ppo.max_grad_norm, 0.1f, 5.0f, "%.1f");

        float lr = config.ppo.learning_rate;
        if (ImGui::InputFloat("Learning Rate", &lr, 1e-5f, 1e-3f, "%.6f")) {
            config.ppo.learning_rate = lr;
        }

        ImGui::SliderInt("PPO Epochs", &config.ppo.num_epochs, 1, 30);
        ImGui::InputInt("Mini Batch Size", &config.ppo.mini_batch_size, 16, 64);
        ImGui::Checkbox("Normalize Advantages", &config.ppo.normalize_advantages);

        float target_kl = config.ppo.target_kl;
        ImGui::InputFloat("Target KL (-1 = off)", &target_kl, 0.001f, 0.01f, "%.4f");
        config.ppo.target_kl = target_kl;
    }

    ImGui::Separator();
    if (ImGui::Button("Start Training", ImVec2(-1, 30))) {
        start = true;
    }

    return start;
}

void render_metrics_dashboard(const std::deque<TrainingMetrics>& metrics_history,
                              int32_t current_iteration,
                              int64_t total_timesteps) {
    ImGui::Text("Iteration: %d | Timesteps: %lld", current_iteration,
                static_cast<long long>(total_timesteps));
    ImGui::Separator();

    if (metrics_history.empty()) {
        ImGui::TextDisabled("No metrics yet...");
        return;
    }

    const auto& latest = metrics_history.back();

    ImGui::Text("Mean Episode Reward: %.2f", latest.mean_episode_reward);
    ImGui::Text("Mean Episode Length: %.0f", latest.mean_episode_length);

    for (const auto& [key, val] : latest.algorithm_metrics) {
        ImGui::Text("%s: %.4f", key.c_str(), val);
    }

    // Plot episode rewards over time
    if (metrics_history.size() > 1) {
        std::vector<float> rewards;
        rewards.reserve(metrics_history.size());
        for (const auto& m : metrics_history) {
            rewards.push_back(m.mean_episode_reward);
        }
        ImGui::PlotLines("Reward", rewards.data(),
                         static_cast<int>(rewards.size()),
                         0, nullptr, FLT_MAX, FLT_MAX,
                         ImVec2(0, 80));

        // Plot policy loss
        std::vector<float> losses;
        losses.reserve(metrics_history.size());
        for (const auto& m : metrics_history) {
            auto it = m.algorithm_metrics.find("policy_loss");
            losses.push_back(it != m.algorithm_metrics.end() ? it->second : 0.0f);
        }
        ImGui::PlotLines("Policy Loss", losses.data(),
                         static_cast<int>(losses.size()),
                         0, nullptr, FLT_MAX, FLT_MAX,
                         ImVec2(0, 60));
    }
}

void render_reward_curve(const std::deque<float>& eval_rewards,
                         const std::string& label) {
    if (eval_rewards.empty()) {
        ImGui::TextDisabled("No evaluation results yet...");
        return;
    }

    std::vector<float> data(eval_rewards.begin(), eval_rewards.end());
    ImGui::PlotLines(label.c_str(), data.data(),
                     static_cast<int>(data.size()),
                     0, nullptr, FLT_MAX, FLT_MAX,
                     ImVec2(0, 100));

    ImGui::Text("Latest: %.2f  |  Best: %.2f",
                data.back(),
                *std::max_element(data.begin(), data.end()));
}

std::string render_training_controls(bool is_running, bool is_paused) {
    std::string action;

    if (!is_running) {
        if (ImGui::Button("Start", ImVec2(80, 0))) {
            action = "start";
        }
    } else {
        if (is_paused) {
            if (ImGui::Button("Resume", ImVec2(80, 0))) {
                action = "resume";
            }
        } else {
            if (ImGui::Button("Pause", ImVec2(80, 0))) {
                action = "pause";
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop", ImVec2(80, 0))) {
            action = "stop";
        }
    }

    return action;
}

} // namespace hne::imgui
