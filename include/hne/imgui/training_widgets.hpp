#pragma once

#include "../training/metrics.hpp"
#include "../training/trainer_config.hpp"
#include <deque>
#include <string>

namespace hne::imgui {

// Render a training config editor. Returns true if the user clicks "Start Training".
bool render_trainer_config_editor(TrainerConfig& config);

// Render a live metrics dashboard.
void render_metrics_dashboard(const std::deque<TrainingMetrics>& metrics_history,
                              int32_t current_iteration,
                              int64_t total_timesteps);

// Render a reward curve plot.
void render_reward_curve(const std::deque<float>& eval_rewards,
                         const std::string& label = "Mean Eval Reward");

// Render training control buttons. Returns action: "start", "stop", "pause", "resume", or "".
std::string render_training_controls(bool is_running, bool is_paused);

} // namespace hne::imgui
