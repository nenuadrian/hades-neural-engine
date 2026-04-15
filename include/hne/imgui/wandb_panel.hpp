#pragma once

#ifdef HNE_WANDB

#include "../wandb/callback.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>

namespace hne::imgui {

// Persistent UI state for the W&B panel. Outlives individual ImGui frames —
// the caller owns one of these alongside its trainer.
//
// The `callback` member is the shared_ptr the consumer attaches to its
// trainer via `trainer->add_callback(state.callback)`. shared_ptr<WandbCallback>
// implicitly converts to shared_ptr<ITrainerCallback>, which is the signature
// of `Trainer::add_callback`.
struct WandbPanelState {
    std::shared_ptr<hne::wandb::WandbCallback> callback;

    // UI buffers. Sized generously so users can paste long keys / names.
    char entity_buf[128]   = "";
    char project_buf[128]  = "";
    char run_name_buf[128] = "";
    char api_key_buf[128]  = "";  // masked; blank → falls back to $WANDB_API_KEY
    char base_url_buf[256] = "https://api.wandb.ai";
    bool upload_checkpoints = false;

    // Human-readable status line the caller can mirror in its own UI if
    // desired. Reset on each Start / Stop.
    std::string status_message;
};

// Draw the panel. Returns true on the frame the user clicks "Start" and a new
// callback was constructed (i.e. state.callback was just reset to a live run).
// `hyperparameters`, when non-null, is deep-copied into the new run's config.
bool render_wandb_panel(WandbPanelState& state,
                        const nlohmann::json* hyperparameters = nullptr);

} // namespace hne::imgui

#endif // HNE_WANDB
