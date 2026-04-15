#ifdef HNE_WANDB

#include <hne/imgui/wandb_panel.hpp>
#include <hne/wandb/callback.hpp>

#include <imgui.h>

#include <memory>
#include <string>
#include <utility>

namespace hne::imgui {

bool render_wandb_panel(WandbPanelState& state,
                        const nlohmann::json* hyperparameters) {
    bool started = false;

    const bool active = static_cast<bool>(state.callback);

    // ── Inputs — disabled while a run is live so the user doesn't edit them
    // under the running callback.
    ImGui::BeginDisabled(active);
    ImGui::InputText("Entity",   state.entity_buf,   sizeof(state.entity_buf));
    ImGui::InputText("Project",  state.project_buf,  sizeof(state.project_buf));
    ImGui::InputText("Run Name", state.run_name_buf, sizeof(state.run_name_buf));
    ImGui::InputText("API Key",  state.api_key_buf,  sizeof(state.api_key_buf),
                     ImGuiInputTextFlags_Password);
    ImGui::InputText("Base URL", state.base_url_buf, sizeof(state.base_url_buf));
    ImGui::Checkbox("Upload Checkpoints", &state.upload_checkpoints);
    ImGui::EndDisabled();

    // ── Controls
    if (!active) {
        const bool can_start = (state.project_buf[0] != '\0');
        ImGui::BeginDisabled(!can_start);
        if (ImGui::Button("Start W&B Run", ImVec2(160, 0))) {
            hne::wandb::RunConfig cfg;
            cfg.entity   = state.entity_buf;
            cfg.project  = state.project_buf;
            cfg.run_name = state.run_name_buf;
            cfg.api_key  = state.api_key_buf;
            if (state.base_url_buf[0] != '\0') {
                cfg.base_url = state.base_url_buf;
            }
            cfg.upload_checkpoints = state.upload_checkpoints;
            if (hyperparameters && !hyperparameters->is_null()) {
                cfg.hyperparameters = *hyperparameters;
            }

            state.callback = std::make_shared<hne::wandb::WandbCallback>(std::move(cfg));
            state.status_message = "Run queued — will start when training begins.";
            started = true;
        }
        ImGui::EndDisabled();
        if (!can_start) {
            ImGui::SameLine();
            ImGui::TextDisabled("Set Project first.");
        }
    } else {
        if (ImGui::Button("Stop W&B Run", ImVec2(160, 0))) {
            // shared_ptr reset → WandbCallback destructor joins its uploader
            // thread and flushes remaining rows best-effort.
            state.callback.reset();
            state.status_message = "Stopped.";
        }
    }

    // ── Status line
    if (active) {
        const auto err = state.callback->last_error();
        if (!err.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Error: %s",
                               err.c_str());
        } else if (state.callback->is_running()) {
            const auto url = state.callback->run_url();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Live");
            if (!url.empty()) {
                ImGui::SameLine();
                ImGui::TextUnformatted(url.c_str());
            }
        } else {
            ImGui::TextDisabled("Queued (waiting for training start)...");
        }
    }
    if (!state.status_message.empty()) {
        ImGui::TextDisabled("%s", state.status_message.c_str());
    }

    return started;
}

} // namespace hne::imgui

#endif // HNE_WANDB
