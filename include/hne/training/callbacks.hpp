#pragma once

#ifdef HNE_TRAINING

#include "metrics.hpp"
#include <functional>
#include <string>

namespace hne {

class Trainer; // forward

class ITrainerCallback {
public:
    virtual ~ITrainerCallback() = default;

    virtual void on_training_start([[maybe_unused]] const Trainer& trainer) {}
    virtual void on_training_end([[maybe_unused]] const Trainer& trainer) {}
    virtual void on_rollout_start([[maybe_unused]] const Trainer& trainer) {}
    virtual void on_rollout_end([[maybe_unused]] const Trainer& trainer) {}
    virtual void on_step([[maybe_unused]] const Trainer& trainer,
                         [[maybe_unused]] int64_t timestep) {}
    virtual void on_update([[maybe_unused]] const Trainer& trainer,
                           [[maybe_unused]] const TrainingMetrics& metrics) {}
    virtual void on_evaluation([[maybe_unused]] const Trainer& trainer,
                               [[maybe_unused]] float mean_reward,
                               [[maybe_unused]] float mean_length) {}
    virtual void on_checkpoint([[maybe_unused]] const Trainer& trainer,
                               [[maybe_unused]] const std::string& path) {}
};

class ConsoleLogCallback : public ITrainerCallback {
public:
    void on_update(const Trainer& trainer, const TrainingMetrics& metrics) override;
    void on_evaluation(const Trainer& trainer, float mean_reward,
                       float mean_length) override;
};

class LambdaCallback : public ITrainerCallback {
public:
    std::function<void(const TrainingMetrics&)> on_update_fn;
    std::function<void(float, float)> on_evaluation_fn;
    std::function<void(const std::string&)> on_checkpoint_fn;

    void on_update(const Trainer& trainer, const TrainingMetrics& metrics) override;
    void on_evaluation(const Trainer& trainer, float mean_reward,
                       float mean_length) override;
    void on_checkpoint(const Trainer& trainer, const std::string& path) override;
};

} // namespace hne

#endif // HNE_TRAINING
