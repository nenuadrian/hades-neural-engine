#pragma once

#include "types.hpp"

#include <concepts>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hne {

// ── Concept ────────────────────────────────────────────────────────────────
// Compile-time constraint for environment types (used in templates).

template <typename T>
concept Environment = requires(T env, const Action& a, int32_t seed) {
    { env.observation_space() } -> std::same_as<SpaceSpec>;
    { env.action_space() } -> std::same_as<SpaceSpec>;
    { env.reset(seed) } -> std::same_as<Tensor>;
    { env.step(a) } -> std::same_as<StepResult>;
};

// ── Virtual interface ──────────────────────────────────────────────────────
// Type-erased interface for containers (VectorizedEnv, Trainer).

class IEnvironment {
public:
    virtual ~IEnvironment() = default;

    [[nodiscard]] virtual SpaceSpec observation_space() const = 0;
    [[nodiscard]] virtual SpaceSpec action_space() const = 0;

    virtual Tensor reset(int32_t seed = -1) = 0;
    virtual StepResult step(const Action& action) = 0;

    virtual void render([[maybe_unused]] const std::string& mode = "human") {}
    [[nodiscard]] virtual std::string name() const { return "unnamed_env"; }
};

// ── Vectorized environment ─────────────────────────────────────────────────
// Runs N independent environment instances. Used by the Trainer for
// parallel rollout collection. Auto-resets terminated environments.

class VectorizedEnv {
public:
    using EnvFactory = std::function<std::unique_ptr<IEnvironment>()>;

    VectorizedEnv(EnvFactory factory, int32_t num_envs);
    ~VectorizedEnv();

    VectorizedEnv(const VectorizedEnv&) = delete;
    VectorizedEnv& operator=(const VectorizedEnv&) = delete;

    [[nodiscard]] int32_t num_envs() const;
    [[nodiscard]] SpaceSpec observation_space() const;
    [[nodiscard]] SpaceSpec action_space() const;

    std::vector<Tensor> reset_all(int32_t seed = -1);
    std::vector<StepResult> step_all(const std::vector<Action>& actions);

private:
    std::vector<std::unique_ptr<IEnvironment>> envs_;
    SpaceSpec obs_space_;
    SpaceSpec act_space_;
};

} // namespace hne
