# API Overview

This page is a high-level map of the public headers shipped by HNE.

## Umbrella Header

- `include/hne/hne.hpp`
  Exposes the main public API surface in one include for projects that want the full stack.

## Core

- `include/hne/core/types.hpp`
  Defines `SpaceSpec`, `Tensor`, `Action`, and `StepResult`.
- `include/hne/core/environment.hpp`
  Defines the `Environment` concept, `IEnvironment`, and `VectorizedEnv`.
- `include/hne/core/agent.hpp`
  Defines the runtime-facing `IAgent` interface.
- `include/hne/core/algorithm.hpp`
  Defines `IAlgorithm` and algorithm metrics contracts.

## Training

- `include/hne/algorithms/ppo.hpp`
  Declares PPO configuration and the PPO algorithm implementation.
- `include/hne/core/policy.hpp`
  Declares `IPolicy`, `MLPPolicy`, and policy outputs.
- `include/hne/core/replay_buffer.hpp`
  Declares rollout storage used during policy optimization.
- `include/hne/training/trainer_config.hpp`
  Defines `TrainerConfig` plus JSON serialization helpers.
- `include/hne/training/metrics.hpp`
  Defines training metrics snapshots.
- `include/hne/training/callbacks.hpp`
  Declares callback hooks such as `ConsoleLogCallback` and `LambdaCallback`.
- `include/hne/training/checkpoint.hpp`
  Declares checkpoint payloads and serialization helpers.
- `include/hne/training/trainer.hpp`
  Declares the `Trainer` orchestration API.

## Inference

- `include/hne/inference/runtime.hpp`
  Declares `InferenceRuntime` for loading and evaluating exported policies.
- `include/hne/inference/batch_evaluator.hpp`
  Declares a helper for batched runtime evaluation.

## ImGui

- `include/hne/imgui/training_widgets.hpp`
  Declares optional editor-facing widgets for training controls and metrics display.

## Next Step

If you want a symbol-by-symbol reference later, the natural upgrade path is to add Doxygen generation and publish its output alongside the MkDocs pages. This initial site is focused on discoverability and integration guidance.
