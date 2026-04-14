# Architecture

## Design Goals

HNE keeps the reinforcement learning stack layered so game integration code does not need to absorb unnecessary machine learning dependencies.

## Main Layers

### `hne_core`

The always-available foundation layer contains:

- shape and transport types like `Tensor`, `Action`, `StepResult`, and `SpaceSpec`
- the `IEnvironment` and `IAgent` interfaces used by training and inference
- serialization helpers for environment-facing data structures

This is the layer an engine runtime can depend on without taking a LibTorch dependency.

### `hne_training`

The training layer adds:

- `Trainer` orchestration
- PPO optimization
- rollout storage
- metrics and checkpointing
- policy modules

This is where vectorized environment sampling and optimization happen.

### `hne_inference`

The inference layer is responsible for:

- loading exported TorchScript policies
- evaluating single observations
- evaluating batches of observations
- creating runtime `IAgent` instances for engine entities

### `hne_imgui`

The optional ImGui layer provides editor-facing tooling. It is intentionally isolated so projects can opt into UI support without forcing it onto headless builds.

## Core Abstractions

| Type | Role |
| --- | --- |
| `hne::IEnvironment` | Minimal interface for reset/step loops and space introspection |
| `hne::VectorizedEnv` | Manages multiple independent environments for rollout collection |
| `hne::Trainer` | High-level control surface for training, evaluation, checkpointing, and export |
| `hne::IAlgorithm` | Pluggable optimization interface |
| `hne::IPolicy` | LibTorch-backed policy abstraction used by training |
| `hne::InferenceRuntime` | Loads and evaluates exported policies at runtime |

## Typical Flow

1. Implement an `IEnvironment`.
2. Configure a `TrainerConfig`.
3. Train with `hne::Trainer`.
4. Export the resulting policy as TorchScript.
5. Load that artifact through `hne::InferenceRuntime`.
6. Create runtime agents or call `evaluate` directly from gameplay code.
