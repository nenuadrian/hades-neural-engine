# Hades Neural Engine

Hades Neural Engine (HNE) is a C++20 reinforcement learning library for training and deploying neural network policies inside game engines and simulation-heavy applications. It is designed to keep the runtime-facing API lightweight while reserving LibTorch for the parts of the stack that actually need it.

## What You Get

- `hne_core` for engine-friendly types and interfaces such as `IEnvironment`, `IAgent`, `SpaceSpec`, `Tensor`, and `Action`
- `hne_training` for PPO training, rollout collection, checkpointing, and metrics
- `hne_inference` for TorchScript policy loading and runtime evaluation
- `hne_imgui` for optional ImGui widgets used to inspect or drive training workflows

## Library Layout

| Library | Dependency footprint | Purpose |
| --- | --- | --- |
| `hne_core` | `nlohmann/json` | Shared data structures and interfaces |
| `hne_training` | `hne_core` + LibTorch | PPO trainer, policy modules, checkpoints, metrics |
| `hne_inference` | `hne_core` + LibTorch CPU | TorchScript policy loading and evaluation |
| `hne_imgui` | `hne_core` + ImGui | Optional editor-facing widgets |

## Quick Start

```cmake
set(HNE_BUILD_TRAINING ON CACHE BOOL "" FORCE)
set(HNE_BUILD_INFERENCE ON CACHE BOOL "" FORCE)
add_subdirectory(hades-neural-engine)

target_link_libraries(MyEditor PRIVATE hne_training hne_inference hne_imgui)
target_link_libraries(MyRuntime PRIVATE hne_core hne_inference)
```

```cpp
#include <hne/hne.hpp>

hne::TrainerConfig config;
config.total_timesteps = 500000;

hne::Trainer trainer(config);
trainer.set_environment_factory([]() {
    return std::make_unique<MyGameEnv>();
});
trainer.add_callback(std::make_shared<hne::ConsoleLogCallback>());
trainer.train();
trainer.export_policy("my_policy.pt");
```

## Why The Split Matters

The main design choice in HNE is to keep the boundary between game code and ML tooling simple:

- game and simulation code can live against plain C++ structs like `Tensor`, `StepResult`, and `Action`
- training and inference stay modular, so consumers can ship only the parts they need
- runtime inference can be embedded without exposing LibTorch details throughout the engine codebase

## Next Steps

- Start with [Getting Started](getting-started.md) for build flags and local setup
- Read [Architecture](architecture.md) for the module boundaries and main abstractions
- Use [Training](training.md) and [Inference](inference.md) when integrating HNE into a project
