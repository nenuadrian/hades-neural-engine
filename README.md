# Hades Neural Engine (HNE)

[![docs](https://github.com/nenuadrian/hades-neural-engine/actions/workflows/docs.yml/badge.svg)](https://github.com/nenuadrian/hades-neural-engine/actions/workflows/docs.yml) [![Build and Test](https://github.com/nenuadrian/hades-neural-engine/actions/workflows/build.yml/badge.svg)](https://github.com/nenuadrian/hades-neural-engine/actions/workflows/build.yml)

C++20 reinforcement learning library for training and deploying neural network policies in game engines. An example engine making use of HNE is [Hades](https://github.com/nenuadrian/hades-game-engine), a custom C++ game engine built for research and prototyping.

## Architecture

Three static libraries with increasing dependency weight:

| Library | Dependency | Purpose |
|---------|-----------|---------|
| `hne_core` | nlohmann/json | Types, interfaces (`IEnvironment`, `IAgent`, `SpaceSpec`, `Tensor`) |
| `hne_training` | hne_core + LibTorch | PPO trainer, MLPPolicy, RolloutBuffer, checkpointing |
| `hne_inference` | hne_core + LibTorch | TorchScript policy loading, single/batched evaluation |
| `hne_imgui` | hne_core + ImGui | Training config editor, metrics dashboard, reward curves |

## Quick Start

### CMake Integration

```cmake
# In your project's CMakeLists.txt:
set(HNE_BUILD_TRAINING ON CACHE BOOL "" FORCE)
set(HNE_BUILD_INFERENCE ON CACHE BOOL "" FORCE)
add_subdirectory(hades-neural-engine)

# Editor (training + inference):
target_link_libraries(MyEditor PRIVATE hne_training hne_inference hne_imgui)

# Runtime (inference only):
target_link_libraries(MyRuntime PRIVATE hne_core hne_inference)
```

### Implement an Environment

```cpp
#include <hne/core/environment.hpp>

class MyGameEnv : public hne::IEnvironment {
public:
    hne::SpaceSpec observation_space() const override {
        return hne::BoxSpace{.shape = {8}};
    }
    hne::SpaceSpec action_space() const override {
        return hne::DiscreteSpace{.n = 4};
    }
    hne::Tensor reset(int32_t seed) override { /* ... */ }
    hne::StepResult step(const hne::Action& action) override { /* ... */ }
};
```

### Train a Policy

```cpp
#include <hne/hne.hpp>

hne::TrainerConfig config;
config.total_timesteps = 500000;

hne::Trainer trainer(config);
trainer.set_environment_factory([]() {
    return std::make_unique<MyGameEnv>();
});
trainer.add_callback(std::make_shared<hne::ConsoleLogCallback>());

trainer.train();                           // blocking
trainer.export_policy("my_policy.pt");     // TorchScript export
```

### Run at Runtime

```cpp
#include <hne/inference/runtime.hpp>

hne::InferenceRuntime runtime;
runtime.load("my_policy.pt");

auto agent = runtime.create_agent(entity_id);
auto action = agent->act(observation, /*deterministic=*/true);
```

## Building

Requires LibTorch for training/inference targets. Core library has no LibTorch dependency.

```bash
cmake -B build \
    -DHNE_BUILD_TRAINING=ON \
    -DHNE_BUILD_INFERENCE=ON \
    -DHNE_BUILD_TESTS=ON \
    -DHNE_BUILD_EXAMPLES=ON \
    -DCMAKE_PREFIX_PATH=/path/to/libtorch

cmake --build build
ctest --test-dir build
```

## Documentation

MkDocs Material documentation is configured in this repository and deployed through GitHub Pages from the `main` branch workflow in `.github/workflows/docs.yml`.

To preview it locally:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```

## Algorithms

- **PPO** (Proximal Policy Optimization) with GAE, clipped surrogate loss, optional KL early stopping

## License

MIT License. See [LICENSE](LICENSE) for details.
