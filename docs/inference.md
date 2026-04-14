# Inference

## Enable Inference

Inference support is compiled behind `HNE_BUILD_INFERENCE` and the `HNE_INFERENCE` define.

```bash
cmake -B build \
    -DHNE_BUILD_INFERENCE=ON \
    -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

## Export From Training

Inference consumes a TorchScript artifact exported by the trainer:

```cpp
trainer.export_policy("my_policy.pt");
```

## Load And Evaluate

```cpp
#include <hne/inference/runtime.hpp>

hne::InferenceRuntime runtime;
runtime.load("my_policy.pt");

auto action = runtime.evaluate(observation, true);
auto batch = runtime.evaluate_batch(observations, true);
```

## Create Runtime Agents

If your engine architecture wants actor-like wrappers, you can create agents directly from the runtime:

```cpp
auto agent = runtime.create_agent(entity_id);
auto action = agent->act(observation, true);
```

## What The Runtime Exposes

- `load(path)` to bring a TorchScript policy into memory
- `is_loaded()` to guard runtime code paths
- `evaluate(...)` for single observations
- `evaluate_batch(...)` for batched rollout or simulation code
- `observation_space()` and `action_space()` so the runtime can validate integration assumptions

The inference API is intentionally narrow so it can sit inside gameplay systems without importing training-only concepts.
