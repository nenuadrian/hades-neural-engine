# Getting Started

## Requirements

- CMake 3.20 or newer
- A C++20-capable compiler
- LibTorch when building training or inference targets
- Python 3.9+ if you want to build this documentation site locally

## Add HNE To A Project

The simplest integration path is to vendor the repository and include it with `add_subdirectory`:

```cmake
set(HNE_BUILD_TRAINING ON CACHE BOOL "" FORCE)
set(HNE_BUILD_INFERENCE ON CACHE BOOL "" FORCE)
set(HNE_BUILD_IMGUI_WIDGETS ON CACHE BOOL "" FORCE)
add_subdirectory(hades-neural-engine)
```

Link only the targets you need:

```cmake
target_link_libraries(MyEditor PRIVATE hne_training hne_inference hne_imgui)
target_link_libraries(MyRuntime PRIVATE hne_core hne_inference)
```

## Build Options

| Option | Default | Meaning |
| --- | --- | --- |
| `HNE_BUILD_TRAINING` | `OFF` | Enables training components that depend on LibTorch |
| `HNE_BUILD_INFERENCE` | `OFF` | Enables TorchScript inference support |
| `HNE_BUILD_IMGUI_WIDGETS` | `OFF` | Builds optional ImGui widgets for training/editor flows |
| `HNE_BUILD_TESTS` | `OFF` | Builds unit tests |
| `HNE_BUILD_EXAMPLES` | `OFF` | Builds example programs |

## Configure And Build

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

## Implement An Environment

All training begins with an `IEnvironment` implementation:

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

## Build The Docs Locally

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```

The generated static site is written to `site/` when you run `mkdocs build`.
