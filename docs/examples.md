# Examples

## CartPole

The repository includes a CartPole example under `examples/cartpole/` that shows the intended integration shape:

- define an environment implementation in `cartpole_env.hpp`
- wire training in `train_cartpole.cpp`
- build examples with `HNE_BUILD_EXAMPLES=ON`

```bash
cmake -B build \
    -DHNE_BUILD_TRAINING=ON \
    -DHNE_BUILD_EXAMPLES=ON \
    -DCMAKE_PREFIX_PATH=/path/to/libtorch

cmake --build build
```

## Suggested Reading Order

If you are new to the codebase, this order tends to work well:

1. `include/hne/core/environment.hpp`
2. `include/hne/training/trainer_config.hpp`
3. `include/hne/training/trainer.hpp`
4. `include/hne/inference/runtime.hpp`
5. `examples/cartpole/train_cartpole.cpp`

That path gets you from the public abstraction boundary to a concrete end-to-end usage example quickly.
