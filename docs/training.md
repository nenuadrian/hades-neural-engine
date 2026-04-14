# Training

## Enable Training

Training support is compiled behind `HNE_BUILD_TRAINING` and the `HNE_TRAINING` define. This keeps the core API usable even when LibTorch is unavailable.

```bash
cmake -B build \
    -DHNE_BUILD_TRAINING=ON \
    -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

## Configure A Trainer

`hne::TrainerConfig` controls rollout collection, PPO settings, evaluation cadence, checkpoint output, and seeding.

Common knobs include:

- `num_envs` for parallel environment instances
- `rollout_length`, `gamma`, and `gae_lambda` for rollout and return calculation
- `total_timesteps` for the overall training budget
- `eval_interval` and `eval_episodes` for periodic evaluation
- `checkpoint_interval` and `checkpoint_dir` for persistence
- `hidden_sizes` for the default MLP policy architecture
- `ppo` for PPO-specific hyperparameters

## Train A Policy

```cpp
#include <hne/hne.hpp>

hne::TrainerConfig config;
config.total_timesteps = 500000;
config.num_envs = 8;
config.checkpoint_dir = "checkpoints";

hne::Trainer trainer(config);
trainer.set_environment_factory([]() {
    return std::make_unique<MyGameEnv>();
});
trainer.add_callback(std::make_shared<hne::ConsoleLogCallback>());

trainer.train();
trainer.export_policy("my_policy.pt");
```

## Runtime Control

`hne::Trainer` supports both blocking and background execution:

- `train()` runs training synchronously
- `train_async()` runs training on a worker thread
- `pause()`, `resume()`, and `request_stop()` control long-running jobs
- `state()`, `current_iteration()`, `total_timesteps()`, and `latest_metrics()` expose progress

## Evaluation And Persistence

The trainer can:

- run evaluation episodes with `evaluate(...)`
- save and load checkpoints
- export a trained policy for inference use

That split is what makes the training/inference boundary practical: training artifacts are produced once, then loaded by the lighter runtime API.
