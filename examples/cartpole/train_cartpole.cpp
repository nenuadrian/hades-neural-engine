#include "cartpole_env.hpp"
#include <hne/hne.hpp>
#include <iostream>

int main() {
    std::cout << "=== HNE CartPole PPO Training ===" << std::endl;

    hne::TrainerConfig config;
    config.num_envs = 8;
    config.rollout_length = 2048;
    config.total_timesteps = 200000;
    config.eval_interval = 5;
    config.eval_episodes = 10;
    config.checkpoint_interval = 0; // disable checkpointing for example
    config.hidden_sizes = {64, 64};
    config.seed = 42;
    config.ppo.learning_rate = 3e-4f;
    config.ppo.num_epochs = 10;
    config.ppo.mini_batch_size = 64;
    config.ppo.clip_epsilon = 0.2f;
    config.ppo.entropy_coeff = 0.01f;

    hne::Trainer trainer(config);

    trainer.set_environment_factory([]() {
        return std::make_unique<hne::examples::CartPoleEnv>();
    });

    auto console_cb = std::make_shared<hne::ConsoleLogCallback>();
    trainer.add_callback(console_cb);

    // Track best eval reward
    float best_reward = 0.0f;
    auto eval_cb = std::make_shared<hne::LambdaCallback>();
    eval_cb->on_evaluation_fn = [&best_reward](float mean_reward, float mean_length) {
        if (mean_reward > best_reward) {
            best_reward = mean_reward;
            std::cout << "  >>> New best eval reward: " << best_reward << std::endl;
        }
    };
    trainer.add_callback(eval_cb);

    std::cout << "Starting training..." << std::endl;
    trainer.train();

    std::cout << "\nTraining complete!" << std::endl;
    std::cout << "Best evaluation reward: " << best_reward << std::endl;

    // Export policy
    if (trainer.export_policy("cartpole_policy.pt")) {
        std::cout << "Policy exported to cartpole_policy.pt" << std::endl;

        // Test inference
        hne::InferenceRuntime runtime;
        if (runtime.load("cartpole_policy.pt")) {
            hne::examples::CartPoleEnv env;
            auto obs = env.reset(123);
            float total_reward = 0.0f;
            for (int step = 0; step < 500; step++) {
                auto action = runtime.evaluate(obs, true);
                auto result = env.step(action);
                total_reward += result.reward;
                obs = result.observation;
                if (result.terminated || result.truncated) break;
            }
            std::cout << "Inference test reward: " << total_reward << std::endl;
        }
    }

    return 0;
}
