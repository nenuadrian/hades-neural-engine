#pragma once

#include <hne/core/environment.hpp>
#include <cmath>
#include <random>

namespace hne::examples {

// Classic CartPole-v1 environment for testing PPO training.
// Observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
// Action: 0 = push left, 1 = push right
// Reward: +1 per step (max 500 steps)

class CartPoleEnv : public IEnvironment {
public:
    SpaceSpec observation_space() const override {
        return BoxSpace{
            .shape = {4},
            .low = {-4.8f, -3.4e38f, -0.418f, -3.4e38f},
            .high = {4.8f, 3.4e38f, 0.418f, 3.4e38f},
        };
    }

    SpaceSpec action_space() const override {
        return DiscreteSpace{.n = 2};
    }

    Tensor reset(int32_t seed = -1) override {
        if (seed >= 0) rng_.seed(seed);

        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        x_ = dist(rng_);
        x_dot_ = dist(rng_);
        theta_ = dist(rng_);
        theta_dot_ = dist(rng_);
        steps_ = 0;

        return get_obs();
    }

    StepResult step(const Action& action) override {
        int32_t act = action.as_discrete();
        float force = (act == 1) ? force_mag_ : -force_mag_;

        float cos_theta = std::cos(theta_);
        float sin_theta = std::sin(theta_);
        float temp = (force + polemass_length_ * theta_dot_ * theta_dot_ * sin_theta) / total_mass_;
        float theta_acc = (gravity_ * sin_theta - cos_theta * temp) /
                          (length_ * (4.0f / 3.0f - masspole_ * cos_theta * cos_theta / total_mass_));
        float x_acc = temp - polemass_length_ * theta_acc * cos_theta / total_mass_;

        // Euler integration
        x_ += tau_ * x_dot_;
        x_dot_ += tau_ * x_acc;
        theta_ += tau_ * theta_dot_;
        theta_dot_ += tau_ * theta_acc;
        steps_++;

        bool terminated = std::abs(x_) > x_threshold_ ||
                          std::abs(theta_) > theta_threshold_;
        bool truncated = steps_ >= max_steps_;

        return StepResult{
            .observation = get_obs(),
            .reward = 1.0f,
            .terminated = terminated,
            .truncated = truncated,
        };
    }

    std::string name() const override { return "CartPole-v1"; }

private:
    Tensor get_obs() const {
        return Tensor::from_flat({x_, x_dot_, theta_, theta_dot_});
    }

    // Physics constants
    static constexpr float gravity_ = 9.8f;
    static constexpr float masscart_ = 1.0f;
    static constexpr float masspole_ = 0.1f;
    static constexpr float total_mass_ = masscart_ + masspole_;
    static constexpr float length_ = 0.5f;
    static constexpr float polemass_length_ = masspole_ * length_;
    static constexpr float force_mag_ = 10.0f;
    static constexpr float tau_ = 0.02f;
    static constexpr float x_threshold_ = 2.4f;
    static constexpr float theta_threshold_ = 12.0f * 2.0f * 3.14159265f / 360.0f;
    static constexpr int32_t max_steps_ = 500;

    // State
    float x_ = 0.0f;
    float x_dot_ = 0.0f;
    float theta_ = 0.0f;
    float theta_dot_ = 0.0f;
    int32_t steps_ = 0;
    std::mt19937 rng_{42};
};

} // namespace hne::examples
