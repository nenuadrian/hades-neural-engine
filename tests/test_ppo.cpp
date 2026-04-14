#include <hne/core/policy.hpp>
#include <hne/core/replay_buffer.hpp>
#include <hne/algorithms/ppo.hpp>
#include <gtest/gtest.h>

using namespace hne;

TEST(MLPPolicyTest, DiscreteForward) {
    auto policy = std::make_shared<MLPPolicy>(MLPPolicy::Config{
        .obs_size = 4,
        .action_space = DiscreteSpace{.n = 2},
        .hidden_sizes = {32, 32},
    });

    auto obs = torch::randn({1, 4});
    auto out = policy->forward(obs);

    EXPECT_EQ(out.action_logits.size(0), 1);
    EXPECT_EQ(out.action_logits.size(1), 2);
    EXPECT_EQ(out.value.size(0), 1);
    EXPECT_EQ(out.value.size(1), 1);
}

TEST(MLPPolicyTest, ContinuousForward) {
    auto policy = std::make_shared<MLPPolicy>(MLPPolicy::Config{
        .obs_size = 8,
        .action_space = BoxSpace{.shape = {3}, .low = {-1, -1, -1}, .high = {1, 1, 1}},
        .hidden_sizes = {64, 64},
    });

    auto obs = torch::randn({4, 8}); // batch of 4
    auto out = policy->forward(obs);

    EXPECT_EQ(out.action_logits.size(0), 4);
    EXPECT_EQ(out.action_logits.size(1), 3);
    EXPECT_EQ(out.value.size(0), 4);
    EXPECT_EQ(out.log_std.size(1), 3);
}

TEST(PPOTest, DiscreteUpdate) {
    auto policy = std::make_shared<MLPPolicy>(MLPPolicy::Config{
        .obs_size = 4,
        .action_space = DiscreteSpace{.n = 2},
        .hidden_sizes = {32},
    });

    RolloutBuffer::Config buf_config{
        .buffer_size = 32,
        .num_envs = 1,
        .obs_size = 4,
        .action_space = DiscreteSpace{.n = 2},
    };
    RolloutBuffer buffer(buf_config);

    // Fill buffer with random data
    torch::NoGradGuard no_grad;
    for (int i = 0; i < 32; i++) {
        auto obs_t = torch::randn({1, 4});
        auto out = policy->forward(obs_t);
        auto probs = torch::softmax(out.action_logits, 1);
        auto action = probs.multinomial(1).squeeze(1);
        auto log_probs = torch::log_softmax(out.action_logits, 1);
        auto log_prob = log_probs.gather(1, action.unsqueeze(0).unsqueeze(1)).squeeze();

        Tensor obs = Tensor::from_flat({
            obs_t[0][0].item<float>(),
            obs_t[0][1].item<float>(),
            obs_t[0][2].item<float>(),
            obs_t[0][3].item<float>()
        });

        buffer.add(obs, Action::discrete(action.item<int>()),
                   1.0f, (i == 31), out.value.item<float>(), log_prob.item<float>());
    }

    buffer.compute_returns_and_advantages({0.0f});

    PPOConfig ppo_config;
    ppo_config.num_epochs = 2;
    ppo_config.mini_batch_size = 16;
    PPO ppo(ppo_config);

    auto metrics = ppo.update(*policy, buffer);

    EXPECT_TRUE(metrics.scalars.count("policy_loss") > 0);
    EXPECT_TRUE(metrics.scalars.count("value_loss") > 0);
    EXPECT_TRUE(metrics.scalars.count("entropy") > 0);
    EXPECT_GT(metrics.scalars.at("n_updates"), 0.0f);
}

TEST(PPOConfigTest, JsonRoundTrip) {
    PPOConfig original;
    original.clip_epsilon = 0.3f;
    original.learning_rate = 1e-3f;
    original.num_epochs = 5;

    nlohmann::json j;
    to_json(j, original);

    PPOConfig loaded;
    from_json(j, loaded);

    EXPECT_FLOAT_EQ(loaded.clip_epsilon, 0.3f);
    EXPECT_FLOAT_EQ(loaded.learning_rate, 1e-3f);
    EXPECT_EQ(loaded.num_epochs, 5);
}
