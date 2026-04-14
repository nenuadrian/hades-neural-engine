#include <hne/core/replay_buffer.hpp>
#include <gtest/gtest.h>

using namespace hne;

TEST(RolloutBufferTest, BasicAddAndFull) {
    RolloutBuffer::Config config{
        .buffer_size = 4,
        .num_envs = 1,
        .obs_size = 3,
        .action_space = DiscreteSpace{.n = 2},
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
    };
    RolloutBuffer buffer(config);

    EXPECT_FALSE(buffer.is_full());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), 4);

    for (int i = 0; i < 4; i++) {
        buffer.add(
            Tensor::from_flat({1.0f, 2.0f, 3.0f}),
            Action::discrete(i % 2),
            1.0f, false, 0.5f, -0.69f
        );
    }

    EXPECT_TRUE(buffer.is_full());
    EXPECT_EQ(buffer.size(), 4);
}

TEST(RolloutBufferTest, GAEComputation) {
    RolloutBuffer::Config config{
        .buffer_size = 3,
        .num_envs = 1,
        .obs_size = 2,
        .action_space = DiscreteSpace{.n = 2},
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
    };
    RolloutBuffer buffer(config);

    // Add 3 transitions
    buffer.add(Tensor::from_flat({1.0f, 0.0f}), Action::discrete(0), 1.0f, false, 0.5f, -0.5f);
    buffer.add(Tensor::from_flat({1.0f, 1.0f}), Action::discrete(1), 1.0f, false, 0.6f, -0.4f);
    buffer.add(Tensor::from_flat({1.0f, 2.0f}), Action::discrete(0), 1.0f, false, 0.7f, -0.3f);

    buffer.compute_returns_and_advantages({0.8f}); // last value

    // Just verify we can get batches without crashing
    auto batches = buffer.get_batches(2);
    EXPECT_FALSE(batches.empty());

    // Total samples across batches should equal buffer size
    int total = 0;
    for (auto& b : batches) {
        total += b.observations.size(0);
    }
    EXPECT_EQ(total, 3);
}

TEST(RolloutBufferTest, ContinuousActions) {
    RolloutBuffer::Config config{
        .buffer_size = 2,
        .num_envs = 1,
        .obs_size = 4,
        .action_space = BoxSpace{.shape = {2}, .low = {-1, -1}, .high = {1, 1}},
        .gamma = 0.99f,
        .gae_lambda = 0.95f,
    };
    RolloutBuffer buffer(config);

    buffer.add(Tensor::from_flat({1, 2, 3, 4}), Action::continuous({0.5f, -0.3f}),
               1.0f, false, 0.5f, -1.0f);
    buffer.add(Tensor::from_flat({5, 6, 7, 8}), Action::continuous({-0.1f, 0.9f}),
               0.0f, true, 0.3f, -0.8f);

    buffer.compute_returns_and_advantages({0.0f});

    auto batches = buffer.get_batches(2);
    EXPECT_EQ(batches.size(), 1);
    EXPECT_EQ(batches[0].actions.size(1), 2); // 2D continuous action
}

TEST(RolloutBufferTest, Reset) {
    RolloutBuffer::Config config{
        .buffer_size = 2,
        .num_envs = 1,
        .obs_size = 2,
        .action_space = DiscreteSpace{.n = 2},
    };
    RolloutBuffer buffer(config);

    buffer.add(Tensor::from_flat({1, 2}), Action::discrete(0), 1.0f, false, 0.5f, -0.5f);
    buffer.add(Tensor::from_flat({3, 4}), Action::discrete(1), 1.0f, false, 0.5f, -0.5f);
    EXPECT_TRUE(buffer.is_full());

    buffer.reset();
    EXPECT_FALSE(buffer.is_full());
    EXPECT_EQ(buffer.size(), 0);
}
