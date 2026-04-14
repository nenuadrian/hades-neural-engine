#include <hne/core/policy.hpp>
#include <hne/training/checkpoint.hpp>
#include <hne/inference/runtime.hpp>
#include <hne/inference/batch_evaluator.hpp>
#include <gtest/gtest.h>
#include <filesystem>

using namespace hne;

class InferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create and export a simple policy
        policy_ = std::make_shared<MLPPolicy>(MLPPolicy::Config{
            .obs_size = 4,
            .action_space = DiscreteSpace{.n = 2},
            .hidden_sizes = {32},
        });

        policy_path_ = "test_policy.pt";
        ASSERT_TRUE(checkpoint::export_policy(
            policy_path_, *policy_,
            BoxSpace{.shape = {4}, .low = {-1,-1,-1,-1}, .high = {1,1,1,1}},
            DiscreteSpace{.n = 2},
            4
        ));
    }

    void TearDown() override {
        std::filesystem::remove(policy_path_);
    }

    std::shared_ptr<MLPPolicy> policy_;
    std::string policy_path_;
};

TEST_F(InferenceTest, LoadAndEvaluate) {
    InferenceRuntime runtime;
    ASSERT_TRUE(runtime.load(policy_path_));
    ASSERT_TRUE(runtime.is_loaded());

    auto obs = Tensor::from_flat({0.1f, -0.2f, 0.05f, 0.3f});
    auto action = runtime.evaluate(obs, true);
    EXPECT_TRUE(action.is_discrete());
    EXPECT_GE(action.as_discrete(), 0);
    EXPECT_LT(action.as_discrete(), 2);
}

TEST_F(InferenceTest, BatchEvaluate) {
    InferenceRuntime runtime;
    ASSERT_TRUE(runtime.load(policy_path_));

    std::vector<Tensor> observations;
    for (int i = 0; i < 10; i++) {
        observations.push_back(Tensor::from_flat({
            static_cast<float>(i) * 0.1f, 0.0f, 0.0f, 0.0f
        }));
    }

    auto actions = runtime.evaluate_batch(observations, true);
    EXPECT_EQ(actions.size(), 10);
    for (auto& a : actions) {
        EXPECT_TRUE(a.is_discrete());
        EXPECT_GE(a.as_discrete(), 0);
        EXPECT_LT(a.as_discrete(), 2);
    }
}

TEST_F(InferenceTest, CreateAgent) {
    InferenceRuntime runtime;
    ASSERT_TRUE(runtime.load(policy_path_));

    auto agent = runtime.create_agent(42);
    EXPECT_EQ(agent->id(), 42);

    auto obs = Tensor::from_flat({0.1f, -0.2f, 0.05f, 0.3f});
    auto action = agent->act(obs, true);
    EXPECT_TRUE(action.is_discrete());
}

TEST_F(InferenceTest, BatchEvaluator) {
    InferenceRuntime runtime;
    ASSERT_TRUE(runtime.load(policy_path_));

    BatchEvaluator evaluator(runtime);

    evaluator.submit(1, Tensor::from_flat({0.1f, 0.0f, 0.0f, 0.0f}));
    evaluator.submit(2, Tensor::from_flat({0.2f, 0.0f, 0.0f, 0.0f}));
    evaluator.submit(3, Tensor::from_flat({0.3f, 0.0f, 0.0f, 0.0f}));

    EXPECT_EQ(evaluator.pending_count(), 3);

    evaluator.evaluate(true);

    EXPECT_EQ(evaluator.pending_count(), 0);

    auto a1 = evaluator.get_action(1);
    auto a2 = evaluator.get_action(2);
    auto a3 = evaluator.get_action(3);
    EXPECT_TRUE(a1.is_discrete());
    EXPECT_TRUE(a2.is_discrete());
    EXPECT_TRUE(a3.is_discrete());
}

TEST_F(InferenceTest, SpaceSpecFromModel) {
    InferenceRuntime runtime;
    ASSERT_TRUE(runtime.load(policy_path_));

    auto obs_space = runtime.observation_space();
    auto act_space = runtime.action_space();

    EXPECT_TRUE(std::holds_alternative<BoxSpace>(obs_space));
    EXPECT_TRUE(std::holds_alternative<DiscreteSpace>(act_space));
    EXPECT_EQ(std::get<DiscreteSpace>(act_space).n, 2);
}
