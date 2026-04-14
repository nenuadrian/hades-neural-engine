#include <hne/core/types.hpp>
#include <hne/core/environment.hpp>
#include <hne/core/agent.hpp>
#include <gtest/gtest.h>

using namespace hne;

TEST(TensorTest, FromFlat) {
    auto t = Tensor::from_flat({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(t.numel(), 3);
    EXPECT_EQ(t.shape.size(), 1);
    EXPECT_EQ(t.shape[0], 3);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[2], 3.0f);
}

TEST(TensorTest, Scalar) {
    auto t = Tensor::scalar(42.0f);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_FLOAT_EQ(t[0], 42.0f);
}

TEST(TensorTest, View) {
    auto t = Tensor::from_flat({1.0f, 2.0f, 3.0f});
    auto v = t.view();
    EXPECT_EQ(v.size(), 3);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
}

TEST(TensorTest, MutableView) {
    auto t = Tensor::from_flat({1.0f, 2.0f});
    auto v = t.mutable_view();
    v[0] = 99.0f;
    EXPECT_FLOAT_EQ(t[0], 99.0f);
}

TEST(ActionTest, Discrete) {
    auto a = Action::discrete(3);
    EXPECT_TRUE(a.is_discrete());
    EXPECT_EQ(a.as_discrete(), 3);
}

TEST(ActionTest, Continuous) {
    auto a = Action::continuous({1.0f, 2.0f, 3.0f});
    EXPECT_FALSE(a.is_discrete());
    EXPECT_EQ(a.as_continuous().size(), 3);
    EXPECT_FLOAT_EQ(a.as_continuous()[1], 2.0f);
}

TEST(SpaceSpecTest, FlatSizeDiscrete) {
    SpaceSpec s = DiscreteSpace{.n = 5};
    EXPECT_EQ(flat_size(s), 1);
}

TEST(SpaceSpecTest, FlatSizeBox) {
    SpaceSpec s = BoxSpace{.shape = {4}, .low = {}, .high = {}};
    EXPECT_EQ(flat_size(s), 4);
}

TEST(SpaceSpecTest, FlatSizeMultiDiscrete) {
    SpaceSpec s = MultiDiscreteSpace{.nvec = {3, 4, 5}};
    EXPECT_EQ(flat_size(s), 3);
}

TEST(SpaceSpecTest, JsonRoundTrip) {
    SpaceSpec original = BoxSpace{
        .shape = {8},
        .low = std::vector<float>(8, -1.0f),
        .high = std::vector<float>(8, 1.0f),
    };

    nlohmann::json j;
    to_json(j, original);

    SpaceSpec loaded;
    from_json(j, loaded);

    ASSERT_TRUE(std::holds_alternative<BoxSpace>(loaded));
    auto& box = std::get<BoxSpace>(loaded);
    EXPECT_EQ(box.shape[0], 8);
    EXPECT_EQ(box.low.size(), 8);
    EXPECT_FLOAT_EQ(box.low[0], -1.0f);
}

TEST(SpaceSpecTest, DiscreteJsonRoundTrip) {
    SpaceSpec original = DiscreteSpace{.n = 4};

    nlohmann::json j;
    to_json(j, original);

    SpaceSpec loaded;
    from_json(j, loaded);

    ASSERT_TRUE(std::holds_alternative<DiscreteSpace>(loaded));
    EXPECT_EQ(std::get<DiscreteSpace>(loaded).n, 4);
}

// Concept check (compile-time verification)
class MockEnv : public IEnvironment {
public:
    SpaceSpec observation_space() const override { return BoxSpace{.shape = {4}}; }
    SpaceSpec action_space() const override { return DiscreteSpace{.n = 2}; }
    Tensor reset(int32_t) override { return Tensor::from_flat({0, 0, 0, 0}); }
    StepResult step(const Action&) override {
        return {Tensor::from_flat({0, 0, 0, 0}), 1.0f, false, false, {}};
    }
};

static_assert(Environment<MockEnv>, "MockEnv must satisfy Environment concept");

TEST(EnvironmentTest, MockEnvWorks) {
    MockEnv env;
    auto obs = env.reset(-1);
    EXPECT_EQ(obs.numel(), 4);

    auto result = env.step(Action::discrete(0));
    EXPECT_FLOAT_EQ(result.reward, 1.0f);
    EXPECT_FALSE(result.terminated);
}

TEST(VectorizedEnvTest, Basic) {
    VectorizedEnv vec_env(
        []() { return std::make_unique<MockEnv>(); },
        4
    );

    EXPECT_EQ(vec_env.num_envs(), 4);

    auto obs = vec_env.reset_all();
    EXPECT_EQ(obs.size(), 4);

    std::vector<Action> actions(4, Action::discrete(0));
    auto results = vec_env.step_all(actions);
    EXPECT_EQ(results.size(), 4);
}
