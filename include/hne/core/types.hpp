#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>
#include <nlohmann/json.hpp>

namespace hne {

// ── Space descriptors ──────────────────────────────────────────────────────
// Define the shape/type of observations and actions without holding data.

enum class SpaceType : uint8_t {
    Discrete,
    MultiDiscrete,
    Box
};

struct DiscreteSpace {
    int32_t n = 0;
};

struct MultiDiscreteSpace {
    std::vector<int32_t> nvec;
};

struct BoxSpace {
    std::vector<int32_t> shape;
    std::vector<float> low;
    std::vector<float> high;
};

using SpaceSpec = std::variant<DiscreteSpace, MultiDiscreteSpace, BoxSpace>;

[[nodiscard]] int32_t flat_size(const SpaceSpec& spec);

// ── Tensor ─────────────────────────────────────────────────────────────────
// Lightweight float buffer used at the HNE boundary. Training internals
// convert to torch::Tensor; game code never touches LibTorch types.

struct Tensor {
    std::vector<float> data;
    std::vector<int32_t> shape;

    Tensor() = default;
    Tensor(std::vector<float> data, std::vector<int32_t> shape);

    static Tensor from_flat(std::vector<float> data);
    static Tensor scalar(float value);

    [[nodiscard]] int32_t numel() const;
    [[nodiscard]] std::span<const float> view() const;
    [[nodiscard]] std::span<float> mutable_view();

    float& operator[](int32_t i);
    float operator[](int32_t i) const;
};

// ── Action ─────────────────────────────────────────────────────────────────

struct Action {
    std::variant<int32_t, std::vector<float>> value;

    static Action discrete(int32_t action_id);
    static Action continuous(std::vector<float> values);

    [[nodiscard]] bool is_discrete() const;
    [[nodiscard]] int32_t as_discrete() const;
    [[nodiscard]] const std::vector<float>& as_continuous() const;
};

// ── StepResult ─────────────────────────────────────────────────────────────

struct StepResult {
    Tensor observation;
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
    nlohmann::json info;
};

// ── JSON serialization ─────────────────────────────────────────────────────

void to_json(nlohmann::json& j, const DiscreteSpace& s);
void from_json(const nlohmann::json& j, DiscreteSpace& s);
void to_json(nlohmann::json& j, const MultiDiscreteSpace& s);
void from_json(const nlohmann::json& j, MultiDiscreteSpace& s);
void to_json(nlohmann::json& j, const BoxSpace& s);
void from_json(const nlohmann::json& j, BoxSpace& s);
void to_json(nlohmann::json& j, const SpaceSpec& s);
void from_json(const nlohmann::json& j, SpaceSpec& s);

} // namespace hne
