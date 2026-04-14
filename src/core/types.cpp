#include <hne/core/types.hpp>
#include <numeric>
#include <stdexcept>

namespace hne {

// ── flat_size ──────────────────────────────────────────────────────────────

int32_t flat_size(const SpaceSpec& spec) {
    return std::visit([](const auto& s) -> int32_t {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, DiscreteSpace>) {
            return 1;
        } else if constexpr (std::is_same_v<T, MultiDiscreteSpace>) {
            return static_cast<int32_t>(s.nvec.size());
        } else {
            return std::accumulate(s.shape.begin(), s.shape.end(),
                                   int32_t{1}, std::multiplies<>());
        }
    }, spec);
}

// ── Tensor ─────────────────────────────────────────────────────────────────

Tensor::Tensor(std::vector<float> d, std::vector<int32_t> s)
    : data(std::move(d)), shape(std::move(s)) {}

Tensor Tensor::from_flat(std::vector<float> d) {
    auto n = static_cast<int32_t>(d.size());
    return Tensor(std::move(d), {n});
}

Tensor Tensor::scalar(float value) {
    return Tensor({value}, {1});
}

int32_t Tensor::numel() const {
    return static_cast<int32_t>(data.size());
}

std::span<const float> Tensor::view() const {
    return {data.data(), data.size()};
}

std::span<float> Tensor::mutable_view() {
    return {data.data(), data.size()};
}

float& Tensor::operator[](int32_t i) { return data[i]; }
float Tensor::operator[](int32_t i) const { return data[i]; }

// ── Action ─────────────────────────────────────────────────────────────────

Action Action::discrete(int32_t action_id) {
    return Action{.value = action_id};
}

Action Action::continuous(std::vector<float> values) {
    return Action{.value = std::move(values)};
}

bool Action::is_discrete() const {
    return std::holds_alternative<int32_t>(value);
}

int32_t Action::as_discrete() const {
    return std::get<int32_t>(value);
}

const std::vector<float>& Action::as_continuous() const {
    return std::get<std::vector<float>>(value);
}

// ── JSON ───────────────────────────────────────────────────────────────────

void to_json(nlohmann::json& j, const DiscreteSpace& s) {
    j = {{"type", "discrete"}, {"n", s.n}};
}

void from_json(const nlohmann::json& j, DiscreteSpace& s) {
    j.at("n").get_to(s.n);
}

void to_json(nlohmann::json& j, const MultiDiscreteSpace& s) {
    j = {{"type", "multi_discrete"}, {"nvec", s.nvec}};
}

void from_json(const nlohmann::json& j, MultiDiscreteSpace& s) {
    j.at("nvec").get_to(s.nvec);
}

void to_json(nlohmann::json& j, const BoxSpace& s) {
    j = {{"type", "box"}, {"shape", s.shape}, {"low", s.low}, {"high", s.high}};
}

void from_json(const nlohmann::json& j, BoxSpace& s) {
    j.at("shape").get_to(s.shape);
    j.at("low").get_to(s.low);
    j.at("high").get_to(s.high);
}

void to_json(nlohmann::json& j, const SpaceSpec& s) {
    std::visit([&j](const auto& v) { to_json(j, v); }, s);
}

void from_json(const nlohmann::json& j, SpaceSpec& s) {
    auto type = j.at("type").get<std::string>();
    if (type == "discrete") {
        DiscreteSpace ds;
        from_json(j, ds);
        s = ds;
    } else if (type == "multi_discrete") {
        MultiDiscreteSpace mds;
        from_json(j, mds);
        s = mds;
    } else if (type == "box") {
        BoxSpace bs;
        from_json(j, bs);
        s = bs;
    } else {
        throw std::runtime_error("Unknown SpaceSpec type: " + type);
    }
}

} // namespace hne
