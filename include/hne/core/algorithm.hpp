#pragma once

#ifdef HNE_TRAINING

#include "policy.hpp"
#include <map>
#include <string>

namespace hne {

class RolloutBuffer; // forward

struct AlgorithmMetrics {
    std::map<std::string, float> scalars;
};

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;

    virtual AlgorithmMetrics update(IPolicy& policy, const RolloutBuffer& buffer) = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace hne

#endif // HNE_TRAINING
