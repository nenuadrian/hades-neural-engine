#pragma once

#include "types.hpp"
#include <cstdint>

namespace hne {

class IAgent {
public:
    virtual ~IAgent() = default;

    [[nodiscard]] virtual Action act(const Tensor& observation,
                                     bool deterministic = true) = 0;

    virtual void reset() {}

    [[nodiscard]] virtual uint32_t id() const = 0;
};

} // namespace hne
