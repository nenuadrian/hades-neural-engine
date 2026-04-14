#pragma once

// Core (always available, no LibTorch dependency)
#include "core/types.hpp"
#include "core/environment.hpp"
#include "core/agent.hpp"

// Training (requires HNE_TRAINING define and LibTorch)
#include "core/policy.hpp"
#include "core/algorithm.hpp"
#include "core/replay_buffer.hpp"
#include "algorithms/ppo.hpp"
#include "training/trainer_config.hpp"
#include "training/metrics.hpp"
#include "training/callbacks.hpp"
#include "training/checkpoint.hpp"
#include "training/trainer.hpp"

// Inference (requires HNE_INFERENCE define and LibTorch CPU)
#include "inference/runtime.hpp"
#include "inference/batch_evaluator.hpp"
