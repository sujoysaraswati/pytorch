#pragma once

#include <ATen/detail/HPUHooksInterface.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>

namespace torch {
namespace hpu {
/// Returns the number of HPU devices available.
size_t TORCH_API device_count();

/// Returns true if at least one HPU device is available.
bool TORCH_API is_available();

} // namespace hpu
} // namespace torch
