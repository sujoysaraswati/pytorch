#include <torch/hpu.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
namespace hpu {
size_t device_count() {
  return at::detail::getHPUHooks().getNumHPUs();
}

bool is_available() {
  // NB: the semantics of this are different from hasHPU();
  // ATen's function tells you if you have a working driver and  build,
  // whereas this function also tells you if you actually have any HPUs.
  // This function matches the semantics of at::hpu::is_available()
  return at::detail::getHPUHooks().hasHPU() && hpu::device_count() > 0;
}

} // namespace hpu
} // namespace torch
