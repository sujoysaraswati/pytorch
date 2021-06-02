#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
class Context;
}

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {
// The HPUHooksInterface is an omnibus interface for any HPU functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of HPU code). See
// CUDAHooksInterface for more detailed motivation.
struct TORCH_API HPUHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~HPUHooksInterface() {}

  virtual bool isPinnedPtr(void* data) const {
    return false;
  }

  virtual bool hasHPU() const {
    return false;
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "Pinned memory requires HPU.");
  }

  virtual int getNumHPUs() const {
    return 0;
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct TORCH_API HPUHooksArgs {};

C10_DECLARE_REGISTRY(HPUHooksRegistry, HPUHooksInterface, HPUHooksArgs);
#define REGISTER_HPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(HPUHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const HPUHooksInterface& getHPUHooks();
} // namespace detail
} // namespace at
