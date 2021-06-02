#include <ATen/detail/HPUHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const HPUHooksInterface& getHPUHooks() {
  static std::unique_ptr<HPUHooksInterface> hpu_hooks;
  static std::once_flag once;
  std::call_once(once, [] {
    hpu_hooks = HPUHooksRegistry()->Create("HPUHooks", HPUHooksArgs{});
    if (!hpu_hooks) {
      hpu_hooks = std::unique_ptr<HPUHooksInterface>(new HPUHooksInterface());
    }
  });
  return *hpu_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(HPUHooksRegistry, HPUHooksInterface, HPUHooksArgs)
} // namespace at
