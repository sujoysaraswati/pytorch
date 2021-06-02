#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HPUHooksInterface.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {

bool is_pinned(const Tensor& self) {
  auto pinned = detail::getCUDAHooks().isPinnedPtr(self.storage().data());
  if (detail::getHPUHooks().hasHPU()) {
    pinned = detail::getHPUHooks().isPinnedPtr(self.storage().data());
  }
  return pinned;
}

Tensor pin_memory(const Tensor& self) {
  if (!self.device().is_cpu()) {
    AT_ERROR("cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  }
  if (self.is_pinned()) {
    return self;
  }

  c10::Allocator* allocator = nullptr;
  if (detail::getCUDAHooks().hasCUDA()) {
    allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  } else if (detail::getHPUHooks().hasHPU()) {
    allocator = detail::getHPUHooks().getPinnedMemoryAllocator();
  }

  if (!allocator) {
    AT_ERROR("cannot pin '", self.toString());
  }

  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

}
}
