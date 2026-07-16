#pragma once

#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <memory>

#include <oneapi/dnnl/dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace oneDNN {

static inline dnnl::memory
make_onednn_memory(dnnl::memory::desc md, dnnl::engine& engine, void* ptr) {
  return dnnl::sycl_interop::make_memory(
      md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      ptr == nullptr ? DNNL_MEMORY_ALLOCATE : ptr);
}

// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager& Instance();  // Singleton

  dnnl::engine&
  get_engine(at::DeviceIndex device_index = c10::xpu::current_device()) {
    c10::xpu::check_device_index(device_index);
    return *engine_pool[device_index];
  }

  dnnl::engine& get_engine(const at::Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == at::kXPU);
    return get_engine(device.index());
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;
  GpuEngineManager(GpuEngineManager&&) = default;
  GpuEngineManager& operator=(GpuEngineManager&&) = default;

 protected:
  GpuEngineManager() {
    int device_count = c10::xpu::device_count_ensure_non_zero();
    for (int i = 0; i < device_count; i++) {
      engine_pool.push_back(
          std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(
              at::xpu::get_raw_device(i), at::xpu::get_device_context())));
    }
  }
  ~GpuEngineManager() = default;

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct GpuStreamManager {
  static GpuStreamManager& Instance();  // Singleton

  dnnl::stream&
  get_stream(at::DeviceIndex device_index = c10::xpu::current_device()) {
    auto stream = c10::xpu::getCurrentXPUStream(device_index);
    auto priority = stream.priority();
    if (stream_pool[device_index][priority].find(stream) ==
        stream_pool[device_index][priority].end()) {
      stream_pool[device_index][priority][stream] =
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine(device_index),
              stream.queue()));
    }
    return *stream_pool[device_index][priority][stream];
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;
  GpuStreamManager(GpuStreamManager&&) = default;
  GpuStreamManager& operator=(GpuStreamManager&&) = default;

 protected:
  GpuStreamManager() {
    int device_count = c10::xpu::device_count_ensure_non_zero();
    stream_pool.resize(device_count);
  }
  ~GpuStreamManager() = default;

 private:
  using stream_hash_map =
      ska::flat_hash_map<c10::xpu::XPUStream, std::shared_ptr<dnnl::stream>>;
  std::vector<
      std::array<stream_hash_map, c10::xpu::max_compile_time_stream_priorities>>
      stream_pool;
};

}  // namespace oneDNN
