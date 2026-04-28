#pragma once
#include <memory>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUFunctions.h>
#include <sycl/sycl.hpp>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// Xe2 2D block load/store requires the contiguous (stride==1) dimension
// to be 16-byte aligned. For the flash-attention tensors here that is
// the last dim (head_size), so we only need to check
// size(-1) * element_size % 16 == 0.
#define CHECK_STRIDE_ALIGNMENT(x)                                      \
  do {                                                                 \
    const int _es = (x).element_size();                                \
    const auto _last = (x).size(-1);                                   \
    TORCH_CHECK(                                                       \
        (x).stride(-1) == 1,                                           \
        #x " expects the last dim to be contiguous (stride==1), got ", \
        (x).stride(-1));                                               \
    TORCH_CHECK(                                                       \
        _last * _es % 16 == 0,                                         \
        #x " last dim size=",                                          \
        _last,                                                         \
        " (",                                                          \
        _last * _es,                                                   \
        " bytes) is not 16-byte aligned (element_size=",               \
        _es,                                                           \
        "). Xe2 2D block requires the contiguous dim to be 16-byte "   \
        "aligned.");                                                   \
  } while (0)

namespace vllm {
namespace xpu {

static inline sycl::queue& vllmGetQueue(at::DeviceIndex device_index = -1) {
  auto current_stream = c10::xpu::getCurrentXPUStream(device_index);
  auto& queue = current_stream.queue();
  return queue;
}

namespace syclex = sycl::ext::oneapi::experimental;

static inline syclex::architecture
get_device_architecture(at::DeviceIndex device_index = -1) {
  auto device_id =
      (device_index == -1) ? c10::xpu::current_device() : device_index;
  auto raw_device = c10::xpu::get_raw_device(device_id);
  return raw_device.get_info<syclex::info::device::architecture>();
}

static inline bool is_bmg(at::DeviceIndex device_index = -1) {
  return get_device_architecture(device_index) ==
             syclex::architecture::intel_gpu_bmg_g21 ||
         get_device_architecture(device_index) ==
             syclex::architecture::intel_gpu_bmg_g31;
}

static inline bool is_pvc(at::DeviceIndex device_index = -1) {
  return get_device_architecture(device_index) ==
         syclex::architecture::intel_gpu_pvc;
}

static inline bool is_xe2_arch(at::DeviceIndex device_index = -1) {
  auto arch = get_device_architecture(device_index);
  return arch == syclex::architecture::intel_gpu_bmg_g21 ||
         arch == syclex::architecture::intel_gpu_bmg_g31 ||
         arch == syclex::architecture::intel_gpu_lnl_m ||
         arch == syclex::architecture::intel_gpu_pvc;
}

static inline bool is_xe3_arch(at::DeviceIndex device_index = -1) {
  auto arch = get_device_architecture(device_index);
  return arch == syclex::architecture::intel_gpu_ptl_h ||
         arch == syclex::architecture::intel_gpu_ptl_u ||
         arch == syclex::architecture::intel_gpu_wcl;
}

static inline std::optional<std::string> getEnv(const char* name) {
  if (const char* val = std::getenv(name)) return val;
  return std::nullopt;
}

static inline bool force_xe_default_kernel() {
  auto env_val = getEnv("VLLM_XPU_FORCE_XE_DEFAULT_KERNEL");
  if (env_val.has_value()) {
    return env_val.value() == "1" || env_val.value() == "true" ||
           env_val.value() == "TRUE";
  }
  return false;
}

template <typename T>
struct SyclTypeTrait {
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half> {
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

template <typename T>
struct AccumulateType {
 private:
  static constexpr bool is_narrow_float =
      std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16> ||
      std::is_same_v<T, c10::Float8_e4m3fn> ||
      std::is_same_v<T, c10::Float8_e5m2>;

  static constexpr bool is_integer =
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
      std::is_same_v<T, char> || std::is_same_v<T, int16_t> ||
      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

  static constexpr bool is_complex = std::is_same_v<T, c10::complex<float>> ||
                                     std::is_same_v<T, c10::complex<double>>;

 public:
  using type = std::conditional_t<
      is_narrow_float,
      float,
      std::conditional_t<
          std::is_floating_point_v<T>,
          T,
          std::conditional_t<
              is_integer,
              int64_t,
              std::conditional_t<is_complex, T, T>>>>;
};

template <typename T>
using acc_type = typename AccumulateType<T>::type;

// aligned vector generates vectorized load/store on XPU
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vec {
  scalar_t val[vec_size];

  scalar_t& operator[](int index) { return val[index]; }

  scalar_t const& operator[](int index) const { return val[index]; }
};

// From float to float.
inline void from_float(float& dst, float src) { dst = src; }
// From float32 to float16.
inline void from_float(sycl::half& dst, float src) { dst = sycl::half(src); }
// From float32 to bfloat16.
inline void from_float(sycl::ext::oneapi::bfloat16& dst, float src) {
  dst = sycl::ext::oneapi::bfloat16(src);
}

// From float to float.
inline float to_float(float u) { return u; }
// From float16 to float32.
inline float to_float(sycl::half u) { return float(u); }
// From bfloat16 to float32.
inline float to_float(sycl::ext::oneapi::bfloat16 u) { return float(u); }

}  // namespace xpu

}  // namespace vllm
