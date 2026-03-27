#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#include <torch/extension.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "utils.h"

namespace omni_xpu {
namespace linear {

namespace {

using ST = torch::ScalarType;
using DT = dnnl::memory::data_type;

struct FP8CacheKey {
    int device_index;
    int input_type;
    int64_t m;
    int64_t k;
    int64_t n;
    bool has_bias;

    bool operator==(const FP8CacheKey& other) const {
        return device_index == other.device_index
            && input_type == other.input_type
            && m == other.m
            && k == other.k
            && n == other.n
            && has_bias == other.has_bias;
    }
};

struct FP8CacheKeyHash {
    size_t operator()(const FP8CacheKey& key) const {
        size_t seed = 0;
        auto combine = [&](size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };

        combine(std::hash<int>{}(key.device_index));
        combine(std::hash<int>{}(key.input_type));
        combine(std::hash<int64_t>{}(key.m));
        combine(std::hash<int64_t>{}(key.k));
        combine(std::hash<int64_t>{}(key.n));
        combine(std::hash<bool>{}(key.has_bias));
        return seed;
    }
};

struct FP8PrimitiveState {
    dnnl::engine engine;
    dnnl::memory::desc x_md;
    dnnl::memory::desc w_md;
    dnnl::memory::desc scales_md;
    dnnl::memory::desc out_md;
    dnnl::memory::desc bias_md;
    dnnl::matmul primitive;

    FP8PrimitiveState(
        dnnl::engine engine,
        dnnl::memory::desc x_md,
        dnnl::memory::desc w_md,
        dnnl::memory::desc scales_md,
        dnnl::memory::desc out_md,
        dnnl::memory::desc bias_md,
        dnnl::matmul::primitive_desc pd
    ) : engine(std::move(engine)),
        x_md(std::move(x_md)),
        w_md(std::move(w_md)),
        scales_md(std::move(scales_md)),
        out_md(std::move(out_md)),
        bias_md(std::move(bias_md)),
        primitive(std::move(pd)) {}
};

struct FP8CacheCounters {
    int64_t hits = 0;
    int64_t misses = 0;
};

std::mutex& fp8_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

FP8CacheCounters& fp8_cache_counters() {
    static FP8CacheCounters counters;
    return counters;
}

std::unordered_map<FP8CacheKey, std::shared_ptr<FP8PrimitiveState>, FP8CacheKeyHash>& fp8_primitive_cache() {
    static std::unordered_map<FP8CacheKey, std::shared_ptr<FP8PrimitiveState>, FP8CacheKeyHash> cache;
    return cache;
}

bool fp8_debug_enabled() {
    return std::getenv("OMNI_FP8_DEBUG") != nullptr;
}

int fp8_diag_stage() {
    const char* value = std::getenv("OMNI_FP8_DIAG_STAGE");
    return value ? std::atoi(value) : 0;
}

void fp8_debug_log(const char* message) {
    if (fp8_debug_enabled()) {
        std::printf("[omni_xpu_kernel::linear::onednn_w8a16_fp8][debug] %s\n", message);
        std::fflush(stdout);
    }
}

torch::Tensor materialize_tensor_if_needed(const torch::Tensor& tensor, const char* name) {
    fp8_debug_log("materialize_tensor_if_needed: begin");
    try {
        (void)tensor.data_ptr();
        fp8_debug_log("materialize_tensor_if_needed: original tensor has accessible data_ptr");
        return tensor;
    } catch (const c10::Error& original_error) {
        fp8_debug_log("materialize_tensor_if_needed: cloning tensor");
        try {
            auto cloned = tensor.contiguous().clone();
            (void)cloned.data_ptr();
            fp8_debug_log("materialize_tensor_if_needed: clone has accessible data_ptr");
            return cloned;
        } catch (const c10::Error& clone_error) {
            TORCH_CHECK(
                false,
                name,
                " could not be materialized for data_ptr access. original error: ",
                original_error.what(),
                "; clone error: ",
                clone_error.what()
            );
        }
    }
}

void* checked_data_ptr(const torch::Tensor& tensor, const char* name) {
    try {
        return tensor.data_ptr();
    } catch (const c10::Error& e) {
        TORCH_CHECK(false, name, " data_ptr() failed: ", e.what());
    }
}

torch::Tensor allocate_output_with_storage(
    at::IntArrayRef sizes,
    const torch::TensorOptions& options
) {
    fp8_debug_log("allocate_output_with_storage: begin");
    auto output = torch::empty(sizes, options);
    try {
        (void)output.data_ptr();
        fp8_debug_log("allocate_output_with_storage: empty output has accessible data_ptr");
        return output;
    } catch (const c10::Error& empty_error) {
        fp8_debug_log("allocate_output_with_storage: falling back to zeros");
        try {
            auto zeros = torch::zeros(sizes, options);
            (void)zeros.data_ptr();
            fp8_debug_log("allocate_output_with_storage: zeros output has accessible data_ptr");
            return zeros;
        } catch (const c10::Error& zeros_error) {
            TORCH_CHECK(
                false,
                "output could not be materialized for data_ptr access. empty error: ",
                empty_error.what(),
                "; zeros error: ",
                zeros_error.what()
            );
        }
    }
}

std::optional<int64_t> select_chunk_n_for_shape(
    int64_t m,
    int64_t k,
    int64_t n,
    torch::ScalarType input_type
) {
    if (m == 4096 && k == 4096) {
        if (n == 12288) {
            return int64_t{4096};
        }

        if (input_type == ST::BFloat16 && n == 24576) {
            return int64_t{4096};
        }
    }

    if (m == 4608 && k == 4096) {
        if (n == 16384) {
            if (input_type == ST::Half) {
                return int64_t{512};
            }
            return int64_t{4096};
        }

        if (input_type == ST::BFloat16 && n == 36864) {
            return int64_t{4096};
        }
    }

    if (input_type == ST::BFloat16) {
        if (m == 4096 && k == 12288 && n == 4096) {
            return int64_t{2048};
        }

        if (m == 4608 && k == 16384 && n == 4096) {
            return int64_t{1024};
        }
    }

    return std::nullopt;
}

void validate_xpu_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::XPU,
        name,
        " must be an XPU tensor"
    );
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <DT InputType>
std::shared_ptr<FP8PrimitiveState> get_or_create_fp8_primitive_state(
    int64_t m,
    int64_t k,
    int64_t n,
    bool has_bias,
    const torch::Device& device,
    const sycl::queue& queue,
    DT weight_type = DT::f8_e4m3
) {
    FP8CacheKey key{
        device.index(),
        static_cast<int>(InputType) * 100 + static_cast<int>(weight_type),  // unique per input+weight type combo
        m,
        k,
        n,
        has_bias,
    };

    auto& cache = fp8_primitive_cache();
    auto& counters = fp8_cache_counters();
    std::lock_guard<std::mutex> lock(fp8_cache_mutex());

    auto it = cache.find(key);
    if (it != cache.end()) {
        ++counters.hits;
        return it->second;
    }

    dnnl::engine engine = dnnl::sycl_interop::make_engine(
        queue.get_device(),
        queue.get_context()
    );
    dnnl::memory::desc x_md({m, k}, InputType, dnnl::memory::format_tag::ab);
    dnnl::memory::desc w_md({k, n}, weight_type, dnnl::memory::format_tag::ba);
    dnnl::memory::desc scales_md({n}, DT::f32, dnnl::memory::format_tag::a);
    dnnl::memory::desc out_md({m, n}, InputType, dnnl::memory::format_tag::ab);
    dnnl::memory::desc bias_md({1, n}, InputType, dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd = has_bias
        ? dnnl::matmul::primitive_desc(engine, x_md, w_md, bias_md, out_md, attr)
        : dnnl::matmul::primitive_desc(engine, x_md, w_md, out_md, attr);

    const std::string impl = pd.impl_info_str();
    if (impl.find("ref") != std::string::npos) {
        std::printf(
            "[omni_xpu_kernel::linear::onednn_w8a16_fp8] WARNING: oneDNN selected reference implementation: %s\n",
            impl.c_str()
        );
    }

    auto state = std::make_shared<FP8PrimitiveState>(
        std::move(engine),
        std::move(x_md),
        std::move(w_md),
        std::move(scales_md),
        std::move(out_md),
        std::move(bias_md),
        std::move(pd)
    );
    cache.emplace(key, state);
    ++counters.misses;
    return state;
}

template <DT InputType>
void onednn_w8a16_fp8_impl(
    void* x,
    void* weight,
    void* scales,
    void* bias,
    void* output,
    int64_t m,
    int64_t k,
    int64_t n,
    const torch::Device& device,
    DT weight_type = DT::f8_e4m3
) {
    sycl::queue& queue = utils::get_queue(device);
    auto state = get_or_create_fp8_primitive_state<InputType>(m, k, n, bias != nullptr, device, queue, weight_type);
    dnnl::stream stream = dnnl::sycl_interop::make_stream(state->engine, queue);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(state->x_md, state->engine, x)},
        {DNNL_ARG_WEIGHTS, dnnl::memory(state->w_md, state->engine, weight)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(state->scales_md, state->engine, scales)},
        {DNNL_ARG_DST, dnnl::memory(state->out_md, state->engine, output)},
    };

    if (bias != nullptr) {
        args.emplace(DNNL_ARG_BIAS, dnnl::memory(state->bias_md, state->engine, bias));
    }

    state->primitive.execute(stream, args);
}

}  // namespace

void fp8_cache_clear() {
    std::lock_guard<std::mutex> lock(fp8_cache_mutex());
    fp8_primitive_cache().clear();
    fp8_cache_counters() = {};
}

std::tuple<int64_t, int64_t, int64_t> fp8_cache_stats() {
    std::lock_guard<std::mutex> lock(fp8_cache_mutex());
    const auto& counters = fp8_cache_counters();
    return {
        counters.hits,
        counters.misses,
        static_cast<int64_t>(fp8_primitive_cache().size()),
    };
}

torch::Tensor onednn_w8a16_fp8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    std::optional<torch::Tensor> bias
) {
    validate_xpu_tensor(x, "x");
    validate_xpu_tensor(weight, "weight");
    validate_xpu_tensor(scales, "scales");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");
    TORCH_CHECK(scales.dim() == 1, "scales must be 1D [N]");

    TORCH_CHECK(
        x.scalar_type() == ST::Half || x.scalar_type() == ST::BFloat16,
        "x must have dtype float16 or bfloat16"
    );
    TORCH_CHECK(
        weight.scalar_type() == ST::Float8_e4m3fn || weight.scalar_type() == ST::Float8_e5m2,
        "weight must have dtype float8_e4m3fn or float8_e5m2"
    );
    TORCH_CHECK(scales.scalar_type() == ST::Float, "scales must have dtype float32");

    TORCH_CHECK(weight.size(1) == x.size(1),
        "weight shape must be [N, K] with K matching x; got weight.shape=",
        weight.sizes(),
        " and x.shape=",
        x.sizes());
    TORCH_CHECK(scales.size(0) == weight.size(0),
        "scales shape must be [N] with N matching weight.shape[0]; got scales.shape=",
        scales.sizes(),
        " and weight.shape=",
        weight.sizes());

    TORCH_CHECK(weight.device() == x.device(), "weight must be on the same XPU device as x");
    TORCH_CHECK(scales.device() == x.device(), "scales must be on the same XPU device as x");

    if (bias.has_value()) {
        validate_xpu_tensor(*bias, "bias");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D [N]");
        TORCH_CHECK(bias->size(0) == weight.size(0),
            "bias shape must be [N] with N matching weight.shape[0]; got bias.shape=",
            bias->sizes(),
            " and weight.shape=",
            weight.sizes());
        TORCH_CHECK(bias->device() == x.device(), "bias must be on the same XPU device as x");
        TORCH_CHECK(bias->scalar_type() == x.scalar_type(), "bias dtype must match x dtype");
    }

    auto x_materialized = materialize_tensor_if_needed(x, "x");
    auto weight_materialized = materialize_tensor_if_needed(weight, "weight");
    auto scales_materialized = materialize_tensor_if_needed(scales, "scales");

    if (fp8_diag_stage() == 1) {
        return torch::zeros({x_materialized.size(0), weight_materialized.size(0)}, x_materialized.options());
    }

    std::optional<torch::Tensor> bias_materialized;
    if (bias.has_value()) {
        bias_materialized = materialize_tensor_if_needed(*bias, "bias");
    }

    if (fp8_diag_stage() == 2) {
        return torch::zeros({x_materialized.size(0), weight_materialized.size(0)}, x_materialized.options());
    }

    const int64_t m = x_materialized.size(0);
    const int64_t k = x_materialized.size(1);
    const int64_t n = weight_materialized.size(0);

    torch::Tensor output = allocate_output_with_storage({m, n}, x_materialized.options());

    if (fp8_diag_stage() == 3) {
        return output;
    }

    // Determine oneDNN weight data type from PyTorch scalar type
    DT wt = (weight_materialized.scalar_type() == ST::Float8_e5m2) ? DT::f8_e5m2 : DT::f8_e4m3;

    auto dispatch = [&](auto fn, void* x_ptr, void* w_ptr, void* s_ptr, void* b_ptr, void* out_ptr, int64_t m_, int64_t k_, int64_t n_) {
        fn(
            x_ptr,
            w_ptr,
            s_ptr,
            b_ptr,
            out_ptr,
            m_,
            k_,
            n_,
            x_materialized.device(),
            wt
        );
    };

    if (fp8_diag_stage() == 4) {
        checked_data_ptr(x_materialized, "x_materialized");
        checked_data_ptr(weight_materialized, "weight_materialized");
        checked_data_ptr(scales_materialized, "scales_materialized");
        checked_data_ptr(output, "output");
        return output;
    }

    auto selected_chunk_n = select_chunk_n_for_shape(m, k, n, x_materialized.scalar_type());

    if (selected_chunk_n.has_value()) {
        int64_t chunk_n = *selected_chunk_n;
        for (int64_t j = 0; j < n; j += chunk_n) {
            int64_t current_n = std::min(chunk_n, n - j);
            
            auto w_chunk = weight_materialized.slice(0, j, j + current_n);
            auto s_chunk = scales_materialized.slice(0, j, j + current_n);
            std::optional<torch::Tensor> b_chunk;
            if (bias_materialized.has_value()) {
                b_chunk = bias_materialized->slice(0, j, j + current_n);
            }
            torch::Tensor out_chunk = allocate_output_with_storage({m, current_n}, x_materialized.options());

            auto do_dispatch = [&](auto fn) {
                dispatch(fn, 
                    checked_data_ptr(x_materialized, "x_materialized"),
                    checked_data_ptr(w_chunk, "w_chunk"),
                    checked_data_ptr(s_chunk, "s_chunk"),
                    b_chunk.has_value() ? checked_data_ptr(*b_chunk, "b_chunk") : nullptr,
                    checked_data_ptr(out_chunk, "out_chunk"),
                    m, k, current_n);
            };

            switch (x_materialized.scalar_type()) {
                case ST::Half: do_dispatch(onednn_w8a16_fp8_impl<DT::f16>); break;
                case ST::BFloat16: do_dispatch(onednn_w8a16_fp8_impl<DT::bf16>); break;
                default: TORCH_CHECK(false, "x must have dtype float16 or bfloat16");
            }

            output.slice(1, j, j + current_n).copy_(out_chunk);
        }
    } else {
        auto do_dispatch = [&](auto fn) {
            dispatch(fn, 
                checked_data_ptr(x_materialized, "x_materialized"),
                checked_data_ptr(weight_materialized, "weight_materialized"),
                checked_data_ptr(scales_materialized, "scales_materialized"),
                bias_materialized.has_value() ? checked_data_ptr(*bias_materialized, "bias_materialized") : nullptr,
                checked_data_ptr(output, "output"),
                m, k, n);
        };

        switch (x_materialized.scalar_type()) {
            case ST::Half: do_dispatch(onednn_w8a16_fp8_impl<DT::f16>); break;
            case ST::BFloat16: do_dispatch(onednn_w8a16_fp8_impl<DT::bf16>); break;
            default: TORCH_CHECK(false, "x must have dtype float16 or bfloat16");
        }
    }

    return output;
}

}  // namespace linear
}  // namespace omni_xpu
