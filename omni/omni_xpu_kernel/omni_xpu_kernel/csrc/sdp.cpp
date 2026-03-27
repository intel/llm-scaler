// ============================================================================
// SDP (Scaled Dot-Product Attention) - ESIMD Flash Attention via lgrf sidecar
// ============================================================================
// Loads the pre-compiled ESIMD Flash Attention kernel from a sidecar shared
// library (lgrf_sdp.so / lgrf_sdp.pyd) built with doubleGRF for Xe2 ISA.
//
// The sidecar exports two C functions:
//   sdp_fp16  — FP16 optimized Flash Attention
//   sdp_bf16io — BF16 I/O hybrid (bf16 QK DPAS + fp16 SxV DPAS)
//
// Input layout: [B, L, H, 128] contiguous, B==1, head_dim==128
// ============================================================================

#include <atomic>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <torch/extension.h>

#include "utils.h"

namespace omni_xpu {
namespace sdp {

using ST = torch::ScalarType;

namespace {

using sdp_kernel_fn = void (*)(
    void* Q,
    void* K,
    void* V,
    void* normAlpha,
    void* out,
    int q_len,
    int kv_len,
    int headQ,
    int headKv,
    void* sycl_queue_ptr);

struct KernelLibrary {
#ifdef _WIN32
    HMODULE handle{nullptr};
#else
    void* handle{nullptr};
#endif
    sdp_kernel_fn fp16{nullptr};
    sdp_kernel_fn bf16io{nullptr};
};

KernelLibrary& get_kernel_library() {
    static KernelLibrary library;
    static std::once_flag load_once;
    static std::string load_error;

    std::call_once(load_once, []() {
        namespace fs = std::filesystem;
        fs::path package_dir;

#ifdef _WIN32
        HMODULE current_module = nullptr;
        if (!GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&get_kernel_library),
                &current_module)) {
            load_error = "failed to resolve the omni_xpu_kernel extension path while locating the lgrf sidecar";
            return;
        }

        std::wstring path_buffer(MAX_PATH, L'\0');
        const DWORD path_length = GetModuleFileNameW(current_module, path_buffer.data(), static_cast<DWORD>(path_buffer.size()));
        if (path_length == 0) {
            load_error = "failed to read the omni_xpu_kernel extension path while locating the lgrf sidecar";
            return;
        }

        path_buffer.resize(path_length);
        package_dir = fs::path(path_buffer).parent_path();
        fs::path library_path = package_dir / "lgrf_uni" / "lgrf_sdp.pyd";
        library.handle = LoadLibraryW(library_path.wstring().c_str());
        if (library.handle == nullptr) {
            load_error = "failed to load lgrf sidecar at " + library_path.string();
            return;
        }

        library.fp16 = reinterpret_cast<sdp_kernel_fn>(GetProcAddress(library.handle, "sdp_fp16"));
        library.bf16io = reinterpret_cast<sdp_kernel_fn>(GetProcAddress(library.handle, "sdp_bf16io"));
#else
        Dl_info current_module_info;
        if (dladdr(reinterpret_cast<void*>(&get_kernel_library), &current_module_info) == 0 || current_module_info.dli_fname == nullptr) {
            load_error = "failed to resolve the omni_xpu_kernel extension path while locating the lgrf sidecar";
            return;
        }

        package_dir = fs::path(current_module_info.dli_fname).parent_path();
        fs::path library_dir = package_dir / "lgrf_uni";
        TORCH_CHECK(fs::exists(library_dir), "missing lgrf sidecar directory: ", library_dir.string());

        fs::path library_path;
        for (const auto& entry : fs::directory_iterator(library_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const auto name = entry.path().filename().string();
            if (name.rfind("lgrf_sdp", 0) == 0 && entry.path().extension() == ".so") {
                library_path = entry.path();
                break;
            }
        }

        if (!fs::exists(library_path)) {
            load_error = "missing lgrf sidecar artifact under " + library_dir.string();
            return;
        }

        library.handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library.handle == nullptr) {
            load_error = std::string("failed to load lgrf sidecar at ") + library_path.string() + ": " + dlerror();
            return;
        }

        library.fp16 = reinterpret_cast<sdp_kernel_fn>(dlsym(library.handle, "sdp_fp16"));
        library.bf16io = reinterpret_cast<sdp_kernel_fn>(dlsym(library.handle, "sdp_bf16io"));
#endif

        if (library.fp16 == nullptr || library.bf16io == nullptr) {
            load_error = "failed to resolve sdp_fp16/sdp_bf16io from lgrf sidecar";
        }
    });

    TORCH_CHECK(load_error.empty(), load_error);
    return library;
}

torch::Tensor& norm_alpha_cache(const torch::Tensor& q) {
    static std::mutex cache_mutex;
    static torch::Tensor cached_norm_alpha;
    static c10::Device cached_device{c10::DeviceType::CPU};
    static int64_t cached_head_count = -1;

    const auto head_count = q.size(2);

    std::lock_guard<std::mutex> guard(cache_mutex);
    if (!cached_norm_alpha.defined() || cached_head_count != head_count || cached_device != q.device()) {
        cached_norm_alpha = torch::ones({head_count * 128}, torch::dtype(torch::kFloat).device(q.device()));
        cached_head_count = head_count;
        cached_device = q.device();
    }

    return cached_norm_alpha;
}
}

static void check_sdp_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::XPU, name, " must be on XPU");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 4, name, " must be 4-D [B, L, H, D]");
    TORCH_CHECK(t.size(0) == 1, name, " batch size must be 1");
    TORCH_CHECK(t.size(3) == 128, name, " head_dim must be 128");
    TORCH_CHECK(
        t.scalar_type() == ST::Half || t.scalar_type() == ST::BFloat16,
        name,
        " dtype must be FP16 or BF16"
    );
}

torch::Tensor sdp(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    check_sdp_tensor(q, "q");
    check_sdp_tensor(k, "k");
    check_sdp_tensor(v, "v");

    TORCH_CHECK(
        q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
        "q, k, v must have the same dtype"
    );
    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same sequence length");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v must have the same head count");
    TORCH_CHECK(q.size(2) == k.size(2), "q, k, v must have the same head count");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "q, k, v must have the same head_dim");

    auto& kernels = get_kernel_library();
    auto& norm_alpha = norm_alpha_cache(q);
    const int64_t H = q.size(2);

    // Adaptive V-scaling: only apply per-head V-scaling when V values are large
    // enough to risk fp16 accumulator overflow in the ESIMD kernel's S×V DPAS.
    //
    // The kernel accumulates: sum_i(softmax_weight_i * V_i) in fp16.
    // Worst case: all softmax weights concentrate on one V row → accumulator ≈ V_max.
    // With multiple tiles: accumulator can reach V_max * compensation_factor.
    // Safe threshold: max(|V|) < 256 (conservative, leaves 256x headroom to 65504).
    //
    // Fast path (V small): direct kernel call, no overhead.
    // Safe path (V large): per-head V-scaling with exact normAlpha cancellation.
    constexpr float V_SCALE_THRESHOLD = 256.0f;

    // Adaptive V-scaling with cached decision to avoid per-call host sync.
    //
    // .item() forces a GPU→CPU sync that costs ~0.5ms (13% overhead for flux-4096).
    // Since V magnitude is consistent within a model (determined by W_v weights),
    // we check once and cache the decision. Recheck every 500 calls as safety net.
    static std::atomic<int> sdp_call_counter{0};
    static std::atomic<bool> cached_needs_scaling{false};
    constexpr int RECHECK_INTERVAL = 500;

    int call_num = sdp_call_counter.fetch_add(1);
    bool needs_scaling;
    if (call_num % RECHECK_INTERVAL == 0) {
        float v_global_max = v.abs().max().item<float>();
        needs_scaling = (v_global_max >= V_SCALE_THRESHOLD);
        cached_needs_scaling.store(needs_scaling);
        OMNI_DEBUG("sdp", "call #%d: V_max=%.1f threshold=%.0f needs_scaling=%d q=[%ld,%ld,%ld,%ld]",
                   call_num, v_global_max, V_SCALE_THRESHOLD, needs_scaling,
                   q.size(0), q.size(1), q.size(2), q.size(3));
    } else {
        needs_scaling = cached_needs_scaling.load();
    }

    const void* v_ptr;
    const void* alpha_ptr;
    torch::Tensor v_scaled;       // keep alive if scaling
    torch::Tensor effective_alpha; // keep alive if scaling

    if (needs_scaling) {
        auto v_absmax = v.abs().amax(/*dim=*/{0, 1, 3});  // [H] per-head max
        auto v_scale = (v_absmax.to(torch::kFloat) / 32.0f).clamp_min(1.0f);  // [H] fp32

        v_scaled = v / v_scale.view({1, 1, H, 1}).to(v.scalar_type());
        effective_alpha = norm_alpha * v_scale.repeat_interleave(128);
        OMNI_DEBUG("sdp", "V-scaling applied: v_scale_max=%.2f V_scaled_max=%.1f",
                   v_scale.max().item<float>(), v_scaled.abs().max().item<float>());

        v_ptr = v_scaled.data_ptr();
        alpha_ptr = effective_alpha.data_ptr();
    } else {
        // Fast path: V values are small, no scaling needed
        v_ptr = v.data_ptr();
        alpha_ptr = norm_alpha.data_ptr();
    }

    auto out = torch::empty_like(q);
    sycl::queue& queue = utils::get_queue(q.device());

    switch (q.scalar_type()) {
        case ST::Half:
            kernels.fp16(
                q.data_ptr(),
                k.data_ptr(),
                const_cast<void*>(v_ptr),
                const_cast<void*>(alpha_ptr),
                out.data_ptr(),
                static_cast<int>(q.size(1)),
                static_cast<int>(k.size(1)),
                static_cast<int>(q.size(2)),
                static_cast<int>(k.size(2)),
                &queue);
            break;
        case ST::BFloat16:
            kernels.bf16io(
                q.data_ptr(),
                k.data_ptr(),
                const_cast<void*>(v_ptr),
                const_cast<void*>(alpha_ptr),
                out.data_ptr(),
                static_cast<int>(q.size(1)),
                static_cast<int>(k.size(1)),
                static_cast<int>(q.size(2)),
                static_cast<int>(k.size(2)),
                &queue);
            break;
        default:
            TORCH_CHECK(false, "sdp: unsupported dtype, only FP16 and BF16 are supported");
    }

    return out;
}

} // namespace sdp
} // namespace omni_xpu
