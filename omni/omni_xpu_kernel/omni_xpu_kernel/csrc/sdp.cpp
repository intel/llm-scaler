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
    auto out = torch::empty_like(q);
    sycl::queue& queue = utils::get_queue(q.device());

    switch (q.scalar_type()) {
        case ST::Half:
            kernels.fp16(
                q.data_ptr(),
                k.data_ptr(),
                v.data_ptr(),
                norm_alpha.data_ptr(),
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
                v.data_ptr(),
                norm_alpha.data_ptr(),
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
