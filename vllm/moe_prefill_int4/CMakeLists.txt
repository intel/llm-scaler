cmake_minimum_required(VERSION 3.18)
project(moe_prefill_gemm_int4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# DPC++ SYCL target — BMG
set(DPCPP_SYCL_TARGET "intel_gpu_bmg_g21" CACHE STRING "DPC++ SYCL target")

# cutlass-sycl headers (symlinked into csrc/)
set(CUTLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/csrc/cutlass_include")
set(CUTLASS_TOOLS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/csrc/cutlass_tools_include")
set(CUTLASS_APP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/csrc/cutlass_app_include")

# Compile flags matching vllm-xpu-kernels
set(SYCL_COMPILE_FLAGS
    -fsycl
    -fsycl-targets=spir64_gen
    -fPIC
    -O3
    -DCUTLASS_ENABLE_HEADERS_ONLY
    -DCUTLASS_ENABLE_SYCL
    -DSYCL_INTEL_TARGET
    -DCUTLASS_VERSIONS_GENERATED
    -DVLLM_XPU_ENABLE_XE2
    -ftemplate-backtrace-limit=0
    -fno-sycl-instrument-device-code
)

# Link flags — device code generation for BMG
set(SYCL_LINK_FLAGS
    -fsycl
    -fsycl-targets=spir64_gen
    -Xs "-device bmg"
    "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
    -flink-huge-device-code
)

# Source files — xe_2 grouped_gemm only
file(GLOB_RECURSE KERNEL_SOURCES "csrc/xe_2/*.cpp")
list(APPEND KERNEL_SOURCES "csrc/grouped_gemm_interface.cpp")
list(APPEND KERNEL_SOURCES "csrc/moe_forward_fused.cpp")

add_library(moe_prefill_gemm_int4 SHARED ${KERNEL_SOURCES})

target_include_directories(moe_prefill_gemm_int4 PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
    ${CMAKE_SOURCE_DIR}/csrc/xe_2
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_TOOLS_INCLUDE_DIR}
    ${CUTLASS_APP_INCLUDE_DIR}
)

target_compile_options(moe_prefill_gemm_int4 PRIVATE ${SYCL_COMPILE_FLAGS})
target_link_options(moe_prefill_gemm_int4 PRIVATE ${SYCL_LINK_FLAGS})

# Link torch
find_package(Torch REQUIRED)
target_link_libraries(moe_prefill_gemm_int4 PRIVATE torch ${TORCH_LIBRARIES})
