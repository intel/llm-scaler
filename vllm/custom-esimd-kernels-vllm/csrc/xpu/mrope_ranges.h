#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <limits>

// Exact physical byte-range checks shared by MRoPE accelerator callers.
// Tensor alias identity is insufficient for independently wrapped DLPack views.
inline bool qwen38_mrope_byte_ranges_overlap(
    std::uintptr_t left_begin, std::uintptr_t left_bytes,
    std::uintptr_t right_begin, std::uintptr_t right_bytes) {
  constexpr auto kMaxAddress =
      std::numeric_limits<std::uintptr_t>::max();
  if (left_bytes == 0 || right_bytes == 0) {
    return false;
  }
  if (left_begin > kMaxAddress - left_bytes ||
      right_begin > kMaxAddress - right_bytes) {
    return true;
  }
  const auto left_end = left_begin + left_bytes;
  const auto right_end = right_begin + right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

inline bool qwen38_mrope_ranges_overlap(
    const at::Tensor& left, const at::Tensor& right) {
  if (left.numel() == 0 || right.numel() == 0) {
    return false;
  }
  if (left.is_contiguous() && right.is_contiguous()) {
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data_ptr());
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data_ptr());
    const auto left_numel = static_cast<std::uintptr_t>(left.numel());
    const auto right_numel = static_cast<std::uintptr_t>(right.numel());
    if (left_numel > std::numeric_limits<std::uintptr_t>::max() /
            left.element_size() ||
        right_numel > std::numeric_limits<std::uintptr_t>::max() /
            right.element_size()) {
      return true;
    }
    return qwen38_mrope_byte_ranges_overlap(
        left_begin, left_numel * left.element_size(), right_begin,
        right_numel * right.element_size());
  }

  // Only positions may be non-contiguous under the QSA ABI.  Reject any
  // other pair conservatively instead of treating a strided tensor as dense.
  if (!left.is_contiguous() && !right.is_contiguous()) {
    return true;
  }
  // Each positions row is contiguous; check exact row intervals including the
  // gap between rows.
  const at::Tensor& positions = left.is_contiguous() ? right : left;
  const at::Tensor& contiguous = left.is_contiguous() ? left : right;
  if (positions.dim() == 2 && positions.size(0) == 3 &&
      positions.stride(1) == 1) {
    const auto contiguous_begin =
        reinterpret_cast<std::uintptr_t>(contiguous.data_ptr());
    const auto contiguous_numel =
        static_cast<std::uintptr_t>(contiguous.numel());
    if (contiguous_numel > std::numeric_limits<std::uintptr_t>::max() /
            contiguous.element_size()) {
      return true;
    }
    const auto contiguous_bytes =
        contiguous_numel * contiguous.element_size();
    const auto position_width = static_cast<std::uintptr_t>(positions.size(1));
    if (position_width > std::numeric_limits<std::uintptr_t>::max() /
            positions.element_size()) {
      return true;
    }
    const auto row_bytes = position_width * positions.element_size();
    for (int64_t row = 0; row < positions.size(0); ++row) {
      if (positions.stride(0) <= 0) {
        return true;
      }
      const auto row_index = static_cast<std::uintptr_t>(row);
      const auto row_stride =
          static_cast<std::uintptr_t>(positions.stride(0));
      if (row_index > std::numeric_limits<std::uintptr_t>::max() /
              row_stride) {
        return true;
      }
      const auto row_offset_elements = row_index * row_stride;
      if (row_offset_elements >
          std::numeric_limits<std::uintptr_t>::max() /
              positions.element_size()) {
        return true;
      }
      const auto row_offset =
          row_offset_elements * positions.element_size();
      const auto position_begin =
          reinterpret_cast<std::uintptr_t>(positions.data_ptr());
      if (position_begin > std::numeric_limits<std::uintptr_t>::max() -
              row_offset) {
        return true;
      }
      if (qwen38_mrope_byte_ranges_overlap(
              position_begin + row_offset, row_bytes, contiguous_begin,
              contiguous_bytes)) {
        return true;
      }
    }
    return false;
  }
  return true;
}
