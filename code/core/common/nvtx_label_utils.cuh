#pragma once

#include <cctype>
#include <cstring>
#include <initializer_list>
#include <string>

namespace aisp_nvtx {
namespace detail {
inline bool contains_any(const std::string& value, std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key) {
      continue;
    }
    if (value.find(key) != std::string::npos) {
      return true;
    }
  }
  return false;
}

inline std::string normalize_label(const char* name) {
  if (!name) {
    return std::string();
  }
  std::string out;
  out.reserve(64);
  for (const char* p = name; *p != '\0'; ++p) {
    unsigned char ch = static_cast<unsigned char>(*p);
    if (std::isalnum(ch)) {
      out.push_back(static_cast<char>(std::tolower(ch)));
    } else if (*p == ':') {
      out.push_back(':');
    } else {
      out.push_back('_');
    }
  }
  std::string collapsed;
  collapsed.reserve(out.size());
  bool last_underscore = false;
  for (char ch : out) {
    if (ch == '_') {
      if (last_underscore) {
        continue;
      }
      last_underscore = true;
      collapsed.push_back(ch);
      continue;
    }
    last_underscore = false;
    collapsed.push_back(ch);
  }
  while (!collapsed.empty() && collapsed.front() == '_') {
    collapsed.erase(collapsed.begin());
  }
  while (!collapsed.empty() && collapsed.back() == '_') {
    collapsed.pop_back();
  }
  return collapsed;
}

inline void strip_prefix(std::string& value, const char* prefix) {
  if (!prefix) {
    return;
  }
  size_t len = std::strlen(prefix);
  if (value.size() >= len && value.rfind(prefix, 0) == 0) {
    value.erase(0, len);
  }
}

inline void strip_suffix(std::string& value, const char* suffix) {
  if (!suffix) {
    return;
  }
  size_t len = std::strlen(suffix);
  if (value.size() >= len && value.compare(value.size() - len, len, suffix) == 0) {
    value.erase(value.size() - len);
  }
}

inline std::string strip_variants(std::string value) {
  strip_prefix(value, "baseline_");
  strip_prefix(value, "optimized_");
  strip_suffix(value, "_baseline");
  strip_suffix(value, "_optimized");
  if (value.empty()) {
    return value;
  }
  if (value.front() == '_') {
    value.erase(0, 1);
  }
  if (!value.empty() && value.back() == '_') {
    value.pop_back();
  }
  return value;
}

inline bool is_standard_prefix(const std::string& prefix) {
  return prefix == "setup" ||
         prefix == "warmup" ||
         prefix == "compute_kernel" ||
         prefix == "compute_math" ||
         prefix == "compute_graph" ||
         prefix == "transfer_async" ||
         prefix == "transfer_sync" ||
         prefix == "prefetch" ||
         prefix == "barrier" ||
         prefix == "reduce" ||
         prefix == "verify" ||
         prefix == "cleanup" ||
         prefix == "batch" ||
         prefix == "tile" ||
         prefix == "iteration" ||
         prefix == "step";
}

inline std::string infer_prefix(const std::string& detail) {
  if (detail.empty()) {
    return "compute_kernel";
  }
  if (contains_any(detail, {"warmup"})) {
    return "warmup";
  }
  if (contains_any(detail, {"setup", "init", "initialize"})) {
    return "setup";
  }
  if (contains_any(detail, {"verify", "validation", "check"})) {
    return "verify";
  }
  if (contains_any(detail, {"cleanup", "teardown"})) {
    return "cleanup";
  }
  if (contains_any(detail, {"prefetch"})) {
    return "prefetch";
  }
  if (contains_any(detail, {"copy", "transfer", "memcpy", "h2d", "d2h", "host_to_device", "device_to_host", "zero_copy"})) {
    if (contains_any(detail, {"async", "overlap", "pipelined", "stream"})) {
      return "transfer_async";
    }
    return "transfer_sync";
  }
  if (contains_any(detail, {"reduce", "reduction"})) {
    return "reduce";
  }
  if (contains_any(detail, {"barrier", "sync"})) {
    return "barrier";
  }
  if (contains_any(detail, {"batch"})) {
    return "batch";
  }
  if (contains_any(detail, {"iteration", "iter"})) {
    return "iteration";
  }
  if (contains_any(detail, {"step"})) {
    return "step";
  }
  if (contains_any(detail, {"graph"})) {
    return "compute_graph";
  }
  if (contains_any(detail, {"matmul", "gemm", "attention", "mlp", "moe", "softmax", "layernorm", "norm", "ffn", "conv", "transformer", "inference", "training", "prefill", "decode", "routing"})) {
    return "compute_math";
  }
  return "compute_kernel";
}
}  // namespace detail

inline std::string standardize_nvtx_label(const char* name) {
  std::string normalized = detail::normalize_label(name);
  if (normalized.empty()) {
    return "compute_kernel:unnamed";
  }
  auto colon = normalized.find(':');
  if (colon != std::string::npos) {
    std::string prefix = normalized.substr(0, colon);
    std::string suffix = detail::strip_variants(normalized.substr(colon + 1));
    if (detail::is_standard_prefix(prefix)) {
      if (suffix.empty()) {
        suffix = "unnamed";
      }
      return prefix + ":" + suffix;
    }
  }
  std::string detail_value = detail::strip_variants(normalized);
  if (detail_value.empty()) {
    detail_value = "unnamed";
  }
  std::string prefix = detail::infer_prefix(detail_value);
  return prefix + ":" + detail_value;
}
}  // namespace aisp_nvtx
