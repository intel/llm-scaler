#include "onednn_runtime.h"

namespace oneDNN {

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager instance;
  return instance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager instance;
  return instance;
}

}  // namespace oneDNN
