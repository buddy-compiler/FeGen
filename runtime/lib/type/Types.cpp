#include "Types.h"
#include "ContextManager.h"
#include <cstdint>

Value fegen::IntegerTypeImpl::createConstant(int64_t value) {
  auto context = ContextManager::getContext();
}