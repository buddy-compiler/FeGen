#include "bindings/TypesWrapper.h"
#include "bindings/ValueWrapper.h"
#include "cores/TypesImpl.h"
#include <memory>
#include <pybind11/pytypes.h>

void fegen::Type::dump() { impl->dump(); }

fegen::IntegerType::IntegerType(int width)
    : Type(std::make_shared<fegen::IntegerTypeImpl>(width)) {}

fegen::Value fegen::IntegerType::createConstant(int64_t value) {
  auto exact_impl =
      std::dynamic_pointer_cast<fegen::IntegerTypeImpl>(this->impl);
  return Value(exact_impl->createConstant(value));
}

fegen::FloatType::FloatType(int width)
    : Type(std::make_shared<fegen::FloatTypeImpl>(width)) {}

fegen::Value fegen::FloatType::createConstant(double value) {
  auto exact_impl = std::dynamic_pointer_cast<fegen::FloatTypeImpl>(this->impl);
  return Value(exact_impl->createConstant(value));
}