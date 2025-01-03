#ifndef FEGEN_RUNTIME_BINDINGS_TYPE_WRAPPER_H
#define FEGEN_RUNTIME_BINDINGS_TYPE_WRAPPER_H

#include "ValueWrapper.h"
#include "cores/TypesImpl.h"
#include <iostream>
#include <memory>
namespace fegen {

class Type {
public:
  void dump();
  Type(std::shared_ptr<fegen::TypeImpl> impl) : impl(impl) {}
  virtual ~Type() = default;

protected:
  std::shared_ptr<fegen::TypeImpl> impl;
};

class IntegerType : public Type {
public:
  IntegerType(int width);
  Value createConstant(int64_t value);
};

class FloatType : public Type {
public:
  FloatType(int width);
  Value createConstant(double value);
};

} // namespace fegen

#endif