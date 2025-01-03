#ifndef FEGEN_RUNTIME_BINDINGS_VALUE_WRAPPER_H
#define FEGEN_RUNTIME_BINDINGS_VALUE_WRAPPER_H

#include "mlir/IR/Value.h"

namespace fegen {
class Value {
public:
  void dump();
  Value(mlir::Value value);
  mlir::Value get();

private:
  mlir::Value impl;
};

} // namespace fegen

#endif