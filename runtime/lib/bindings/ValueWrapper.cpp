#include "bindings/ValueWrapper.h"
#include "mlir/IR/Value.h"

fegen::Value::Value(mlir::Value value) : impl(value) {}

void fegen::Value::dump() { this->impl.dump(); }

mlir::Value fegen::Value::get() { return this->impl; }