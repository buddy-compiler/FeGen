#include "cores/TypesImpl.h"
#include "cores/ContextManagerImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cassert>
#include <cstdint>

void fegen::TypeImpl::dump() { this->mlirType.dump(); }

fegen::IntegerTypeImpl::IntegerTypeImpl(int width)
    : TypeImpl(fegen::ContextManagerImpl::builder().getIntegerType(width)),
      width(width){};

mlir::Value fegen::IntegerTypeImpl::createConstant(int64_t value) {
  auto &builder = ContextManagerImpl::builder();
  auto attr = builder.getIntegerAttr(this->mlirType, value);
  auto v =
      builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), attr);
  return v;
}

mlir::Type getFloatType(int width) {
  if (width == 32) {
    return mlir::FloatType::getF32(&fegen::ContextManagerImpl::context());
  } else if (width == 16) {
    return mlir::FloatType::getF16(&fegen::ContextManagerImpl::context());
  } else if (width == 64) {
    return mlir::FloatType::getF64(&fegen::ContextManagerImpl::context());
  } else {
    assert(false && "unsupported float width");
  }
}

fegen::FloatTypeImpl::FloatTypeImpl(int width)
    : TypeImpl(getFloatType(width)), width(width){};

mlir::Value fegen::FloatTypeImpl::createConstant(double value) {
  mlir::Type floatTy;
  if (this->width == 32) {
    floatTy = mlir::FloatType::getF32(&ContextManagerImpl::context());
  } else if (this->width == 16) {
    floatTy = mlir::FloatType::getF16(&ContextManagerImpl::context());
  } else {
    floatTy = mlir::FloatType::getF64(&ContextManagerImpl::context());
  }
  auto &builder = ContextManagerImpl::builder();
  auto attr = builder.getFloatAttr(floatTy, value);
  auto v =
      builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), attr);
  return v;
}