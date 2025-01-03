#include "cores/ContextManagerImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <mutex>

using namespace fegen;

std::shared_ptr<ContextManagerImpl> ContextManagerImpl::manager = nullptr;
std::once_flag ContextManagerImpl::flag;

void ContextManagerImpl::initialize() {
  // load dialects
  this->_context.getOrLoadDialect<mlir::arith::ArithDialect>();
  this->_context.getOrLoadDialect<mlir::func::FuncDialect>();
  this->_context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  // set insert point
  this->_builder.setInsertionPointToEnd(this->_theModule.getBody());
  return;
}

std::shared_ptr<ContextManagerImpl> ContextManagerImpl::get() {
  std::call_once(ContextManagerImpl::flag, [&] {
    static mlir::MLIRContext context;
    static mlir::OpBuilder builder(&context);
    static mlir::ModuleOp theMoudle =
        mlir::ModuleOp::create(builder.getUnknownLoc());
    ContextManagerImpl::manager = std::shared_ptr<ContextManagerImpl>(
        new ContextManagerImpl(context, builder, theMoudle));
    manager->initialize();
  });
  return ContextManagerImpl::manager;
}

mlir::MLIRContext &ContextManagerImpl::context() {
  auto manager = ContextManagerImpl::get();
  return manager->_context;
}

mlir::OpBuilder &ContextManagerImpl::builder() {
  return ContextManagerImpl::get()->_builder;
}

mlir::ModuleOp &ContextManagerImpl::theModule() {
  return ContextManagerImpl::get()->_theModule;
}

void ContextManagerImpl::dump() { ContextManagerImpl::theModule().dump(); }

ContextManagerImpl::ContextManagerImpl(mlir::MLIRContext &context,
                                       mlir::OpBuilder &builder,
                                       mlir::ModuleOp &theModule)
    : _context(context), _builder(builder), _theModule(theModule) {}