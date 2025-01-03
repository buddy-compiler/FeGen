#include "ContextManager.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <mutex>

using namespace fegen;

std::shared_ptr<ContextManager> ContextManager::get() {
  std::call_once(ContextManager::flag, [&] {
    ContextManager::manager = std::make_shared<ContextManager>();
  });
  return ContextManager::manager;
}

mlir::MLIRContext *ContextManager::getContext() {
  auto manager = ContextManager::get();
  return &(manager->context);
}

ContextManager::ContextManager() {
  this->context = new mlir::MLIRContext();
  this->builder = new mlir::OpBuilder(this->context);
}