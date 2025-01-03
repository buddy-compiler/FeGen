#ifndef FEGEN_RUNTIME_BINDINGS_MANAGER_BINDING_H
#define FEGEN_RUNTIME_BINDINGS_MANAGER_BINDING_H

#include "cores/ContextManagerImpl.h"

namespace fegen {
class Manager {
public:
  Manager() : manager(ContextManagerImpl::get()) {}
  void dump();

private:
  std::shared_ptr<ContextManagerImpl> manager;
};

} // namespace fegen

#endif