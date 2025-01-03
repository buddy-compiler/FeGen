#include "bindings/ContextManagerWrapper.h"
#include "bindings/TypesWrapper.h"
#include "bindings/ValueWrapper.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(fegen_mlir, m) {
  py::class_<fegen::Value>(m, "Value").def("dump", &fegen::Value::dump);
  py::class_<fegen::Type>(m, "Type").def("dump", &fegen::Type::dump);
  py::class_<fegen::IntegerType, fegen::Type>(m, "IntegerType")
      .def(py::init<int>())
      .def("createConstant", &fegen::IntegerType::createConstant);
  py::class_<fegen::Manager>(m, "ContextManager")
      .def(py::init())
      .def("dump", &fegen::Manager::dump);
}