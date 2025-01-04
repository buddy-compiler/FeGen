set(FeGen_PYTHON_RUNTIMR_DIR ${FeGen_PYTHON_DIR}/FeGen/runtime)

pybind11_add_module(fegen_mlir 
    bindings.cpp
    ContextManagerWrapper.cpp
    TypesWrapper.cpp
    ValueWrapper.cpp
)

set_target_properties(fegen_mlir 
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${FeGen_PYTHON_RUNTIMR_DIR}
)

target_link_libraries(fegen_mlir 
    PRIVATE
    codegencore
)

set(PYI_CODE "from pybind11_stubgen import main\;main\\(\\)")

add_custom_target(fegen_mlir_pyi
    ALL
    COMMAND ${Python3_EXECUTABLE} -c "${PYI_CODE}" fegen_mlir -o .
    WORKING_DIRECTORY ${FeGen_PYTHON_RUNTIMR_DIR}
    DEPENDS fegen_mlir
    BYPRODUCTS fegen_mlir.pyi
    COMMENT "Generating pyi file for FeGen runtime by executing: ${Python3_EXECUTABLE} -c "${PYI_CODE}" fegen_mlir -o ."
)