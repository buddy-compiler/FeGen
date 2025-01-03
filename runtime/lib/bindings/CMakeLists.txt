pybind11_add_module(fegen_mlir 
    bindings.cpp
    ContextManagerWrapper.cpp
    TypesWrapper.cpp
    ValueWrapper.cpp
)

set_target_properties(fegen_mlir 
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${FeGen_LIBRARY_DIR}/python
)

target_link_libraries(fegen_mlir 
    PRIVATE
    codegencore
)
