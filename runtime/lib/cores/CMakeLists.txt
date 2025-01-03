add_library(codegencore
    STATIC
    TypesImpl.cpp
    ContextManagerImpl.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(codegencore
    PUBLIC
    LLVMSupport
    MLIRToLLVMIRTranslationRegistration
    MLIRFromLLVMIRTranslationRegistration
    MLIRTargetCpp
    MLIRFuncAllExtensions
    MLIRSPIRVTranslateRegistration
    MLIRLLVMIRToLLVMTranslation
    ${dialect_libs}
    ${conversion_libs}
)

get_target_property(codegencore_LIBOUTPUT codegencore LIBRARY_OUTPUT_DIRECTORY)

message(STATUS "codegencore_LIBOUTPUT: ${codegencore_LIBOUTPUT}")