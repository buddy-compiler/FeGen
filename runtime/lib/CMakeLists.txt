add_executable(test
    test.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(test
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