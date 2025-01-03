#ifndef _FEGEN_RUNTIME_TYPE_IMPL_H
#define _FEGEN_RUNTIME_TYPE_IMPL_H

// #include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
// #include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
// #include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
// #include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
// #include
// "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
// #include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
// #include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
// #include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
// #include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
// #include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/LLVMIR/LLVMTypes.h"
// #include "mlir/Dialect/Linalg/Passes.h"
// #include "mlir/Dialect/MLProgram/IR/MLProgram.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/MemRef/Transforms/Passes.h"
// #include "mlir/Dialect/Shape/IR/Shape.h"
// #include "mlir/Dialect/Tensor/IR/Tensor.h"
// #include "mlir/IR/Attributes.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
// #include "mlir/IR/OpDefinition.h"
// #include "mlir/IR/Operation.h"
// #include "mlir/IR/TypeRange.h"
// #include "mlir/IR/ValueRange.h"
// #include "mlir/IR/Verifier.h"
// #include "mlir/InitAllDialects.h"
// #include "mlir/InitAllExtensions.h"
// #include "mlir/InitAllPasses.h"
// #include "mlir/InitAllTranslations.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/Pass/PassRegistry.h"
// #include "mlir/Support/FileUtilities.h"
// #include "mlir/Support/LLVM.h"
// #include "mlir/Target/LLVMIR/Dialect/All.h"
// #include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
// #include "mlir/Target/LLVMIR/Export.h"
// #include "mlir/Tools/mlir-opt/MlirOptMain.h"
// #include "mlir/Transforms/Passes.h"
#include <cstddef>
#include <cstdint>

using namespace mlir;
namespace fegen {
class Type {};
class IntegerTypeImpl : public Type {
public:
  Value createConstant(int64_t value);

private:
  int width;
};
} // namespace fegen

#endif