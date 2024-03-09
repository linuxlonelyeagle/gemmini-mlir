#ifndef BUDDY_TRANSFORMS_PASSES_H
#define BUDDY_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

class LLVMTypeConverter;
class RewritePatternSet;
class LLVMConversionTarget;

void populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t addrLen, int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
    size_t sizeOfAccT);

void configureGemminiLegalizeForExportTarget(LLVMConversionTarget &target);

void populateConvertLinalgToGemminiConversionPatterns(RewritePatternSet &patterns,
                                                    std::string accType);

} // namespace mlir
#endif