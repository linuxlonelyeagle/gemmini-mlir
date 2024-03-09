#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "GemminiDialect.h"
#include "GemminiOps.h"
#include "Passes.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTLINALGTOGEMMINIPASS
#include "Passes.h.inc"

namespace {
class ConvertLinalgToGemminiPass
    : public impl::ConvertLinalgToGemminiPassBase<ConvertLinalgToGemminiPass> {
public:
  using ConvertLinalgToGemminiPassBase<ConvertLinalgToGemminiPass>::ConvertLinalgToGemminiPassBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    target.addLegalDialect<memref::MemRefDialect, gemmini::GemminiDialect,
                          arith::ArithDialect, scf::SCFDialect>();
    target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
    RewritePatternSet patterns(context);
    populateConvertLinalgToGemminiConversionPatterns(patterns, accType);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gemmini::GemminiDialect, func::FuncDialect,
                    memref::MemRefDialect, linalg::LinalgDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }
};
} // namespace
} // mlir

