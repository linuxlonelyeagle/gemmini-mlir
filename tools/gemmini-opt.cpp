#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "GemminiDialect.h"
#include "GemminiOps.h"
#include "Passes.h"

int main(int argc, char *argv[]) {

  // Register all MLIR passes.
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;

  // Register all MLIR core dialects.
  registerAllDialects(registry);

  // Register Gemmini dialect.
  registry.insert<mlir::gemmini::GemminiDialect>();
  mlir::registerPasses();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "buddy-mlir optimizer driver", registry));
}
