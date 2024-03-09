add_mlir_dialect(Gemmini gemmini)
add_mlir_doc(Gemmini Gemmini Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Gemmini.td)
mlir_tablegen(GemminiConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(GemminiConversionsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(PassesIncGen)
