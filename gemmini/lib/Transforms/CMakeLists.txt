add_mlir_dialect_library(GemminiTransforms

  LegalizeForLLVMExport.cpp
  LINK_LIBS PUBLIC
  GemminiDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
