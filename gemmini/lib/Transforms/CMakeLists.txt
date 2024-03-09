add_mlir_dialect_library(GemminiTransforms
  LegalizeForLLVMExport.cpp
  LinalgToGemminiConversionPatterns.cpp

  LINK_LIBS PUBLIC
  GemminiDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
