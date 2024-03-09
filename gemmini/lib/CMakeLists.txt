add_mlir_dialect_library(GemminiDialect
  GemminiDialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
