add_mlir_library(LowerGemminiPass
  LowerGemmini.cpp

  DEPENDS

  PassesIncGen
  LINK_LIBS PUBLIC
  GemminiTransforms
  )
