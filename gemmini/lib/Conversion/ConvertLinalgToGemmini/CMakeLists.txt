add_mlir_library(ConvertLinalgToGemminiPass
  ConvertLinalgToGemmini.cpp
  DEPENDS

  PassesIncGen
  LINK_LIBS PUBLIC
  GemminiTransforms
  )
