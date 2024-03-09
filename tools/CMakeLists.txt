get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(gemmini-opt
  gemmini-opt.cpp
)

target_link_libraries(gemmini-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  GemminiDialect
  LowerGemminiPass
)
