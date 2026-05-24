// Sample MLIR for GPU track tests
module {
  func.func @vec_add(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
    %0 = arith.addi %a, %b : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}
