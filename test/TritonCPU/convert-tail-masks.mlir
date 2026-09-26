// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-ops -triton-cpu-convert-ptr-ops -triton-cpu-convert-elementwise-ops -triton-cpu-convert-elem-manip-ops -triton-cpu-convert-reduction=use-masked-reduction=true -cse -canonicalize | FileCheck %s

// triton_cpu.tail_mask through the TTIR -> TTCIR conversions of a 1x16 tile
// (rsqrt_25-like): the loads get a shape_cast of the 2-D create_mask (shared
// with the select), and tl.sum(tl.where(mask, x*x, 0)) becomes a masked
// vector.reduction over x*x.

// CHECK-LABEL: @one_row
// CHECK:       %[[B0:.+]] = arith.index_cast %{{.+}} : i32 to index
// CHECK:       %[[M:.+]] = vector.create_mask %[[B0]], %{{.+}} : vector<1x16xi1>
// CHECK:       %[[RM:.+]] = vector.shape_cast %[[M]] : vector<1x16xi1> to vector<16xi1>
// CHECK:       vector.maskedload %{{.+}}[%{{.+}}], %[[RM]], %{{.+}}
// CHECK:       %[[X:.+]] = vector.shape_cast %{{.+}} : vector<1x16xf32> to vector<16xf32>
// CHECK:       %[[RM2:.+]] = vector.shape_cast %[[M]] : vector<1x16xi1> to vector<16xi1>
// CHECK:       vector.mask %[[RM2]] { vector.reduction <add>, %[[X]], %{{.+}} fastmath<reassoc> : vector<16xf32> into f32 } : vector<16xi1> -> f32
// CHECK-NOT:   vector.shuffle

module {
  tt.func public @one_row(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<1xf32> {
    %c14_i32 = arith.constant 14 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x16xf32>
    %m = triton_cpu.tail_mask %arg1, %c14_i32 : tensor<1x16xi1>
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %a = tt.addptr %p, %r2 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %v = tt.load %a, %m, %cst : tensor<1x16x!tt.ptr<f32>>
    %sq = arith.mulf %v, %v : tensor<1x16xf32>
    %w = arith.select %m, %sq, %cst : tensor<1x16xi1>, tensor<1x16xf32>
    %s = "tt.reduce"(%w) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %z = arith.addf %x, %y : f32
      tt.reduce.return %z : f32
    }) : (tensor<1x16xf32>) -> tensor<1xf32>
    tt.return %s : tensor<1xf32>
  }
}

// -----

// A 4x16 tile: each row's mask is built from the bounds,
// create_mask(row < b0 ? b1 : 0), instead of a vector.extract of a 2-D mask.

// CHECK-LABEL: @four_rows
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : i32
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : i32
// CHECK:       %[[IN0:.+]] = arith.cmpi sgt, %arg1, %[[C0]] : i32
// CHECK:       %[[L0:.+]] = arith.select %[[IN0]], %arg2, %[[C0]] : i32
// CHECK:       %[[I0:.+]] = arith.index_cast %[[L0]] : i32 to index
// CHECK:       %[[M0:.+]] = vector.create_mask %[[I0]] : vector<16xi1>
// CHECK:       vector.maskedload %{{.+}}[%{{.+}}], %[[M0]], %{{.+}}
// CHECK:       arith.cmpi sgt, %arg1, %[[C3]] : i32
// CHECK-NOT:   vector.extract %{{.+}}[{{.+}}] : vector<16xi1> from vector<4x16xi1>

module {
  tt.func public @four_rows(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<4x16xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x16xf32>
    %m = triton_cpu.tail_mask %arg1, %arg2 : tensor<4x16xi1>
    %xr = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %x2 = tt.expand_dims %xr {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %c16 = arith.constant dense<16> : tensor<4x1xi32>
    %xs = arith.muli %x2, %c16 : tensor<4x1xi32>
    %xb = tt.broadcast %xs : tensor<4x1xi32> -> tensor<4x16xi32>
    %rr = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %rr {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rb = tt.broadcast %r2 : tensor<1x16xi32> -> tensor<4x16xi32>
    %idx = arith.addi %xb, %rb : tensor<4x16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
    %a = tt.addptr %p, %idx : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %v = tt.load %a, %m, %cst : tensor<4x16x!tt.ptr<f32>>
    tt.return %v : tensor<4x16xf32>
  }
}
