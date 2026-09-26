// RUN: triton-opt %s -split-input-file -triton-cpu-analyze-tail-masks -canonicalize -cse | FileCheck %s

// Persistent reduction with XBLOCK = 1 (rsqrt_25-like, R0_BLOCK = 16,
// r0_numel = 14): mask = (r0_index < 14) & splat(pid < 14) on a 1x16 tile.
// Bounds: [pid < 14 ? 1 : 0, 14], used by the loads, the store and the
// tl.where select.

// CHECK-LABEL: @row_and_prefix
// CHECK-DAG:   %[[C14:.+]] = arith.constant 14 : i32
// CHECK:       %[[ROW:.+]] = arith.cmpi slt, %arg3, %[[C14]] : i32
// CHECK:       %[[B0:.+]] = arith.{{select|extui}} %[[ROW]]{{.*}}i32
// CHECK:       %[[M:.+]] = triton_cpu.tail_mask %[[B0]], %[[C14]] : tensor<1x16xi1>
// CHECK:       tt.load %{{.+}}, %[[M]], %{{.+}}
// CHECK:       arith.select %[[M]], %{{.+}}, %{{.+}} : tensor<1x16xi1>, tensor<1x16xf32>
// CHECK:       tt.store %{{.+}}, %{{.+}}, %[[M]]

module {
  tt.func public @row_and_prefix(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c14_i32 = arith.constant 14 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x16xf32>
    %rnumel = arith.constant dense<14> : tensor<1x16xi32>
    %xmask = arith.cmpi slt, %arg3, %c14_i32 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rmask = arith.cmpi slt, %r2, %rnumel : tensor<1x16xi32>
    %off = arith.muli %arg3, %c16_i32 : i32
    %offs = tt.splat %off : i32 -> tensor<1x16xi32>
    %idx = arith.addi %r2, %offs : tensor<1x16xi32>
    %xm = tt.splat %xmask : i1 -> tensor<1x16xi1>
    %mask = arith.andi %rmask, %xm : tensor<1x16xi1>
    %p0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %a0 = tt.addptr %p0, %idx : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %v = tt.load %a0, %mask, %cst : tensor<1x16x!tt.ptr<f32>>
    %sq = arith.mulf %v, %v : tensor<1x16xf32>
    %w = arith.select %mask, %sq, %cst : tensor<1x16xi1>, tensor<1x16xf32>
    %p1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %a1 = tt.addptr %p1, %idx : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    tt.store %a1, %w, %mask : tensor<1x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// offsets = pid * 16 + arange(16); mask = offsets < n: bound n - pid * 16.

// CHECK-LABEL: @offsets_lt_n
// CHECK:       %[[OFF:.+]] = arith.muli %arg3, %{{.+}} : i32
// CHECK:       %[[B:.+]] = arith.subi %arg2, %[[OFF]] : i32
// CHECK:       %[[M:.+]] = triton_cpu.tail_mask %[[B]] : tensor<16xi1>
// CHECK:       tt.load %{{.+}}, %[[M]]

module {
  tt.func public @offsets_lt_n(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) {
    %c16_i32 = arith.constant 16 : i32
    %off = arith.muli %arg3, %c16_i32 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %offv = tt.splat %off : i32 -> tensor<16xi32>
    %offs = arith.addi %offv, %r : tensor<16xi32>
    %nv = tt.splat %arg2 : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offs, %nv : tensor<16xi32>
    %p0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %a0 = tt.addptr %p0, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %a0, %mask : tensor<16x!tt.ptr<f32>>
    %p1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %a1 = tt.addptr %p1, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %a1, %v, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 2-D tile [XBLOCK = 4, RBLOCK = 16]: xmask = (pid*4 + arange(4))[:, None] < xnumel
// broadcast along columns, rmask = arange(16)[None, :] < rnumel broadcast
// along rows. Bounds: [xnumel - pid*4, rnumel].

// CHECK-LABEL: @two_d
// CHECK:       %[[OFF:.+]] = arith.muli %arg3, %{{.+}} : i32
// CHECK:       %[[B0:.+]] = arith.subi %arg1, %[[OFF]] : i32
// CHECK:       %[[M:.+]] = triton_cpu.tail_mask %[[B0]], %arg2 : tensor<4x16xi1>
// CHECK:       tt.load %{{.+}}, %[[M]]

module {
  tt.func public @two_d(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i32) -> tensor<4x16xf32> {
    %c4_i32 = arith.constant 4 : i32
    %off = arith.muli %arg3, %c4_i32 : i32
    %xr = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offv = tt.splat %off : i32 -> tensor<4xi32>
    %x = arith.addi %offv, %xr : tensor<4xi32>
    %x2 = tt.expand_dims %x {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %xn = tt.splat %arg1 : i32 -> tensor<4x1xi32>
    %xm = arith.cmpi slt, %x2, %xn : tensor<4x1xi32>
    %xmb = tt.broadcast %xm : tensor<4x1xi1> -> tensor<4x16xi1>
    %rr = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %rr {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rn = tt.splat %arg2 : i32 -> tensor<1x16xi32>
    %rm = arith.cmpi slt, %r2, %rn : tensor<1x16xi32>
    %rmb = tt.broadcast %rm : tensor<1x16xi1> -> tensor<4x16xi1>
    %mask = arith.andi %xmb, %rmb : tensor<4x16xi1>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
    %rb = tt.broadcast %r2 : tensor<1x16xi32> -> tensor<4x16xi32>
    %a = tt.addptr %p, %rb : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %v = tt.load %a, %mask : tensor<4x16x!tt.ptr<f32>>
    tt.return %v : tensor<4x16xf32>
  }
}

// -----

// Not boxes: a strided mask, an unsigned compare and a suffix mask
// (n < offsets) are left alone.

// CHECK-LABEL: @not_boxes
// CHECK-NOT:   triton_cpu.tail_mask

module {
  tt.func public @not_boxes(%arg0: !tt.ptr<f32>, %arg1: i32) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %even = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %nv = tt.splat %arg1 : i32 -> tensor<4xi32>
    %um = arith.cmpi ult, %r, %nv : tensor<4xi32>
    %sm = arith.cmpi slt, %nv, %r : tensor<4xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %a = tt.addptr %p, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v0 = tt.load %a, %even : tensor<4x!tt.ptr<f32>>
    %v1 = tt.load %a, %um : tensor<4x!tt.ptr<f32>>
    %v2 = tt.load %a, %sm : tensor<4x!tt.ptr<f32>>
    tt.return %v0, %v1, %v2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
