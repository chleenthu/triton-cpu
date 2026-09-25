// RUN: triton-opt %s -split-input-file -triton-cpu-tail-mask-to-evl -canonicalize -cse | FileCheck %s

// Row mask (pid < 14) broadcast and ANDed with a constant prefix mask, as in
// a persistent reduction with XBLOCK = 1 (r0_numel = 14 of R0_BLOCK = 16
// here): the tail length is select(pid < 14, 16, 0) min 14.

// CHECK-LABEL: @row_and_prefix_mask
// CHECK-DAG:   %[[C14:.+]] = arith.constant 14 : index
// CHECK:       %[[COND:.+]] = arith.cmpi slt, %arg2, %{{.+}} : i32
// CHECK:       %[[SEL:.+]] = arith.select %[[COND]], %{{.+}}, %{{.+}} : index
// CHECK:       %[[LEN:.+]] = arith.minsi %[[SEL]], %[[C14]] : index
// CHECK:       %[[MASK:.+]] = vector.create_mask %[[LEN]] : vector<16xi1>
// CHECK:       vector.maskedload %{{.+}}[%{{.+}}], %[[MASK]], %{{.+}}
// CHECK:       vector.maskedstore %{{.+}}[%{{.+}}], %[[MASK]], %{{.+}}

module {
  tt.func public @row_and_prefix_mask(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c14_i32 = arith.constant 14 : i32
    %pass = arith.constant dense<0.000000e+00> : vector<16xbf16>
    %r0_mask = arith.constant dense<[[true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false]]> : vector<1x16xi1>
    %xmask = arith.cmpi slt, %arg2, %c14_i32 : i32
    %b = vector.broadcast %xmask : i1 to vector<1x16xi1>
    %m2 = arith.andi %b, %r0_mask : vector<1x16xi1>
    %m = vector.shape_cast %m2 : vector<1x16xi1> to vector<16xi1>
    %src = triton_cpu.ptr_to_memref %arg0 : <bf16> -> memref<16xbf16>
    %v = vector.maskedload %src[%c0], %m, %pass : memref<16xbf16>, vector<16xi1>, vector<16xbf16> into vector<16xbf16>
    %dst = triton_cpu.ptr_to_memref %arg1 : <bf16> -> memref<16xbf16>
    vector.maskedstore %dst[%c0], %m, %v : memref<16xbf16>, vector<16xi1>, vector<16xbf16>
    tt.return
  }
}

// -----

// offsets = pid * 16 + arange(16); mask = offsets < n: tail length n - pid * 16.

// CHECK-LABEL: @offsets_lt_n
// CHECK:       %[[OFF:.+]] = arith.muli %arg3, %{{.+}} : i32
// CHECK-DAG:   %[[N64:.+]] = arith.extsi %arg2 : i32 to i64
// CHECK-DAG:   %[[OFF64:.+]] = arith.extsi %[[OFF]] : i32 to i64
// CHECK:       %[[DIFF:.+]] = arith.subi %[[N64]], %[[OFF64]] : i64
// CHECK:       %[[LEN:.+]] = arith.index_cast %[[DIFF]] : i64 to index
// CHECK:       %[[MASK:.+]] = vector.create_mask %[[LEN]] : vector<16xi1>
// CHECK:       vector.maskedload %{{.+}}[%{{.+}}], %[[MASK]], %{{.+}}
// CHECK:       vector.maskedstore %{{.+}}[%{{.+}}], %[[MASK]], %{{.+}}

module {
  tt.func public @offsets_lt_n(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) {
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %iota = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
    %pass = arith.constant dense<0.000000e+00> : vector<16xf32>
    %off = arith.muli %arg3, %c16_i32 : i32
    %offv = vector.broadcast %off : i32 to vector<16xi32>
    %offs = arith.addi %offv, %iota : vector<16xi32>
    %nv = vector.broadcast %arg2 : i32 to vector<16xi32>
    %m = arith.cmpi slt, %offs, %nv : vector<16xi32>
    %src = triton_cpu.ptr_to_memref %arg0 : <f32> -> memref<16xf32>
    %v = vector.maskedload %src[%c0], %m, %pass : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    %dst = triton_cpu.ptr_to_memref %arg1 : <f32> -> memref<16xf32>
    vector.maskedstore %dst[%c0], %m, %v : memref<16xf32>, vector<16xi1>, vector<16xf32>
    tt.return
  }
}

// -----

// Not tail masks: a strided mask (even lanes) and an unsigned compare are
// left alone.

// CHECK-LABEL: @not_tail_masks
// CHECK-NOT:   vector.create_mask
// CHECK:       vector.maskedload %{{.+}}[%{{.+}}], %{{.+}}, %{{.+}} : memref<4xf32>
// CHECK:       arith.cmpi {{ult|ugt}}
// CHECK-NOT:   vector.create_mask
// CHECK:       vector.maskedload

module {
  tt.func public @not_tail_masks(%arg0: !tt.ptr<f32>, %arg1: i32) -> (vector<4xf32>, vector<4xf32>) {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0.000000e+00> : vector<4xf32>
    %even = arith.constant dense<[true, false, true, false]> : vector<4xi1>
    %iota = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %src = triton_cpu.ptr_to_memref %arg0 : <f32> -> memref<4xf32>
    %v0 = vector.maskedload %src[%c0], %even, %pass : memref<4xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
    %nv = vector.broadcast %arg1 : i32 to vector<4xi32>
    %m = arith.cmpi ult, %iota, %nv : vector<4xi32>
    %v1 = vector.maskedload %src[%c0], %m, %pass : memref<4xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
    tt.return %v0, %v1 : vector<4xf32>, vector<4xf32>
  }
}

// -----

// Tail lanes of the loads are never observed: %a only reaches a select on
// the same mask and a masked store on the same mask; %w (size 14) only
// reaches a store whose mask is min(row, 14) <= 14. Their pass-through
// becomes poison. %r feeds a full reduction, so its pass-through is kept.

// CHECK-LABEL: @tail_lanes_unused
// CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:   %[[POISON:.+]] = ub.poison : vector<16xf32>
// CHECK:       %[[A:.+]] = vector.maskedload %{{.+}}[%{{.+}}], %{{.+}}, %[[POISON]]
// CHECK:       %[[W:.+]] = vector.maskedload %{{.+}}[%{{.+}}], %{{.+}}, %[[POISON]]
// CHECK:       %[[R:.+]] = vector.maskedload %{{.+}}[%{{.+}}], %{{.+}}, %[[ZERO]]
// CHECK:       vector.reduction <add>, %[[R]]

module {
  tt.func public @tail_lanes_unused(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) -> (vector<16xf32>, f32) {
    %c0 = arith.constant 0 : index
    %c14_i32 = arith.constant 14 : i32
    %pass = arith.constant dense<0.000000e+00> : vector<16xf32>
    %r0_mask = arith.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false]> : vector<16xi1>
    %row = arith.cmpi slt, %arg3, %c14_i32 : i32
    %b = vector.broadcast %row : i1 to vector<16xi1>
    %m = arith.andi %b, %r0_mask : vector<16xi1>
    %src = triton_cpu.ptr_to_memref %arg0 : <f32> -> memref<16xf32>
    %a = vector.maskedload %src[%c0], %m, %pass : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    %wsrc = triton_cpu.ptr_to_memref %arg1 : <f32> -> memref<16xf32>
    %w = vector.maskedload %wsrc[%c0], %r0_mask, %pass : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    %r = vector.maskedload %wsrc[%c0], %m, %pass : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    %sq = arith.mulf %a, %a : vector<16xf32>
    %sel = arith.select %m, %sq, %pass : vector<16xi1>, vector<16xf32>
    %prod = arith.mulf %w, %a : vector<16xf32>
    %dst = triton_cpu.ptr_to_memref %arg2 : <f32> -> memref<16xf32>
    vector.maskedstore %dst[%c0], %m, %prod : memref<16xf32>, vector<16xi1>, vector<16xf32>
    %sum = vector.reduction <add>, %r : vector<16xf32> into f32
    tt.return %sel, %sum : vector<16xf32>, f32
  }
}
