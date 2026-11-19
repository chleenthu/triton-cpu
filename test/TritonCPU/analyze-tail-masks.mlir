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

// -----

// OR of nested boxes on an 8x16 tile: the union is the larger box.
//  - constants: [4, 8] | [6, 16]                       -> [6, 16]
//  - symbolic:  [n, m] | [n + 1, 16]  (r < n vs r <= n) -> [n + 1, 16]
// Both pairs differ in both axes, so before the containment check neither
// was recognized.

// CHECK-LABEL: @or_nested
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : i32
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : i32
// CHECK:       %[[MC:.+]] = triton_cpu.tail_mask %[[C6]], %[[C16]] : tensor<8x16xi1>
// CHECK:       tt.load %{{.+}}, %[[MC]]
// CHECK:       %[[N1:.+]] = arith.addi %arg1, %[[C1]] : i32
// CHECK:       %[[MS:.+]] = triton_cpu.tail_mask %[[N1]], %[[C16]] : tensor<8x16xi1>
// CHECK:       tt.load %{{.+}}, %[[MS]]

module {
  tt.func public @or_nested(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %c = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    // constants
    %k4 = arith.constant dense<4> : tensor<8x1xi32>
    %k6 = arith.constant dense<6> : tensor<8x1xi32>
    %k8 = arith.constant dense<8> : tensor<1x16xi32>
    %ra = arith.cmpi slt, %r2, %k4 : tensor<8x1xi32>
    %ca = arith.cmpi slt, %c2, %k8 : tensor<1x16xi32>
    %rb = arith.cmpi slt, %r2, %k6 : tensor<8x1xi32>
    %rab = tt.broadcast %ra : tensor<8x1xi1> -> tensor<8x16xi1>
    %cab = tt.broadcast %ca : tensor<1x16xi1> -> tensor<8x16xi1>
    %rbb = tt.broadcast %rb : tensor<8x1xi1> -> tensor<8x16xi1>
    %boxa = arith.andi %rab, %cab : tensor<8x16xi1>
    %maskc = arith.ori %boxa, %rbb : tensor<8x16xi1>
    // symbolic
    %n = tt.splat %arg1 : i32 -> tensor<8x1xi32>
    %m = tt.splat %arg2 : i32 -> tensor<1x16xi32>
    %rs1 = arith.cmpi slt, %r2, %n : tensor<8x1xi32>
    %cs1 = arith.cmpi slt, %c2, %m : tensor<1x16xi32>
    %rs2 = arith.cmpi sle, %r2, %n : tensor<8x1xi32>
    %rs1b = tt.broadcast %rs1 : tensor<8x1xi1> -> tensor<8x16xi1>
    %cs1b = tt.broadcast %cs1 : tensor<1x16xi1> -> tensor<8x16xi1>
    %rs2b = tt.broadcast %rs2 : tensor<8x1xi1> -> tensor<8x16xi1>
    %box1 = arith.andi %rs1b, %cs1b : tensor<8x16xi1>
    %masks = arith.ori %box1, %rs2b : tensor<8x16xi1>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
    %cb = tt.broadcast %c2 : tensor<1x16xi32> -> tensor<8x16xi32>
    %a = tt.addptr %p, %cb : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %v0 = tt.load %a, %maskc : tensor<8x16x!tt.ptr<f32>>
    %v1 = tt.load %a, %masks : tensor<8x16x!tt.ptr<f32>>
    tt.return %v0, %v1 : tensor<8x16xf32>, tensor<8x16xf32>
  }
}

// -----

// OR of boxes that are not provably nested ([n, m] | [m, n]) differs in both
// axes, so it is still not a box.

// CHECK-LABEL: @or_not_nested
// CHECK-NOT:   triton_cpu.tail_mask

module {
  tt.func public @or_not_nested(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<8x16xf32> {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %c = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rn = tt.splat %arg1 : i32 -> tensor<8x1xi32>
    %rm = tt.splat %arg2 : i32 -> tensor<8x1xi32>
    %cn = tt.splat %arg1 : i32 -> tensor<1x16xi32>
    %cm = tt.splat %arg2 : i32 -> tensor<1x16xi32>
    %r1 = arith.cmpi slt, %r2, %rn : tensor<8x1xi32>
    %c1 = arith.cmpi slt, %c2, %cm : tensor<1x16xi32>
    %r3 = arith.cmpi slt, %r2, %rm : tensor<8x1xi32>
    %c3 = arith.cmpi slt, %c2, %cn : tensor<1x16xi32>
    %r1b = tt.broadcast %r1 : tensor<8x1xi1> -> tensor<8x16xi1>
    %c1b = tt.broadcast %c1 : tensor<1x16xi1> -> tensor<8x16xi1>
    %r3b = tt.broadcast %r3 : tensor<8x1xi1> -> tensor<8x16xi1>
    %c3b = tt.broadcast %c3 : tensor<1x16xi1> -> tensor<8x16xi1>
    %b1 = arith.andi %r1b, %c1b : tensor<8x16xi1>
    %b2 = arith.andi %r3b, %c3b : tensor<8x16xi1>
    %mask = arith.ori %b1, %b2 : tensor<8x16xi1>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
    %cb = tt.broadcast %c2 : tensor<1x16xi32> -> tensor<8x16xi32>
    %a = tt.addptr %p, %cb : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %v = tt.load %a, %mask : tensor<8x16x!tt.ptr<f32>>
    tt.return %v : tensor<8x16xf32>
  }
}

// -----

// A mask yielded by an scf.if: then yields arange < n, else arange < m.
// Bound: cond ? n : m.

// CHECK-LABEL: @if_yield
// CHECK:       %[[C:.+]] = arith.cmpi sgt, %arg3
// CHECK:       %[[B:.+]] = arith.select %[[C]], %arg1, %arg2 : i32
// CHECK:       %[[M:.+]] = triton_cpu.tail_mask %[[B]] : tensor<16xi1>
// CHECK:       tt.load %{{.+}}, %[[M]]

module {
  tt.func public @if_yield(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i32) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cond = arith.cmpi sgt, %arg3, %c0 : i32
    %mask = scf.if %cond -> (tensor<16xi1>) {
      %n = tt.splat %arg1 : i32 -> tensor<16xi32>
      %m = arith.cmpi slt, %r, %n : tensor<16xi32>
      scf.yield %m : tensor<16xi1>
    } else {
      %n = tt.splat %arg2 : i32 -> tensor<16xi32>
      %m = arith.cmpi slt, %r, %n : tensor<16xi32>
      scf.yield %m : tensor<16xi1>
    }
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %a = tt.addptr %p, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %a, %mask : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }
}

// -----

// The then-branch bound n * 2 is computed inside the branch, so it is not
// visible at the load after the if: the mask is left alone. (A bound like
// n + 1 would be fine: add / sub of visible scalars are rebuilt at the user.)

// CHECK-LABEL: @if_yield_inner_bound
// CHECK-NOT:   triton_cpu.tail_mask

module {
  tt.func public @if_yield_inner_bound(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i32) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cond = arith.cmpi sgt, %arg3, %c0 : i32
    %mask = scf.if %cond -> (tensor<16xi1>) {
      %n1 = arith.muli %arg1, %c2 : i32
      %n = tt.splat %n1 : i32 -> tensor<16xi32>
      %m = arith.cmpi slt, %r, %n : tensor<16xi32>
      scf.yield %m : tensor<16xi1>
    } else {
      %n = tt.splat %arg2 : i32 -> tensor<16xi32>
      %m = arith.cmpi slt, %r, %n : tensor<16xi32>
      scf.yield %m : tensor<16xi1>
    }
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %a = tt.addptr %p, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %a, %mask : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }
}

// -----

// A loop-carried mask: m0 = arange < n, m = m & (arange < n - k) each
// iteration, used inside the loop and after it. Its bound is carried as an
// extra i32 iter_arg: init n, yield min(bound, n - k). The i1 iter_arg is
// dead after that (only its own update reads it) and is removed.

// CHECK-LABEL: @for_carried
// CHECK:       %[[R:.+]] = scf.for %[[K:.+]] = {{.+}} iter_args(%[[B:.+]] = %arg1) -> (i32)
// CHECK-NOT:     arith.andi
// CHECK:         %[[TM:.+]] = triton_cpu.tail_mask %[[B]] : tensor<16xi1>
// CHECK:         tt.load %{{.+}}, %[[TM]]
// CHECK:         %[[NK:.+]] = arith.subi %arg1, %[[K]] : i32
// CHECK:         %[[NB:.+]] = arith.minsi %[[B]], %[[NK]] : i32
// CHECK:         scf.yield %[[NB]] : i32
// CHECK:       %[[TA:.+]] = triton_cpu.tail_mask %[[R]] : tensor<16xi1>
// CHECK:       tt.load %{{.+}}, %[[TA]]

module {
  tt.func public @for_carried(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %a = tt.addptr %p, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %m0 = arith.cmpi slt, %r, %n : tensor<16xi32>
    %res = scf.for %k = %c0 to %arg2 step %c1 iter_args(%m = %m0) -> (tensor<16xi1>) : i32 {
      %x = tt.load %a, %m : tensor<16x!tt.ptr<f32>>
      tt.store %a, %x, %m : tensor<16x!tt.ptr<f32>>
      %nk = arith.subi %arg1, %k : i32
      %nks = tt.splat %nk : i32 -> tensor<16xi32>
      %cur = arith.cmpi slt, %r, %nks : tensor<16xi32>
      %mn = arith.andi %m, %cur : tensor<16xi1>
      scf.yield %mn : tensor<16xi1>
    }
    %v = tt.load %a, %res : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }
}

// -----

// A loop-carried mask updated with xor is not a box: no bound iter_args and
// no tail_mask.

// CHECK-LABEL: @for_carried_not_box
// CHECK:       scf.for {{.+}} -> (tensor<16xi1>)
// CHECK-NOT:   triton_cpu.tail_mask

module {
  tt.func public @for_carried_not_box(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %m0 = arith.cmpi slt, %r, %n : tensor<16xi32>
    %res = scf.for %k = %c0 to %arg2 step %c1 iter_args(%m = %m0) -> (tensor<16xi1>) : i32 {
      %mn = arith.xori %m, %m0 : tensor<16xi1>
      scf.yield %mn : tensor<16xi1>
    }
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %a = tt.addptr %p, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %a, %res : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }
}

// -----

// A loop-carried index tensor: offs = arange, offs += 16 each iteration,
// mask = offs < n. The range's offset is carried as an extra i32 iter_arg
// (init 0, yield o + 16), and the mask's bound is n - o.

// CHECK-LABEL: @for_carried_offs
// CHECK:       scf.for %{{.+}} iter_args(%{{.+}} = %{{.+}}, %[[O:.+]] = %c0_i32) -> (tensor<16xi32>, i32)
// CHECK:         %[[B:.+]] = arith.subi %arg1, %[[O]] : i32
// CHECK:         %[[M:.+]] = triton_cpu.tail_mask %[[B]] : tensor<16xi1>
// CHECK:         tt.load %{{.+}}, %[[M]]
// CHECK:         %[[ON:.+]] = arith.addi %[[O]], %c16_i32 : i32
// CHECK:         scf.yield %{{.+}}, %[[ON]] : tensor<16xi32>, i32

module {
  tt.func public @for_carried_offs(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %step = arith.constant dense<16> : tensor<16xi32>
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %res = scf.for %k = %c0 to %arg2 step %c1 iter_args(%offs = %r) -> (tensor<16xi32>) : i32 {
      %m = arith.cmpi slt, %offs, %n : tensor<16xi32>
      %a = tt.addptr %p, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %x = tt.load %a, %m : tensor<16x!tt.ptr<f32>>
      tt.store %a, %x, %m : tensor<16x!tt.ptr<f32>>
      %next = arith.addi %offs, %step : tensor<16xi32>
      scf.yield %next : tensor<16xi32>
    }
    tt.return
  }
}

// -----

// offs = offs * 2 is not a range: no extra iter_arg, no tail_mask.

// CHECK-LABEL: @for_carried_offs_not_range
// CHECK:       scf.for {{.+}} -> (tensor<16xi32>)
// CHECK-NOT:   triton_cpu.tail_mask

module {
  tt.func public @for_carried_offs_not_range(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %two = arith.constant dense<2> : tensor<16xi32>
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %res = scf.for %k = %c0 to %arg2 step %c1 iter_args(%offs = %r) -> (tensor<16xi32>) : i32 {
      %m = arith.cmpi slt, %offs, %n : tensor<16xi32>
      %a = tt.addptr %p, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %x = tt.load %a, %m : tensor<16x!tt.ptr<f32>>
      tt.store %a, %x, %m : tensor<16x!tt.ptr<f32>>
      %next = arith.muli %offs, %two : tensor<16xi32>
      scf.yield %next : tensor<16xi32>
    }
    tt.return
  }
}

// -----

// An index tensor yielded by an scf.if: then arange + a, else arange + b.
// mask = offs < n after the if. Bound: n - (cond ? a : b).

// CHECK-LABEL: @if_yield_offs
// CHECK:       %[[C:.+]] = arith.cmpi sgt, %arg4
// CHECK:       %[[S:.+]] = arith.select %[[C]], %arg2, %arg3 : i32
// CHECK:       %[[B:.+]] = arith.subi %arg1, %[[S]] : i32
// CHECK:       %[[M:.+]] = triton_cpu.tail_mask %[[B]] : tensor<16xi1>
// CHECK:       tt.load %{{.+}}, %[[M]]

module {
  tt.func public @if_yield_offs(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cond = arith.cmpi sgt, %arg4, %c0 : i32
    %offs = scf.if %cond -> (tensor<16xi32>) {
      %a = tt.splat %arg2 : i32 -> tensor<16xi32>
      %o = arith.addi %a, %r : tensor<16xi32>
      scf.yield %o : tensor<16xi32>
    } else {
      %b = tt.splat %arg3 : i32 -> tensor<16xi32>
      %o = arith.addi %b, %r : tensor<16xi32>
      scf.yield %o : tensor<16xi32>
    }
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %m = arith.cmpi slt, %offs, %n : tensor<16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptr = tt.addptr %p, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %ptr, %m : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }
}

// -----

// The branches yield different shapes (a range and a splat): not an index
// tensor the analysis can describe, so no tail_mask.

// CHECK-LABEL: @if_yield_offs_mixed
// CHECK-NOT:   triton_cpu.tail_mask

module {
  tt.func public @if_yield_offs_mixed(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg4: i32) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cond = arith.cmpi sgt, %arg4, %c0 : i32
    %offs = scf.if %cond -> (tensor<16xi32>) {
      scf.yield %r : tensor<16xi32>
    } else {
      %b = tt.splat %arg2 : i32 -> tensor<16xi32>
      scf.yield %b : tensor<16xi32>
    }
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %m = arith.cmpi slt, %offs, %n : tensor<16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptr = tt.addptr %p, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %ptr, %m : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }
}

// -----

// An scf.while carrying an index tensor: offs = arange, offs += 16 in the
// after region, mask = offs < n there. The range's offset is carried as an
// extra before argument (init 0), forwarded by scf.condition to an extra
// after argument, and yielded as o + 16.

// CHECK-LABEL: @while_carried_offs
// CHECK:       scf.while (%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %[[OB:.+]] = %c0_i32) : (tensor<16xi32>, i32, i32)
// CHECK:         scf.condition(%{{.+}}) %{{.+}}, %{{.+}}, %[[OB]] : tensor<16xi32>, i32, i32
// CHECK:       ^bb0(%{{.+}}: tensor<16xi32>, %{{.+}}: i32, %[[OA:.+]]: i32):
// CHECK:         %[[B:.+]] = arith.subi %arg1, %[[OA]] : i32
// CHECK:         %[[M:.+]] = triton_cpu.tail_mask %[[B]] : tensor<16xi1>
// CHECK:         tt.load %{{.+}}, %[[M]]
// CHECK:         %[[ON:.+]] = arith.addi %[[OA]], %c16_i32 : i32
// CHECK:         scf.yield %{{.+}}, %{{.+}}, %[[ON]] : tensor<16xi32>, i32, i32

module {
  tt.func public @while_carried_offs(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %step = arith.constant dense<16> : tensor<16xi32>
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %res:2 = scf.while (%offs = %r, %k = %c0) : (tensor<16xi32>, i32) -> (tensor<16xi32>, i32) {
      %go = arith.cmpi slt, %k, %arg1 : i32
      scf.condition(%go) %offs, %k : tensor<16xi32>, i32
    } do {
    ^bb0(%offs: tensor<16xi32>, %k: i32):
      %m = arith.cmpi slt, %offs, %n : tensor<16xi32>
      %a = tt.addptr %p, %offs : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %x = tt.load %a, %m : tensor<16x!tt.ptr<f32>>
      tt.store %a, %x, %m : tensor<16x!tt.ptr<f32>>
      %next = arith.addi %offs, %step : tensor<16xi32>
      %kn = arith.addi %k, %c16 : i32
      scf.yield %next, %kn : tensor<16xi32>, i32
    }
    tt.return
  }
}

// -----

// An scf.while carrying a mask, used after the loop: the bound comes out as
// an extra result of the scf.while, and the dead i1 argument is removed.

// CHECK-LABEL: @while_carried_mask
// CHECK:       %[[R:.+]]:2 = scf.while ({{.+}}) : (i32, i32) -> (i32, i32)
// CHECK-NOT:     xi1>
// CHECK:       } do {
// CHECK-NOT:     xi1>
// CHECK:         scf.yield
// CHECK:       %[[M:.+]] = triton_cpu.tail_mask %[[R]]#1 : tensor<16xi1>
// CHECK:       tt.load %{{.+}}, %[[M]]

module {
  tt.func public @while_carried_mask(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %n = tt.splat %arg1 : i32 -> tensor<16xi32>
    %m0 = arith.cmpi slt, %r, %n : tensor<16xi32>
    %res:2 = scf.while (%m = %m0, %k = %c0) : (tensor<16xi1>, i32) -> (tensor<16xi1>, i32) {
      %go = arith.cmpi slt, %k, %arg2 : i32
      scf.condition(%go) %m, %k : tensor<16xi1>, i32
    } do {
    ^bb0(%m: tensor<16xi1>, %k: i32):
      %nk = arith.subi %arg1, %k : i32
      %nks = tt.splat %nk : i32 -> tensor<16xi32>
      %cur = arith.cmpi slt, %r, %nks : tensor<16xi32>
      %mn = arith.andi %m, %cur : tensor<16xi1>
      %kn = arith.addi %k, %c1 : i32
      scf.yield %mn, %kn : tensor<16xi1>, i32
    }
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %a = tt.addptr %p, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %a, %res#0 : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }
}
