# Vector Reverse Rematerialization: Reversibility Tables

Which Triton ops and which RVV instructions `--custom-reverse` can rebuild a value through (moved from `DocVecReverse.md`).

## Which Triton op can be reversed

### Non-Elementwise Ops That Can Be Reversed

| Group | Lowered to | Rebuild of the source | status |
|---|---|---|---|
| Lane rotation (slides) | `vslidedown k` ; `vslideup n-k`, or `vslideup k` ; `vslidedown n-k` at VL `k`, tail undisturbed | inverse slides at VL `n`: down `n-k`, up `k` (down `k`, up `n-k` for the up-first form) | implemented (`inverse-slide-rotation`, both orders) |
| Bit rotation in element | `vsrl s` ; `vsll SEW-s` ; `vor` | `vsll s` ; `vsrl SEW-s` ; `vor` | implemented (`inverse-bit-rotation`) |
| Adjacent lane-pair swap | bit rotation by 32 at e64 | the same rotation again | implemented (`inverse-bit-rotation`) |
| Zip (interleave, `tl.join`) | `vwaddu.vv` ; `vwmaccu.vx -1` | `a = vnsrl.wi 0` ; `b = vnsrl.wx SEW` | implemented (`deinterleave-of-zip`) |
| Unzip (`tl.split`) | `vnsrl.wi 0` ; `vnsrl.wx SEW` | `v = zip(lo, hi)`, needs both halves | implemented (`zip-of-deinterleave`) |
| Widen | `vsext` / `vzext`, `vwcvt(u)`, `vfwcvt` | `vnsrl.wi 0` / `vfncvt(.rtz)`, per x2 step | implemented (`narrow-of-widen`), vf2 / vf4 / vf8 |
| Flip (`tl.flip`), permutation | `vid` ; `vrsub` ; `vrgather` ; `vslidedown` | same gather again, or inverse index | implemented (`flip-of-flip`, `inverse-permutation-gather`) |
| Self-inverse permutation (block swaps, `tl.sum` butterfly step) | `vrgatherei16.vv` by a constant index `p`, `p == inverse(p)` | the same gather with the same index (`vrgatherei16.vv y, idx`) | implemented (self-inverse) |
| Mask to vector | `vmerge.vim 0, -1` | `vmsne.vi 0` | implemented (`mask-of-merge`) |
| Concatenate (`tl.cat`) | `vslideup.v[xi] y, b, off`, `a` as passthru | `a` = `COPY` of `y`'s low sub-register; `b = vslidedown y, off` at `b`'s VL | implemented (`low-of-concat`, `slidedown-of-concat`) |
| Prefix sum (`tl.cumsum`) | `vslideup` (some masked by constant masks) ; `vmv.v.v` merges ; `vadd`, `log2(n)` steps | `v = y - vslide1up(y, 0)` | implemented (`diff-of-prefix-sum`), integer only |
| Widen by k | `vwadd(u).vx`, `vwsub(u).vx`, `vfwmul.vf` | `vnsrl.wi 0` ; `vsub.vx k` (`vadd.vx k` for `vwsub`) at the source's SEW; `narrow(y / k)` | implemented for `vwadd(u).vx` / `vwsub(u).vx` (`narrow-of-widen`); `vfwmul.vf` possible |

### Non-Elementwise Ops That Cannot

| Group | Lowered to | Why the source is lost | status |
|---|---|---|---|
| Slide alone | `vslideup` / `vslidedown`, `vslide1up` / `vslide1down` | k lanes fall off one end | no (the rotation pair is implemented) |
| Narrow alone | `vnsrl`, `vnsra`, `vnclip`, `vfncvt` | drops the upper half, or rounds | no (the unzip pair is implemented) |
| Gather with repeats, splat lane | `vrgather.vv` / `.vx` / `.vi` | several lanes read one source lane | no |
| Compress, merge | `vcompress.vm`, `vmerge.vvm` | unselected lanes are dropped | no |
| Reduction | `vredsum`, `vredmax`, `vfredosum`, ... | a vector becomes one scalar | no |
| Element move | `vmv.x.s`, `vfmv.f.s`, `vmv.s.x` | only one lane is kept | no |
| Mask ops | `vmseq`, `vmflt`, `vcpop`, `vfirst`, `viota`, `vmsbf` | many values map to one bit or count | no |
| Widening multiply, both unknown | `vwmul.vv`, `vfwmul.vv` | a product does not fix both factors | no |

## Which RVV instructions can be reversed

The tables below classify RVV instructions in two groups. The elementwise tables hold instructions that compute lane `i` of `y` from lane `i` of the sources: arithmetic, logic, shifts, compares and conversions, including widening and narrowing, where lane `i` of the narrow vector maps to lane `i` of the wide one. The data-movement tables hold instructions that copy or select values without computing on them, or that read other lanes: copies, slides, gathers, compress, merge, scalar moves, splats, reductions and mask scans. Widening, interleave through widening arithmetic (`vwaddu` + `vwmaccu`), deinterleave through `vnsrl` and lane swaps through shift pairs change the layout, so they are listed in the data-movement tables, not in the elementwise ones.

`y` is the result, `v` the value to rebuild, `K` the other source. "Implemented" refers to `ReversePatterns` / `matchReverseOp`, `matchGroupPlans`, `matchFlipPlan` and `matchPermPlan` in `ExpandPseudos.cpp`. "Possible" means the inverse exists but is not matched yet.

### Elementwise instructions

#### Reversible

| RVV instruction (`y = ...`) | rebuild of `v` | live at the site | exact | status |
|---|---|---|---|---|
| `vadd.vv/vx` (`v + K`) | `vsub y, K` | y, K | exact | implemented (elementwise) |
| `vadd.vi` (`v + imm`) | `vadd.vi y, -imm` (not for imm = -16) | y | exact | implemented |
| `vsub.vv/vx` (`v - K` / `K - v`) | `vadd y, K` / `vsub K, y` | y, K | exact | implemented |
| `vrsub.vx/vi` (incl. `vneg`) | `vrsub` again | y, K | exact | implemented |
| `vxor.vv/vx/vi` (incl. `vnot`) | `vxor` again | y, K | exact | implemented |
| `vfadd` / `vfsub` / `vfrsub` | inverse op | y, K | rounded | implemented |
| `vfmul` / `vfdiv` / `vfrdiv` | inverse op | y, K | rounded | implemented |
| `vsext.vf2` / `vzext.vf2` | `vnsrl.wi y, 0` | y | exact | implemented (`narrow-of-widen`) |
| `vwadd(u).vx v, x0` (`vwcvt`) | `vnsrl.wi y, 0` | y | exact | implemented |
| `vfwcvt.f.f.v` | `vfncvt.f.f.w` | y | exact | implemented |
| `vfwcvt.f.x(u).v` | `vfncvt.rtz.x(u).f.w` | y | exact | implemented |
| `vrgather(ei16).vv v, idx`, constant permutation | gather by the inverse table | y | exact | implemented (`inverse-permutation-gather`) |
| same, self-inverse permutation (block swaps) | `vrgatherei16 y, idx` | y, idx | exact | implemented (self-inverse) |
| copies (`vmv.v.v`, `vmv<N>r`, `COPY`) | same value | y | exact | equivalence class |
| `vmul.vx v, c`, c an odd constant | `vmul.vx y, c^-1 mod 2^SEW` | y | exact | possible (integer `MUL` is never reversed today) |
| `vmacc` / `vmadd` (`y = a * b + v`) | `vnmsac`: `v = y - a * b` | y, a, b | exact | possible (tied passthru is rejected today) |
| `vfmacc` / `vfmadd` (accumulator `v`) | `vfnmsac` | y, a, b | rounded | possible |
| `vwadd(u).wv/wx` (`y = v + ext(b)`, `v` wide) | `vwsub(u).wv/wx` | y, b | exact | possible |
| `vsext` / `vzext.vf4/vf8` | 2 or 3 `vnsrl.wi 0` | y | exact | possible (only vf2 today) |
| `vfsgnjn.vv v, v` (`vfneg`) | itself | y | exact | possible |
| `vfmul` / `vfdiv` by a power of 2 | inverse op | y | exact without overflow / underflow | implemented as rounded |

#### Not reversible alone

These instructions drop part of `v`. They become reversible only where the dropped part is still held by another value: a partner instruction's result, or a side value that is live at the site.

| RVV instruction | what is lost | reversible when | status |
|---|---|---|---|
| `vslidedown k` | lanes 0..k-1 | paired with `vslideup n-k` of the same `v` (rotation) | implemented (`inverse-slide-rotation`) |
| `vslideup k` | top k lanes | same rotation pair | implemented |
| `vsrl` / `vsll` by s | low / high s bits | paired with the shift by SEW-s and `vor` / `vxor` / `vadd` (bit rotation; s = 32 at e64 swaps adjacent 32-bit lanes) | implemented (`inverse-bit-rotation`) |
| `vsll` by s alone | high s bits | the high bits are known zero (e.g. after `vzext`): `vsrl` | possible (needs known bits) |
| `vwaddu.vv a, b` alone | `b` is mixed in | followed by `vwmaccu.vx -1, b` (zip) | implemented (`deinterleave-of-zip`) |
| `vnsrl.wi 0` / `vnsrl.w* SEW` | the other half | both halves exist (unzip pair): `vwaddu` + `vwmaccu` | implemented (`zip-of-deinterleave`) |
| `vid`, `vrsub`, `vrgather` x L, `vslidedown` (lane reverse) | each step alone | the whole group replayed on `y`; idx reused if live | implemented (`flip-of-flip`) |
| `vrgather` by a non-constant or non-permutation index (`vrgather.vi`, broadcast) | duplicated lanes / unknown inverse | the index is a constant permutation | no |
| `vnsra`, `vnclip(u)`, `vfncvt.*`, `vfcvt.x.f`, `vfcvt.rtz.*` | high bits, rounding, saturation | - | no |
| `vand`, `vor`, `vandn`, `vmin(u)`, `vmax(u)`, `vfmin`, `vfmax` | masked bits / losing operand | the complement is live (`v = (v & m) \| (v & ~m)`) | possible, rare |
| `vmerge`, `vcompress`, masked ops with a live passthru | masked-off lanes | - | no |
| `vmul.vv`, `vmul.vx` by even or unknown K, `vmulh*`, `vdiv`, `vrem`, `vwmul` | high bits, remainder, division by zero | K an odd constant (above) | no |
| `vsadd`, `vssub`, `vaadd`, `vasub`, `vsmul`, `vssrl`, `vssra` | saturation / rounding bit | - | no |
| `vfsqrt`, `vfrec7`, `vfrsqrt7`, `vfabs`, `vfsgnj(x)` with K | sign / precision | - | no |
| `vredsum`, `vredmax`, `vfred*` | lanes collapsed into one | - | no |
| compares (`vmseq`, `vmflt`, ...) | all but one bit | - | no |
| `vslide1up` / `vslide1down` | one lane | the lost lane is in a live scalar | possible, rare |
| `vid`, `vmv.v.x/i`, loads | no vector source | - | not a reverse case (direct remat or RA remat) |

### Data-movement instructions

#### Data movement that can be reversed

| instruction or group (`y = ...`) | rebuild of `v` | live at the site | exact | status |
|---|---|---|---|---|
| `vmv.v.v` (same VL), `vmv1r/2r/4r/8r.v`, `vmmv.m` (`vmand.mm m, m`), `COPY`, a whole `INSERT_SUBREG` / `REG_SEQUENCE` operand | `y` itself | y | exact | equivalence class |
| `vrgather.vv` / `vrgatherei16.vv v, idx`, idx a constant permutation `p` | gather `y` by `q` (new table) | y | exact | implemented (`inverse-permutation-gather`) |
| same, `p == q` (block swaps, neighbour swaps, byte swap through an e8 gather) | `vrgatherei16 y, idx` | y, idx | exact | implemented (self-inverse) |
| `vid`, `vrsub`, `vrgather.vv` x L, `vslidedown` | the same group on `y` | y (idx if live) | exact | implemented (`flip-of-flip`) |
| `vslidedown k` + `vslideup n-k` of the same `v` | rotation by `n-k` | y | exact | implemented (`inverse-slide-rotation`) |
| `vslideup.vx/vi P, v, k` (tied passthru), `v`'s VL at most VLMAX - k | `vslidedown y, k` at `v`'s VL | y | exact | possible |
| `vsll s` + `vsrl SEW-s` + `vor` / `vxor` / `vadd` (no Zvbb `vror` on the board) | opposite rotation | y | exact | implemented (`inverse-bit-rotation`) |
| `vsext` / `vzext.vf2`, `vwcvt(u).x.x.v` (`vwadd(u).vx v, x0`) | `vnsrl.wi y, 0` (`vncvt.x.x.w`) | y | exact | implemented |
| `vsext` / `vzext.vf4` / `vf8` | 2 or 3 `vnsrl.wi 0` | y | exact | possible |
| `vfwcvt.f.f.v` (f16 to f32 with Zvfh, f32 to f64) | `vfncvt.f.f.w` | y | exact | implemented |
| `vwaddu.vv a, b` + `vwmaccu.vx -1, b` | `vnsrl.wi y, 0` / `vnsrl.wi y, SEW` | y | exact | implemented (`deinterleave-of-zip`) |
| `vnsrl.wi w, 0` + `vnsrl.wi w, SEW` | `vwaddu` + `vwmaccu` | both halves | exact | implemented (`zip-of-deinterleave`) |
| `y = vcompress v, m` + `z = vcompress v, ~m` | `vrgather z, viota(~m)`, then `vrgather y, viota(m)` masked by `m` into it | y, z, m | exact | possible (about 5 instructions) |
| `vmnot.m` (`vmnand.mm m, m`) | itself | y | exact | possible (masks are one register) |

#### Data movement that cannot be reversed alone

| instruction | what is lost | reversible when | status |
|---|---|---|---|
| `vslidedown.vx/vi k` alone | lanes 0..k-1 | the partner `vslideup` exists (rotation), or lanes 0..k-1 of `v` are not demanded (`vslideup y, k`) | rotation implemented; demanded lanes not tracked |
| `vslideup.vx/vi k` alone, `v` fills the register | top k lanes | the partner `vslidedown` exists (rotation) | implemented as a pair only |
| `vslide1up.vx` / `vfslide1up.vf v, x` | `v[n-1]` | `v[n-1]` is in a live scalar: `vslide1down y, s` | possible, rare |
| `vslide1down.vx` / `vfslide1down.vf v, x` | `v[0]` (`x` lands in lane `n-1`) | `v[0]` is in a live scalar: `vslide1up y, s` | possible, rare |
| insert element: `vmv.s.x t, x` + `vslideup v, t, idx` at VL `idx+1`, tail undisturbed | `v[idx]` | old `v[idx]` is in a live scalar: the same pair with it | possible, rare |
| `vmv.s.x` / `vfmv.s.f` into `v` | `v[0]` | `v[0]` is in a live scalar: `vmv.s.x y, s` | possible, rare |
| `vmv.x.s` / `vfmv.f.s` (and extract element = `vslidedown` + `vmv.x.s`) | every other lane (scalar result) | `n == 1` | no |
| `vrgather.vv` / `vrgatherei16.vv` by a constant non-permutation (repeated or missing lanes; an index of VLMAX or more gives 0) | lanes not selected | - | no |
| `vrgather.vv` / `vrgatherei16.vv` by a computed index | unknown permutation | - | no |
| `vrgather.vx` / `vrgather.vi` | all lanes but one (splat) | - | no |
| `vcompress.vm` alone | lanes where `m` is 0 | the complementary compress is live (first table) | possible |
| `vmerge.vvm` / `vxm` / `vim`, `vfmerge.vfm` (`vselect`) | lanes of `v` replaced by the other source | the replaced lanes are live in another value (a second `vmerge`) | no, rare |
| `vnsrl.wi 0` / `vnsrl.wi SEW` alone | the other half | the partner is live (first table) | implemented as a pair only |
| `vmv.v.v` with a smaller VL and tail undisturbed (`PseudoVMV_V_V` with a passthru) | lanes from VL on | the tail of `v` is not demanded | no |
| `EXTRACT_SUBREG` of one part of `v` (also tuple extract after `vlseg`) | the other parts | all parts are live: `REG_SEQUENCE` | possible |
| any instruction of the first table, masked (`v0.t`) or tail undisturbed | inactive / tail lanes take the passthru | those lanes of `v` are not demanded | no |
| `vsll` / `vsrl` / `vsra` / `vnsra` by `s` alone | high / low bits | the partner shift exists (rotation), or the lost bits are known | rotation implemented |
| `vredsum`, `vredmax(u)`, `vredmin(u)`, `vredand/or/xor`, `vfred*`, `vwred*` | lanes collapsed into one | - | no |
| `viota.m`, `vmsbf.m`, `vmsif.m`, `vmsof.m`, `vcpop.m`, `vfirst.m` | mask turned into a count or a prefix | - | no |
| `vmv.v.x` / `vmv.v.i`, `vfmv.v.f`, `vid.v`, `vmset.m` / `vmclr.m`, `PseudoClearVR` | no vector source | - | not a reverse case (direct remat) |
| loads: `vle`, `vle*ff`, `vlse`, `vluxei` / `vloxei`, `vlseg` / `vlsseg` / `vluxseg` / `vloxseg`, `vl<N>re<eew>`, `vlm.v`, `PseudoVRELOAD` | data comes from memory | - | not a reverse case (direct or RA remat while memory is unchanged) |
| stores: `vse`, `vsse`, `vsuxei` / `vsoxei`, `vsseg` / ..., `vs<N>r`, `vsm.v`, `PseudoVSPILL` | no vector result | - | not a reverse case |

The two tables above cover only the RVV instructions that move data without computing on it: lane permutations (gathers, slides, compress, merge), lane-width reshapes, scalar-lane moves, register and mask copies. They are limited to instructions that execute on the BPI-F3 (SpacemiT X60, `rv64imafdcv_..._zvfh_zvfhmin_zvkt`).

`n` is the VL, `p` a constant permutation, `q` its inverse.

### Partially reversing a Triton op

A Triton op that lowers to several RVV instructions can be reversed one instruction at a time from its last one, since each rebuilt value joins `v`'s equivalence class and the next step can read it (`DocReverse.md`, "Chained pattern"). Only the instructions on the data path from `v` to `y` must be reversible. Instructions off that path (index builders such as `vid` / `vrsub`, constant loads, masks) are never inverted; their results only have to be live at the site, as the flip rebuild does with `idx`.

For an op `v -> i1 -> t1 -> ... -> i4 -> t4 -> i5 -> y`, peeling `i5` and `i4` gives `t3` back. If `i3` is in a "not reversible alone" table:

- If the part `i3` drops is held by a value live at the site (the partner slide, the other `vnsrl` half, the complement of an `vand`), `t2` can be rebuilt and peeling continues down to `v`.
- If not, `v` cannot be rebuilt by reversal. `t3` and `t4` still can, which helps when one of them has a distant use; `v` itself is left to a direct option from a root that is still live.
