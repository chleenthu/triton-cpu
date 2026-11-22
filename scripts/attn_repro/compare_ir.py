# Compare the repro's IR dump against model.sh's dump, ignoring only
# debug/location info (source file/line differs by construction).
#   python compare_ir.py [repro_dump_dir] [model_dump_dir]
import difflib
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "triton_dump")
B = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "triton_dump")
KERNELS = ("kernel_unified_attention", "reduce_segments")
EXTS = ("ttir", "ttgir", "llir", "ptx", "sass")


def norm_mlir(txt):
    m, n, out = {}, [0], []
    for mt in re.finditer(r'(%[\w#]+) = arith.constant (.*?) : (\S+)', txt):
        m[mt.group(1)] = f"%C<{mt.group(2)}:{mt.group(3)}>"
    for l in txt.splitlines():
        if l.startswith("#loc") or not l.strip():
            continue
        l = re.sub(r'\s*loc\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)', "", l)

        def ren(mt):
            s = mt.group(0)
            if s not in m:
                m[s] = f"%v{n[0]}"
                n[0] += 1
            return m[s]
        out.append(re.sub(r'%[\w#]+', ren, l).rstrip())
    consts = sorted(x for x in out if "arith.constant" in x)
    return consts + [x for x in out if "arith.constant" not in x]


def norm_llir(txt):
    out = []
    for l in txt.splitlines():
        if l.startswith("!") or l.startswith("; ModuleID") or l.startswith("source_filename"):
            continue
        l = re.sub(r',?\s*!dbg !\d+', "", l)
        l = re.sub(r'#dbg_\w+\(.*\)', "", l)
        out.append(l.rstrip())
    return [l for l in out if l.strip()]


def norm_ptx(txt):
    out, skip = [], False
    for l in txt.splitlines():
        s = l.strip()
        if s.startswith(".section") and ".debug" in s:
            skip = True
        if skip:
            if s == "}":
                skip = False
            continue
        if s.startswith((".loc", ".file", "//")) or not s or re.match(r'\$L__tmp\d+:', s):
            continue  # $L__tmpN: debug-line labels, placed per source location
        l = re.sub(r'\$L__tmp\d+', "$L__tmp", l)
        out.append(re.sub(r'\s*//.*', "", l).rstrip())
    return out


def norm_sass(txt):
    return [l.rstrip() for l in txt.splitlines()
            if l.strip() and not l.lstrip().startswith(("//##", "code for"))]


NORM = {"ttir": norm_mlir, "ttgir": norm_mlir, "llir": norm_llir,
        "ptx": norm_ptx, "sass": norm_sass}


def variant(ttir_path, kernel):
    if kernel != "kernel_unified_attention":
        return kernel
    return kernel + ("_3d" if "get_program_id z" in open(ttir_path).read() else "_2d")


def index(d):
    r = {}
    for k in KERNELS:
        for p in glob.glob(os.path.join(d, "*", k + ".ttir")):
            r[variant(p, k)] = (os.path.dirname(p), k)
    return r


ia, ib = index(A), index(B)
allok = True
for v in sorted(set(ia) | set(ib)):
    if v not in ia or v not in ib:
        print(f"{v}: only in {'repro' if v in ia else 'model.sh'} dump")
        allok = False
        continue
    (da, k), (db, _) = ia[v], ib[v]
    res = []
    for e in EXTS:
        fa, fb = os.path.join(da, f"{k}.{e}"), os.path.join(db, f"{k}.{e}")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            res.append(f"{e}=missing")
            continue
        a, b = NORM[e](open(fa).read()), NORM[e](open(fb).read())
        if a == b:
            res.append(f"{e}=same({len(a)})")
        else:
            allok = False
            d = [x for x in difflib.unified_diff(b, a, "model.sh", "repro", lineterm="", n=0)
                 if x[:1] in "+-" and x[:3] not in ("+++", "---")]
            res.append(f"{e}=DIFF({len(d)} lines)")
            print("\n".join(d[:20]))
    print(f"{v:30s} {os.path.basename(db)[:8]} vs {os.path.basename(da)[:8]}: " + "  ".join(res))
print("IR IDENTICAL (modulo debug/loc info)" if allok else "IR DIFFERS")
