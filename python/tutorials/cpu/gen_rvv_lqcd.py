"""
Generates rvv_lqcd_*_elf.py: the tmLQCD macros and Hopping_Matrix bodies that
Bahi & Eisenbeis evaluate reverse rematerialization on ("Register Reverse
Rematerialization", Tables 2-4: _complex_times_vector, _complexcjg_times_vector,
_su3_multiply, _su3_inverse_multiply, vec times vec, Hopping_Matrix loops k and
l), vectorized over lattice sites: one site per lane, every real or imaginary
part of a spinor / SU(3) link is one 64-lane vector (one LMUL8 register group
at VLEN 256), stored as a plane of a structure-of-arrays field.

Sources (tmLQCD, github.com/etmc/tmLQCD):
  src/lib/su3.h              _vector_add/_sub, _su3_multiply,
                             _su3_inverse_multiply, _complex_times_vector,
                             _complexcjg_times_vector
  src/lib/operator/hopping.h _hop_t_p (loop k), _hop_t_m (loop l)
C99 complex products are expanded in the C order (ac - bd) + i(ad + bc).

Run: python gen_rvv_lqcd.py   (rewrites the rvv_lqcd_*_elf.py files)
"""


class Gen:
    """Straight-line real arithmetic, shared by the Triton kernel and the
    torch reference so both evaluate the same expressions in the same order."""

    def __init__(self, inputs=()):
        self.lines = []
        self.n = 0
        # Input planes are loaded right before their first use, as the C
        # macros read sp->s0.c0 etc. where they need them.
        self.inputs = set(inputs)
        self.loaded = set()

    def use(self, *names):
        for x in names:
            if x in self.inputs and x not in self.loaded:
                self.loaded.add(x)
                self.lines.append(f"{x} = LOAD({x})")

    def tmp(self, expr):
        self.use(*[w for w in expr.replace("+", " ").replace("-", " ").replace("*", " ").split()])
        self.n += 1
        name = f"t{self.n}"
        self.lines.append(f"{name} = {expr}")
        return name

    def add(self, a, b):
        return self.tmp(f"{a} + {b}")

    def sub(self, a, b):
        return self.tmp(f"{a} - {b}")

    def mul(self, a, b):
        return self.tmp(f"{a} * {b}")

    # complex values are (re, im) pairs of names
    def cmul(self, a, b):  # a * b
        return (self.sub(self.mul(a[0], b[0]), self.mul(a[1], b[1])),
                self.add(self.mul(a[0], b[1]), self.mul(a[1], b[0])))

    def cjgmul(self, a, b):  # conj(a) * b
        return (self.add(self.mul(a[0], b[0]), self.mul(a[1], b[1])),
                self.sub(self.mul(a[0], b[1]), self.mul(a[1], b[0])))

    def cadd(self, a, b):
        return (self.add(a[0], b[0]), self.add(a[1], b[1]))

    def csub(self, a, b):
        return (self.sub(a[0], b[0]), self.sub(a[1], b[1]))

    # macros
    def vector_add(self, s1, s2):
        return [self.cadd(x, y) for x, y in zip(s1, s2)]

    def vector_sub(self, s1, s2):
        return [self.csub(x, y) for x, y in zip(s1, s2)]

    def complex_times_vector(self, c, s):
        return [self.cmul(c, x) for x in s]

    def complexcjg_times_vector(self, c, s):
        return [self.cjgmul(c, x) for x in s]

    def su3_multiply(self, u, s):
        out = []
        for r in range(3):
            acc = self.cmul(u[r][0], s[0])
            acc = self.cadd(acc, self.cmul(u[r][1], s[1]))
            acc = self.cadd(acc, self.cmul(u[r][2], s[2]))
            out.append(acc)
        return out

    def su3_inverse_multiply(self, u, s):
        out = []
        for r in range(3):
            acc = self.cjgmul(u[0][r], s[0])
            acc = self.cadd(acc, self.cjgmul(u[1][r], s[1]))
            acc = self.cadd(acc, self.cjgmul(u[2][r], s[2]))
            out.append(acc)
        return out


# Field layouts: plane index of each real number.
def vec_planes(name, base=0):
    return [(f"{name}_{2*(base+k)}", f"{name}_{2*(base+k)+1}") for k in range(3)]


def spinor_planes(name):
    return [vec_planes(name, 3 * j) for j in range(4)]


def su3_planes(name):
    return [[(f"{name}_{2*(3*r+c)}", f"{name}_{2*(3*r+c)+1}") for c in range(3)] for r in range(3)]


def flat(x):
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        return list(x)
    out = []
    for e in x:
        out += flat(e)
    return out


BENCHES = {}


def bench(name, doc, fields, scalars, build):
    """fields: [(field, nplanes)] read, build(g) -> list of output names."""
    BENCHES[name] = (doc, fields, scalars, build)


bench("complex_times_vector", "_complex_times_vector(x, c, y): x = c * y, c a per-site complex",
      [("c", 2), ("y", 6)], [], lambda g: flat(g.complex_times_vector(("c_0", "c_1"), vec_planes("y"))))
bench("complexcjg_times_vector", "_complexcjg_times_vector(r, c, s): r = conj(c) * s, c a per-site complex",
      [("c", 2), ("s", 6)], [], lambda g: flat(g.complexcjg_times_vector(("c_0", "c_1"), vec_planes("s"))))
bench("su3_multiply", "_su3_multiply(r, u, s): r = u * s, u a 3x3 complex SU(3) link, s a color vector",
      [("u", 18), ("s", 6)], [], lambda g: flat(g.su3_multiply(su3_planes("u"), vec_planes("s"))))
bench("su3_inverse_multiply", "_su3_inverse_multiply(r, u, s): r = u^dagger * s",
      [("u", 18), ("s", 6)], [], lambda g: flat(g.su3_inverse_multiply(su3_planes("u"), vec_planes("s"))))


def _vtv(g):
    a, b = vec_planes("a"), vec_planes("b")
    acc = g.cjgmul(a[0], b[0])
    acc = g.cadd(acc, g.cjgmul(a[1], b[1]))
    acc = g.cadd(acc, g.cjgmul(a[2], b[2]))
    return list(acc)


bench("vec_times_vec", "vec times vec: the color-vector scalar product sum_c conj(a.c) * b.c",
      [("a", 6), ("b", 6)], [], _vtv)


def _hop_t_p(g):
    sp, u, ka0 = spinor_planes("sp"), su3_planes("u"), ("ka0_re", "ka0_im")
    temp = [None] * 4
    for (j0, j1), (o0, o1) in (((0, 2), (0, 2)), ((1, 3), (1, 3))):
        psi = g.vector_add(sp[j0], sp[j1])
        chi = g.su3_multiply(u, psi)
        psi = g.complex_times_vector(ka0, chi)
        temp[o0] = psi
        temp[o1] = psi
    return flat(temp)


bench("hop_t_p", "Hopping_Matrix loop k body, _hop_t_p (src/lib/operator/hopping.h):\n"
      "    psi = sp.s0 + sp.s2; chi = u * psi; psi = ka0 * chi; temp.s0 = temp.s2 = psi\n"
      "    psi = sp.s1 + sp.s3; chi = u * psi; psi = ka0 * chi; temp.s1 = temp.s3 = psi\n"
      "The link u is used by both halves.", [("sp", 24), ("u", 18)], ["ka0_re", "ka0_im"], _hop_t_p)


def _hop_t_m(g):
    sm, um, ka0, tin = spinor_planes("sm"), su3_planes("um"), ("ka0_re", "ka0_im"), spinor_planes("tin")
    temp = [list(x) for x in tin]
    for (j0, j1) in ((0, 2), (1, 3)):
        psi = g.vector_sub(sm[j0], sm[j1])
        chi = g.su3_inverse_multiply(um, psi)
        psi = g.complexcjg_times_vector(ka0, chi)
        temp[j0] = g.vector_add(temp[j0], psi)
        temp[j1] = g.vector_sub(temp[j1], psi)
    return flat(temp)


bench("hop_t_m", "Hopping_Matrix loop l body, _hop_t_m (src/lib/operator/hopping.h):\n"
      "    psi = sm.s0 - sm.s2; chi = um^dagger * psi; psi = conj(ka0) * chi; temp.s0 += psi; temp.s2 -= psi\n"
      "    psi = sm.s1 - sm.s3; chi = um^dagger * psi; psi = conj(ka0) * chi; temp.s1 += psi; temp.s3 -= psi",
      [("sm", 24), ("um", 18), ("tin", 24)], ["ka0_re", "ka0_im"], _hop_t_m)

TEMPLATE = '''"""
tmLQCD {name}, vectorized over lattice sites
{ul}
{doc}

Generated by gen_rvv_lqcd.py (see there for the sources). One lattice site per
lane; every real or imaginary part is a {block}-lane f32 vector, one LMUL8
register group at VLEN 256, loaded from plane p of a structure-of-arrays field
(field_ptr + p * n_sites + site). Inputs are small nonzero integers so the
float results are exact and the torch reference matches bit for bit.
"""

import torch

import triton
import triton.language as tl

CPU_BLOCK_SIZE = {block}


@triton.jit
def lqcd_{name}_kernel({params}out_ptr, n_sites, {sparams}BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    site = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = site < n_sites
{loads}
{body}
{stores}


torch.manual_seed(0)
n_sites = {nsites}
triton.runtime.driver.set_active_to_cpu()
values = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
{inputs}
{scalar_vals}
from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def reference():
{ref_loads}
{ref_body}
    return torch.stack([{outs}])


def run_on_board(kernel, constexprs, name):
    expected = reference().reshape(-1)
    arguments = {{
{args}
        "out_ptr": [0.0] * ({nout} * n_sites),
        "n_sites": n_sites,
{sargs}
    }}
    grid = (triton.cdiv(n_sites, CPU_BLOCK_SIZE), )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{{name}}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={{"out_ptr": expected.tolist()}},
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{{name}}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(lqcd_{name}_kernel, {{"BLOCK_SIZE": CPU_BLOCK_SIZE}}, "rvv-lqcd-{dname}")
'''


def emit(name, block=64, nsites=4096):
    doc, fields, scalars, build = BENCHES[name]
    g = Gen(f"{f}_{p}" for f, n in fields for p in range(n))
    outs = build(g)
    g.use(*outs)  # outputs that are plain inputs (none today)
    params = "".join(f"{f}_ptr, " for f, _ in fields)
    sparams = "".join(f"{s}, " for s in scalars)
    loads = ""

    def load_expr(line, kernel):
        name = line.split(" = ")[0]
        if " = LOAD(" not in line:
            return line
        f, p = name.rsplit("_", 1)
        return (f"{name} = tl.load({f}_ptr + {p} * n_sites + site, mask=mask)" if kernel else f"{name} = {f}[{p}]")

    body = "\n".join("    " + load_expr(l, True) for l in g.lines)
    ref_body = "\n".join("    " + load_expr(l, False) for l in g.lines)
    stores = "\n".join(f"    tl.store(out_ptr + {i} * n_sites + site, {o}, mask=mask)" for i, o in enumerate(outs))
    inputs = "\n".join(
        f"{f} = values[torch.randint(0, len(values), ({n}, n_sites))]" for f, n in fields)
    scalar_vals = "\n".join(f"{s} = {v}" for s, v in zip(scalars, ("2.0", "-1.0")))
    ref_loads = "" 
    args = "\n".join(f'        "{f}_ptr": {f}.reshape(-1).tolist(),' for f, _ in fields)
    sargs = "\n".join(f'        "{s}": {s},' for s in scalars)
    src = TEMPLATE.format(name=name, ul="=" * (len(name) + 45), doc=doc, block=block, params=params,
                          sparams=sparams, loads=loads, body=body, stores=stores, nsites=nsites, inputs=inputs,
                          scalar_vals=scalar_vals, ref_loads=ref_loads, ref_body=ref_body, outs=", ".join(outs),
                          nout=len(outs), args=args, sargs=sargs, dname=name.replace("_", "-"))
    with open(f"rvv_lqcd_{name}_elf.py", "w") as f:
        f.write(src)
    print(f"rvv_lqcd_{name}_elf.py: {len(g.lines)} ops, {sum(n for _, n in fields)} input planes, "
          f"{len(outs)} outputs")


if __name__ == "__main__":
    for name in BENCHES:
        emit(name)
