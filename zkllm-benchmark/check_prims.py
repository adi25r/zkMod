"""Verify the files written by test-prims against Python bignums.
usage: check_prims.py fr prims_fr.txt | check_prims.py fp prims_fp.txt
Fr columns: a b | add sub rmul fmul mont unmont ex2 ex3 | the same 8 via __noinline__ wrappers.
Fp columns: a b | add sub mul mul(mul(a,b),a) mul(a,a)   (mul is the raw Montgomery product a*b/R)."""
import sys

FR_P = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
FP_P = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab


def fr_expect(a, b):
    p, R = FR_P, 1 << 256
    Rinv = pow(R, -1, p)
    d = (b - a) % p
    return [(a + b) % p, (a - b) % p, a * b * Rinv % p, a * b % p, a * R % p, a * Rinv % p,
            (b + d) % p, (b + 2 * d) % p]


def fp_expect(a, b):
    p, R = FP_P, 1 << 384
    Rinv = pow(R, -1, p)
    ab = a * b * Rinv % p
    return [(a + b) % p, (a - b) % p, ab, ab * a * Rinv % p, a * a * Rinv % p]


mode, path = sys.argv[1], sys.argv[2]
p = FR_P if mode == "fr" else FP_P
names = (["add", "sub", "rmul", "fmul", "mont", "unmont", "ex2", "ex3"] if mode == "fr"
         else ["add", "sub", "mul", "mul2", "sqr"])
variants = [("inlined", 0), ("noinline", 8)] if mode == "fr" else [("fp", 0)]
expect = fr_expect if mode == "fr" else fp_expect

bad, firsts, rows, invalid = {}, {}, 0, 0
for line in open(path):
    w = [int(x, 16) for x in line.split()]
    a, b, got = w[0], w[1], w[2:]
    rows += 1
    if a >= p or b >= p:
        invalid += 1
        continue
    exp = expect(a, b)
    for variant, off in variants:
        for k, nm in enumerate(names):
            if got[off + k] != exp[k]:
                key = (variant, nm)
                bad[key] = bad.get(key, 0) + 1
                firsts.setdefault(key, (a, b, exp[k], got[off + k]))
print(f"[{mode}] rows={rows} invalid_inputs_skipped={invalid}")
if not bad:
    print(f"PRIMS_OK[{mode}]: every primitive matches the bignum reference")
for (variant, nm), c in sorted(bad.items()):
    a, b, e, g = firsts[(variant, nm)]
    print(f"BAD[{mode}] {variant:9s} {nm:7s} {c:7d} mismatches; first: a={a:x}\n"
          f"{'':30s}b={b:x}\n{'':30s}exp={e:x}\n{'':30s}got={g:x}\n{'':30s}(got-exp) mod p = {(g - e) % p:x}")
sys.exit(1 if bad else 0)
