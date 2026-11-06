"""Reproduce the zkllm-benchmark README prover-time recipe (zkAttn bottleneck) and add zkMod boundary overhead.

  K x tlookup-bm  (L*H instances, n*n each, range [0, 65536))
  1 x hadamard-bm (L*H instances, n*n each)
  K x com-bm      (2*L*H instances, n*n each, 65536 generators)
"""
import os, re, sys, json, subprocess, tempfile, time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='opt-125m ... opt-13b, llama-7b, llama-13b')
parser.add_argument('--seq_len', type=int, default=2048)
parser.add_argument('--splits', type=str, default='1,2,4,8')
parser.add_argument('--out', type=str, default=None)

# (layers, heads, hidden, K tlookups in zkAttn). README: "zkAttn involves 5 (or 2) tlookups".
MODELS = {
    'opt-125m': (12, 12, 768, 5), 'opt-350m': (24, 16, 1024, 5), 'opt-1.3b': (24, 32, 2048, 5),
    'opt-2.7b': (32, 32, 2560, 5), 'opt-6.7b': (32, 32, 4096, 5), 'opt-13b': (40, 40, 5120, 5),
    'llama-7b': (32, 32, 4096, 2), 'llama-13b': (40, 40, 5120, 2),
}

TIME_RE = re.compile(r'time: ([0-9.eE+-]+)')


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    m = TIME_RE.findall(r.stdout)
    if r.returncode != 0 or not m:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f'failed: {cmd}')
    return float(m[-1])


if __name__ == '__main__':
    args = parser.parse_args()
    L, H, d, K = MODELS[args.model]
    n = args.seq_len
    R = 65536
    inst = L * H

    ops = {}
    ops['tlookup'] = run(f'python tlookup-bm.py {inst} {n * n} 0 {R}')
    ops['hadamard'] = run(f'python hadamard-bm.py {inst} {n * n}')
    ops['opening'] = run(f'python com-bm.py {2 * inst} {n * n} {R}')
    prover_s = K * ops['tlookup'] + ops['hadamard'] + K * ops['opening']

    with tempfile.TemporaryDirectory() as td:
        import torch
        from fileio_utils import save_int
        save_int(torch.randn(n, d), 65536, f'{td}/X.bin')
        if os.system(f'./ppgen {d} {td}/act-pp.bin > /dev/null'):
            raise RuntimeError('ppgen failed')
        t0 = time.time()
        if os.system(f'./commit-param {td}/act-pp.bin {td}/X.bin {td}/C.bin {n} {d} > /dev/null'):
            raise RuntimeError('commit-param failed')
        boundary_commit_s = time.time() - t0
    boundary_open_s = run(f'python com-bm.py 1 {n * d} {d}')
    per_cut_s = 2 * (boundary_commit_s + boundary_open_s)
    per_cut_MB = 2 * n * 144 / 1e6

    zkmod = {}
    for p in (int(x) for x in args.splits.split(',')):
        zkmod[p] = {'cuts': p - 1, 'overhead_s': (p - 1) * per_cut_s, 'prover_s': prover_s + (p - 1) * per_cut_s,
                    'extra_commitment_MB': (p - 1) * per_cut_MB}

    result = {'model': args.model, 'layers': L, 'heads': H, 'hidden': d, 'K': K, 'seq_len': n, 'instances': inst,
              'ops_s': ops, 'prover_s': prover_s, 'boundary_commit_s': boundary_commit_s,
              'boundary_open_s': boundary_open_s, 'per_cut_s': per_cut_s, 'zkmod': zkmod}
    print(f'ZKATTN_BENCH model={args.model} L={L} H={H} d={d} K={K} n={n} tlookup_s={ops["tlookup"]:.1f} '
          f'hadamard_s={ops["hadamard"]:.1f} opening_s={ops["opening"]:.1f} prover_s={prover_s:.1f} per_cut_s={per_cut_s:.3f} '
          + ' '.join(f'zkmod{p}_s={r["prover_s"]:.1f}' for p, r in zkmod.items()))
    if args.out:
        json.dump(result, open(args.out, 'w'), indent=2)
