"""zkMod on one GPU: serial witgen carry chain + per-partition proving that starts as soon as its input is ready.

Partition i owns layers [first_i, first_i + size_i). Its prover (the README zkAttn recipe over size_i * H instances)
is launched at t = witgen time of layers [0, first_i + size_i), so proofs overlap later witgen and each other.
Witgen per layer: LLaMa-2-7B measured (6.4 s/layer at n=2048); other models scaled by per-layer FLOPs.
"""
import os, re, sys, json, subprocess, time, tempfile, threading
import argparse
import torch
from fileio_utils import save_int

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--splits', type=str, default='1,2,4,8')
parser.add_argument('--seq_len', type=int, default=2048)
parser.add_argument('--witgen_ref_s', type=float, default=6.4, help='measured witgen s/layer for llama-7b at n=2048')
parser.add_argument('--out', type=str, default=None)

# layers, heads, hidden, ffn, K
MODELS = {
    'opt-125m': (12, 12, 768, 3072, 5), 'opt-350m': (24, 16, 1024, 4096, 5), 'opt-1.3b': (24, 32, 2048, 8192, 5),
    'opt-2.7b': (32, 32, 2560, 10240, 5), 'opt-6.7b': (32, 32, 4096, 16384, 5), 'opt-13b': (40, 40, 5120, 20480, 5),
    'llama-7b': (32, 32, 4096, 11008, 2), 'llama-13b': (40, 40, 5120, 13824, 2),
}
TIME_RE = re.compile(r'time: ([0-9.eE+-]+)')


def layer_flops(d, ffn, n):
    return 2 * n * (4 * d * d + 2 * d * ffn) + 4 * n * n * d


def run_bin(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    m = TIME_RE.findall(r.stdout)
    if r.returncode != 0 or not m:
        print(r.stdout[-1500:], r.stderr[-1500:], file=sys.stderr)
        raise RuntimeError(f'failed: {cmd}')
    return float(m[-1])


def prove_partition(inst, n, R, gen):
    """README recipe for `inst` attention instances; returns compute timers."""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for i in range(inst):
            save_int(torch.randint(0, R, (n * n,)), 1, f'{td}/A_{i}.bin')
        out['tlookup'] = run_bin(f'./tlookup-bm {td}/A_ {inst} {n * n} 0 {R}')
    with tempfile.TemporaryDirectory() as td:
        for i in range(inst):
            save_int(torch.randn(n * n, generator=gen), 65536, f'{td}/A_{i}.bin')
            save_int(torch.randn(n * n, generator=gen), 65536, f'{td}/B_{i}.bin')
        out['hadamard'] = run_bin(f'./hadamard-bm {td}/A_ {td}/B_ {inst} {n * n}')
    with tempfile.TemporaryDirectory() as td:
        for i in range(2 * inst):
            save_int(torch.randn(n * n, generator=gen), 65536, f'{td}/A_{i}.bin')
        out['opening'] = run_bin(f'./com-bm {td}/A_ {td}/com_ {2 * inst} {n * n} {R}')
    return out


def partition_sizes(L, P):
    base, rem = divmod(L, P)
    return [base + (1 if i < rem else 0) for i in range(P)]


if __name__ == '__main__':
    args = parser.parse_args()
    L, H, d, ffn, K = MODELS[args.model]
    n, R = args.seq_len, 65536
    ref = MODELS['llama-7b']
    witgen_layer_s = args.witgen_ref_s * layer_flops(d, ffn, n) / layer_flops(ref[2], ref[3], n)
    if os.system('make tlookup-bm hadamard-bm com-bm > /dev/null 2>&1'):
        raise RuntimeError('make failed')

    results = {}
    for P in (int(x) for x in args.splits.split(',')):
        sizes = partition_sizes(L, P)
        ready = [sum(sizes[:i + 1]) * witgen_layer_s for i in range(P)]
        parts = [None] * P
        starts = [None] * P
        ends = [None] * P

        def worker(i):
            time.sleep(ready[i])
            starts[i] = time.time() - t0
            gen = torch.Generator().manual_seed(1000 * P + i)
            parts[i] = prove_partition(sizes[i] * H, n, R, gen)
            ends[i] = time.time() - t0

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(P)]
        for th in threads: th.start()
        for th in threads: th.join()

        prover = [K * p['tlookup'] + p['hadamard'] + K * p['opening'] for p in parts]
        pipeline_est = max(ready[i] + prover[i] for i in range(P))
        results[P] = {'sizes': sizes, 'witgen_ready_s': ready, 'parts': parts, 'partition_prover_s': prover,
                      'prove_makespan_s': max(prover), 'prove_sum_s': sum(prover),
                      'pipeline_makespan_est_s': pipeline_est, 'wall_start_s': starts, 'wall_end_s': ends,
                      'wall_makespan_s': max(ends)}
        print(f'ZKMOD_PIPE model={args.model} P={P} witgen_layer_s={witgen_layer_s:.2f} witgen_chain_s={ready[-1]:.1f} '
              f'prove_makespan_s={max(prover):.1f} prove_sum_s={sum(prover):.1f} pipeline_makespan_s={pipeline_est:.1f} '
              f'wall_s={max(ends):.0f} parts=' + ','.join(f'{x:.1f}' for x in prover), flush=True)
    if args.out:
        json.dump({'model': args.model, 'L': L, 'H': H, 'd': d, 'ffn': ffn, 'K': K, 'seq_len': n,
                   'witgen_layer_s': witgen_layer_s, 'runs': results}, open(args.out, 'w'), indent=2)
