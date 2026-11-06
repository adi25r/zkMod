#!/usr/bin/env python3
"""Re-commit an OPT workdir with a different commitment shape.

  python3 opt-recommit.py 6.7b --dst zkllm-workdir-sqrt/opt-6.7b [--log-gens-offset 0] [--device 1]

opt-commit.py commits each weight as pow2(rows) row commitments over pow2(cols) << 5 generators
(2^17 to 2^19 generators, 128 to 512 rows for the larger models), so every opening folds a huge
generator vector. Here each parameter with pow2-padded size N gets 2^(floor(log2 N / 2) + offset)
generators, the Hyrax square-root shape: the opening's group work shrinks by 2^(log2 N / 2), and the
extra rows are folded with field arithmetic. The integer weights are unchanged, so they are symlinked
from the source workdir; new generators are drawn with ppgen and every layer is re-committed.
"""
import argparse, glob, os, re, subprocess, sys, time

DIMS = {'125m': (768, 3072), '1.3b': (2048, 8192), '2.7b': (2560, 10240), '6.7b': (4096, 16384), '13b': (5120, 20480)}

ap = argparse.ArgumentParser()
ap.add_argument('model', choices=sorted(DIMS))
ap.add_argument('--src', default=None, help='workdir with the integer weights (default zkllm-workdir/opt-<model>)')
ap.add_argument('--dst', required=True)
ap.add_argument('--log-gens-offset', type=int, default=0, help='generators = 2^(floor(log2 N / 2) + offset)')
ap.add_argument('--device', default=None, help='CUDA device (default: inherit CUDA_VISIBLE_DEVICES)')
args = ap.parse_args()

src = args.src or f'zkllm-workdir/opt-{args.model}'
E, H = DIMS[args.model]
os.makedirs(args.dst, exist_ok=True)
env = dict(os.environ, **({'CUDA_VISIBLE_DEVICES': args.device} if args.device is not None else {}))


def pow2(n):
    return 1 << (n - 1).bit_length()


def shape(name):
    """(rows, cols) as opt-commit.py commits the parameter (2-D weights transposed to in x out)."""
    if name.endswith('.bias') or 'layer_norm' in name:
        return int(os.path.getsize(f'{src}/layer-0-{name}-int.bin') // 4), 1
    if name.startswith('fc1'):
        return E, H
    if name.startswith('fc2'):
        return H, E
    return E, E


names = sorted(re.sub(r'-pp\.bin$', '', os.path.basename(p)) for p in glob.glob(f'{src}/*-pp.bin'))
layers = sorted({int(m[1]) for p in glob.glob(f'{src}/layer-*-int.bin') if (m := re.search(r'layer-(\d+)-', p))})
t0 = time.time()
for name in names:
    rows, cols = shape(name)
    logn = (pow2(rows) * pow2(cols)).bit_length() - 1
    gens = 1 << max(0, min(logn, logn // 2 + args.log_gens_offset))
    subprocess.run(['./ppgen', str(gens), f'{args.dst}/{name}-pp.bin'], check=True, env=env, stdout=subprocess.DEVNULL)
    print(f'{name}: {rows}x{cols} padded 2^{logn}, generators {gens} (was {os.path.getsize(f"{src}/{name}-pp.bin") // 144}), '
          f'rows {(1 << logn) // gens}', flush=True)
    for l in layers:
        base = f'layer-{l}-{name}'
        link = f'{args.dst}/{base}-int.bin'
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(f'{src}/{base}-int.bin'), link)
        subprocess.run(['./commit-param', f'{args.dst}/{name}-pp.bin', link, f'{args.dst}/{base}-commitment.bin', str(rows), str(cols)],
                       check=True, env=env, stdout=subprocess.DEVNULL)
com_mb = sum(os.path.getsize(p) for p in glob.glob(f'{args.dst}/layer-*-commitment.bin')) / 1e6
print(f'RECOMMIT model=opt-{args.model} layers={len(layers)} params={len(names)} commitment_MB={com_mb:.1f} s={time.time() - t0:.1f}')
