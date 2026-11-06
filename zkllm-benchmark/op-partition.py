#!/usr/bin/env python3
"""zkMod with tensor-operation partitions: every partition proves k consecutive tensor operations
(OPT_OPS_PER_LAYER = 31 per OPT layer, see opt-layer.cuh) instead of whole layers.

  python3 op-partition.py <model> <seq_len> --calibrate            # per-op footprint of layer 0 -> <out>/calib.json
  python3 op-partition.py <model> <seq_len> --ops-per-part K        # one run, prints OPS_SUMMARY

One witgen process runs the forward pass and dumps the state at every partition start (in
/dev/shm); prover i starts as soon as its input state exists and the memory gate allows it. The
gate reserves, per running process, the calibrated peak of the most expensive op it contains plus a
per-process context allowance; a prover that still runs out of memory is retried. Every boundary
tensor is committed and opened on both sides; the run checks that producer and consumer commitments
agree and that the model output equals the reference (the K = all-ops run).
Run it with CUDA_MPS_PIPE_DIRECTORY set to a running MPS daemon, as for opt-partition.py.
"""
import argparse, json, os, re, shutil, subprocess, sys, time

OPS_PER_LAYER = 31
DIMS = {'125m': (768, 3072, 12), '1.3b': (2048, 8192, 24), '2.7b': (2560, 10240, 32),
        '6.7b': (4096, 16384, 32), '13b': (5120, 20480, 40)}
LIVE = [['x'], ['Xc', 'gir', 'x'], ['Xc', 'gir_', 'x'], ['x', 'yln'], ['x', 'yln_'], ['n', 'x'],
        ['n', 'q0', 'x'], ['n', 'q1', 'x'], ['n', 'q', 'x'], ['k0', 'n', 'q', 'x'], ['k1', 'n', 'q', 'x'], ['k', 'n', 'q', 'x'],
        ['k', 'q', 'v0', 'x'], ['k', 'q', 'v1', 'x'], ['k', 'q', 'v', 'x'], ['s', 'v', 'x'], ['v', 'x', 'ysm'], ['o', 'x'], ['o_', 'x'],
        ['h'], ['Xc', 'gir', 'h'], ['Xc', 'gir_', 'h'], ['h', 'yln'], ['h', 'yln_'], ['h', 'n'],
        ['f0', 'h'], ['f1', 'h'], ['fp', 'h'], ['h', 'r'], ['g0', 'h'], ['g1', 'h']]

ap = argparse.ArgumentParser()
ap.add_argument('model', choices=sorted(DIMS))
ap.add_argument('seq_len', type=int)
ap.add_argument('--ops-per-part', type=int, default=None, help='K; 0 = all ops in one partition')
ap.add_argument('--calibrate', action='store_true')
ap.add_argument('--workdir', default=None, help='committed weights (default zkllm-workdir/opt-<model>)')
ap.add_argument('--num-layers', type=int, default=None)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--concurrent', type=int, default=16, help='max provers at once')
ap.add_argument('--ctx-mb', type=int, default=500, help='per-process allowance on top of the tracked allocation peak')
ap.add_argument('--safety-mb', type=int, default=512)
ap.add_argument('--out', default='results-ops', help='results directory (calib.json, summary.log, per-run json)')
ap.add_argument('--state-dir', default='/dev/shm/zkmod-ops')
args = ap.parse_args()

E, H, L_all = DIMS[args.model]
L = args.num_layers or L_all
TOTAL = OPS_PER_LAYER * L
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = os.path.join(args.out, f'opt-{args.model}')
os.makedirs(OUT, exist_ok=True)
SD = os.path.join(args.state_dir, f'{args.model}')
workdir = args.workdir or f'zkllm-workdir/opt-{args.model}'
model_input = f'opt-{args.model}_input_{args.seq_len}.bin'
act_pp = os.path.join(args.state_dir, f'act-pp-{E}.bin')
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))


def pow2(n):
    return 1 << (n - 1).bit_length()


def ensure_act_pp():
    os.makedirs(args.state_dir, exist_ok=True)
    if not os.path.exists(act_pp) or os.path.getsize(act_pp) != pow2(E) * 144:
        subprocess.run(['./ppgen', str(E), act_pp], check=True, env=env, stdout=subprocess.DEVNULL)


def cmd(inp, outp, mode, a, n, extra=()):
    return ['./partition', inp, str(args.seq_len), str(E), str(H), workdir, '0', str(L), outp, mode,
            '--arch', 'opt', '--swiglu-table', 'relu-table.bin', '--act-pp', act_pp, '--op-range', str(a), str(n), *extra]


def parse(stdout):
    m = re.findall(r'^PARTITION (.*)$', stdout, re.M)
    return dict(kv.split('=', 1) for kv in m[-1].split()) if m else {}


def smi(q):
    r = subprocess.run(['nvidia-smi', '-i', str(args.device), f'--query-gpu={q}', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return int(r.stdout.strip().split('\n')[0])


def state_files(base, cut):
    if cut % OPS_PER_LAYER == 0:
        return [base]
    return [f'{base}.{n}.bin' for n in LIVE[cut % OPS_PER_LAYER]]


def calibrate():
    """Layer 0 as single-op partitions, one at a time: per-op tracked peak and times."""
    ensure_act_pp()
    shutil.rmtree(SD, ignore_errors=True)
    os.makedirs(SD)
    cuts = ','.join(str(c) for c in range(1, OPS_PER_LAYER))
    r = subprocess.run(cmd(model_input, f'{SD}/wg', 'witgen', 0, OPS_PER_LAYER,
                           ['--dump-cuts', cuts, '--dump-prefix', f'{SD}/cut']), env=env, capture_output=True, text=True)
    wg = parse(r.stdout)
    ops = []
    for a in range(OPS_PER_LAYER):
        inp = model_input if a == 0 else f'{SD}/cut{a}'
        r = subprocess.run(cmd(inp, f'{SD}/o{a}', 'prove', a, 1, ['--commit-input', '1', '--commit-output', '1', '--prefix', f'{SD}/p{a}']),
                           env=env, capture_output=True, text=True)
        f = parse(r.stdout)
        if f.get('status') != 'OK':
            sys.exit(f'calibration of op {a} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
        ops.append({k: float(f[k]) for k in ['alloc_peak_mb', 'prove_s', 'witgen_s', 'input_commit_s', 'output_commit_s', 'wall_s']})
        print(f'calibrate op {a:2d}: ' + ' '.join(f'{k}={v:.3f}' for k, v in ops[-1].items()), flush=True)
    shutil.rmtree(SD, ignore_errors=True)
    json.dump({'model': args.model, 'seq_len': args.seq_len, 'witgen_alloc_peak_mb': float(wg.get('alloc_peak_mb', 0)), 'ops': ops},
              open(os.path.join(OUT, 'calib.json'), 'w'), indent=1)


def run(k):
    calib = json.load(open(os.path.join(OUT, 'calib.json')))
    peak = [o['alloc_peak_mb'] for o in calib['ops']]
    k = k or TOTAL
    starts = list(range(0, TOTAL, k))
    parts = [(a, min(k, TOTAL - a)) for a in starts]
    P = len(parts)
    need = [max(peak[g % OPS_PER_LAYER] for g in range(a, a + n)) + args.ctx_mb for a, n in parts]
    if os.environ.get('ZKLLM_SQUASH') == '1':
        # Calibration proves each op's lookups alone; a multi-op partition proves merged lookups of up to
        # 2^ZKLLM_SQUASH_MAX_LOG elements, measured at about 128 B per element above the op peaks.
        squash_mb = (128 << int(os.environ.get('ZKLLM_SQUASH_MAX_LOG', '24'))) >> 20
        need = [x + (squash_mb if n > 1 else 0) for x, (a, n) in zip(need, parts)]
    witgen_need = calib['witgen_alloc_peak_mb'] + args.ctx_mb

    ensure_act_pp()
    shutil.rmtree(SD, ignore_errors=True)
    os.makedirs(SD)
    total_mb, base_used = smi('memory.total'), smi('memory.used')
    budget = total_mb - base_used - args.safety_mb
    if max(need) > budget:
        sys.exit(f'a single partition needs {max(need)} MB > budget {budget} MB')

    t0 = time.time()
    wg = None
    if P > 1:
        wg = subprocess.Popen(cmd(model_input, f'{SD}/wg_end', 'witgen', 0, starts[-1],
                                  ['--dump-cuts', ','.join(map(str, starts[1:])), '--dump-prefix', f'{SD}/cut',
                                   '--dump-max-pending', str(args.concurrent)]),
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    running, done, results, retries, max_conc, peak_used = {}, set(), {}, 0, 0, 0
    queue = list(range(P))
    wg_s = None
    last_smi = 0

    def ready(i):
        return i == 0 or os.path.exists(f'{SD}/cut{parts[i][0]}.ready')

    def launch(i):
        a, n = parts[i]
        inp = model_input if i == 0 else f'{SD}/cut{a}'
        c = cmd(inp, f'{SD}/out{i}', 'prove', a, n, ['--part-idx', str(i), '--commit-input', '1', '--commit-output', '1', '--prefix', f'{SD}/p{i}'])
        running[i] = (subprocess.Popen(c, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), time.time() - t0)

    while queue or running:
        # reap
        for i in [i for i, (p, _) in running.items() if p.poll() is not None]:
            p, ts = running.pop(i)
            out, err = p.communicate()
            f = parse(out)
            if p.returncode != 0 or f.get('status') != 'OK':
                if 'cudaMalloc failed' in err + out:
                    retries += 1
                    queue.insert(0, i)
                    continue
                sys.exit(f'partition {i} failed:\n{out[-2000:]}\n{err[-3000:]}')
            f['launch_s'], f['end_s'] = ts, time.time() - t0
            results[i] = f
            a = parts[i][0]
            if i > 0:   # its input state is consumed
                for fn in state_files(f'{SD}/cut{a}', a) + [f'{SD}/cut{a}.ready']:
                    if os.path.exists(fn):
                        os.remove(fn)
            if i < P - 1:
                for fn in state_files(f'{SD}/out{i}', a + parts[i][1]):
                    if os.path.exists(fn):
                        os.remove(fn)
        if wg is not None and wg_s is None and wg.poll() is not None:
            o, e = wg.communicate()
            if wg.returncode != 0:
                sys.exit(f'witgen failed:\n{o[-2000:]}\n{e[-3000:]}')
            wg_s = time.time() - t0
            for fn in state_files(f'{SD}/wg_end', starts[-1]):
                if os.path.exists(fn):
                    os.remove(fn)
        # launch in order
        reserved = sum(need[j] for j in running) + (witgen_need if wg is not None and wg_s is None else 0)
        while queue and len(running) < args.concurrent and ready(queue[0]) and reserved + need[queue[0]] <= budget:
            i = queue.pop(0)
            launch(i)
            reserved += need[i]
        max_conc = max(max_conc, len(running))
        if time.time() - last_smi > 0.5:
            peak_used = max(peak_used, smi('memory.used'))
            last_smi = time.time()
        time.sleep(0.05)
    e2e = time.time() - t0

    # chain: producer's output commitments == consumer's input commitments
    chain_ok = True
    for i in range(1, P):
        for name in LIVE[parts[i][0] % OPS_PER_LAYER]:
            a, b = f'{SD}/p{i - 1}-out-{name}-commitment.bin', f'{SD}/p{i}-in-{name}-commitment.bin'
            chain_ok &= os.path.exists(a) and os.path.exists(b) and open(a, 'rb').read() == open(b, 'rb').read()
    final = f'{SD}/out{P - 1}'
    ref = os.path.join(OUT, f'final_out_L{L}.bin')
    if k == TOTAL and not os.path.exists(ref):
        shutil.copy(final, ref)
    output_ok = os.path.exists(ref) and open(ref, 'rb').read() == open(final, 'rb').read()
    shutil.rmtree(SD, ignore_errors=True)

    s = lambda key: sum(float(r[key]) for r in results.values())
    summ = {'model': args.model, 'seq_len': args.seq_len, 'layers': L, 'total_ops': TOTAL, 'ops_per_part': k, 'parts': P,
            'e2e_s': e2e, 'witgen_s': wg_s or 0.0, 'prover_sum_s': s('prover_total_s'), 'prove_sum_s': s('prove_s'),
            'commit_sum_s': s('input_commit_s') + s('output_commit_s'), 'witgen_in_provers_s': s('witgen_s'),
            'proc_wall_sum_s': s('wall_s'), 'max_concurrent': max_conc, 'retries': retries, 'gpu_peak_used_mb': peak_used,
            'chain_ok': chain_ok, 'output_ok': output_ok,
            'status': 'OK' if chain_ok and output_ok else 'FAIL'}
    json.dump({'summary': summ, 'partitions': [results[i] for i in sorted(results)]},
              open(os.path.join(OUT, f'run_k{k}_L{L}.json'), 'w'), indent=1)
    line = 'OPS_SUMMARY ' + ' '.join(f'{a}={b:.3f}' if isinstance(b, float) else f'{a}={b}' for a, b in summ.items())
    print(line, flush=True)
    with open(os.path.join(OUT, 'summary.log'), 'a') as f:
        f.write(line + '\n')


if args.calibrate:
    calibrate()
else:
    if args.ops_per_part is None:
        sys.exit('give --ops-per-part K or --calibrate')
    run(args.ops_per_part)
