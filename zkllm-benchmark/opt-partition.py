import os, sys, re, json, time, subprocess, threading, importlib.util
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='OPT layer-partitioned proving (zkMod-style)')
parser.add_argument('model_size', type=str, help='125m, 350m, 1.3b, 2.7b, 6.7b, 13b')
parser.add_argument('seq_len', type=int, help='The sequence length to prove (power of two)')
parser.add_argument('--splits', type=int, default=1, help='Number of contiguous layer partitions (even split, remainder to the first ones)')
parser.add_argument('--layers', type=str, default=None, help='Explicit partition sizes, e.g. 8,8,8,8 (overrides --splits)')
parser.add_argument('--num_layers', type=int, default=None, help='Number of decoder layers to prove (default: all)')
parser.add_argument('--devices', type=str, default='0', help='Comma-separated CUDA device ids; partition i runs on devices[i %% n]')
parser.add_argument('--concurrent', type=int, default=None, help='Wave width: partitions proven simultaneously (default: all)')
parser.add_argument('--sequential', action='store_true', help='Prove partitions one at a time (wave width 1)')
parser.add_argument('--input_file', type=str, default='layer_input.bin', help='Input activation (random if missing)')
parser.add_argument('--workdir', type=str, default=None, help='Directory with committed weights (default ./zkllm-workdir/Llama-2-{N}b)')
parser.add_argument('--out-json', type=str, default='opt-partition-summary.json', help='Where to write the JSON summary')
parser.add_argument('--skip-witgen', action='store_true', help='Reuse existing part*_in.bin files from a previous witgen chain')
parser.add_argument('--mem-per-proc-mb', type=int, default=4500, help='Pipeline mode: only launch a prover when this much GPU memory is free')
parser.add_argument('--witgen-margin-mb', type=int, default=1500, help='Pipeline mode: GPU memory kept free for the concurrently running witgen chain')
parser.add_argument('--pipeline', action='store_true', help='Start proving partition i as soon as its witgen input exists (up to --concurrent provers at once)')
parser.add_argument('--prover-mode', choices=['layer', 'op'], default='layer',
                    help='op: run each layer partition through the op path (partition --op-range) with streamed activations, '
                         'the same proofs and boundary commitments in about a quarter of the GPU memory')
parser.add_argument('--early-carry', action='store_true', default=os.environ.get('ZKLLM_EARLY_CARRY') == '1',
                    help='One witgen process for the whole chain that publishes each partition input (carry_<cut>, '
                         'plus a .ready marker) as soon as its forward values are final, instead of one witgen process '
                         'per partition each waiting for the previous to exit. Each carry is checked against the '
                         'producing prover\'s own output. Default from ZKLLM_EARLY_CARRY=1')
parser.add_argument('--witgen-first', action='store_true',
                    help='Witness-first proving (zkLLM\'s method): the prover first runs witgen for every layer, keeping '
                         'every tensor the proofs need in host memory, then proves from the stored tensors. '
                         'Needs --splits 1 --prover-mode layer, no --pipeline')

OPS_PER_LAYER = 31  # opt-layer.cuh OPT_OPS_PER_LAYER

from transformers import AutoConfig

PARTITION_RE = re.compile(r'^PARTITION (.*)$', re.M)


def parse_partition_line(stdout):
    m = PARTITION_RE.findall(stdout)
    if not m:
        return None
    fields = {}
    for kv in m[-1].split():
        k, v = kv.split('=', 1)
        try:
            fields[k] = float(v) if '.' in v else int(v)
        except ValueError:
            fields[k] = v
    return fields


def ensure_relu_table(in_range_num_bit=10, in_prec_num_bit=12, out_prec_num_bit=16):
    if os.path.isfile('relu-table.bin'):
        return
    import torch, fileio_utils
    Xs = torch.arange(-(1 << (in_range_num_bit - 1)), 1 << (in_range_num_bit - 1), step=1 / (1 << in_prec_num_bit))
    fileio_utils.save_int(torch.relu(Xs), 1 << out_prec_num_bit, 'relu-table.bin')


def partition_sizes(num_layers, splits, explicit):
    if explicit:
        sizes = [int(s) for s in explicit.split(',')]
        if sum(sizes) != num_layers or any(s <= 0 for s in sizes):
            raise ValueError(f'--layers {explicit} must be positive and sum to {num_layers}')
        return sizes
    base, rem = divmod(num_layers, splits)
    if base == 0:
        raise ValueError(f'--splits {splits} > num_layers {num_layers}')
    return [base + (1 if i < rem else 0) for i in range(splits)]


if __name__ == '__main__':
    args = parser.parse_args()
    if os.system('make partition ppgen'):
        print('Error compiling partition/ppgen')
        exit(1)

    model_card = f'facebook/opt-{args.model_size}'
    config = AutoConfig.from_pretrained(model_card, local_files_only=True, cache_dir='./model-storage')
    if not config.do_layer_norm_before:
        raise ValueError(f'opt-{args.model_size} is post-LN (do_layer_norm_before=False); '
                          'opt-layer.cu only implements pre-LN OPT. Known post-LN size: opt-350m.')
    embed_dim, hidden_dim = config.hidden_size, config.ffn_dim
    num_layers = args.num_layers or config.num_hidden_layers
    eps = 1e-5
    workdir = args.workdir or f'./zkllm-workdir/opt-{args.model_size}'
    devices = args.devices.split(',')

    pow2 = lambda n: 1 << (n - 1).bit_length()
    ensure_relu_table()
    # One shared basis for every boundary commitment. Never regenerate between witgen and prove.
    if not os.path.isfile('act-pp.bin') or os.path.getsize('act-pp.bin') != pow2(embed_dim) * 144:
        if os.system(f'./ppgen {embed_dim} act-pp.bin'):
            print('Error generating act-pp.bin')
            exit(1)
    if not os.path.isfile(args.input_file):
        np.round(np.random.randn(args.seq_len, embed_dim) * (1 << 16)).astype(np.int32).tofile(args.input_file)

    if args.witgen_first:
        # zkLLM's own flow (the monolithic baseline); zkMod keeps its concurrent witgen/proving.
        if args.prover_mode != 'layer' or args.pipeline or args.splits != 1:
            raise ValueError('--witgen-first needs --splits 1 --prover-mode layer and no --pipeline')

    sizes = partition_sizes(num_layers, args.splits, args.layers)
    P = len(sizes)
    firsts = [sum(sizes[:i]) for i in range(P)]
    # A cut is where partition i starts, in the units of the witgen path: layers, or ops with --prover-mode op.
    cut_of = lambda i: OPS_PER_LAYER * firsts[i] if args.prover_mode == 'op' else firsts[i]
    if args.early_carry:
        inputs = [args.input_file] + [f'carry_{cut_of(i)}' for i in range(1, P)]
    else:
        inputs = [args.input_file] + [f'part{i}_in.bin' for i in range(1, P)]
    outputs = [f'part{i + 1}_in.bin' for i in range(P - 1)] + ['final_out.bin']

    def base_cmd(i, mode, output):
        cmd = ['./partition', inputs[i], str(args.seq_len), str(embed_dim), str(hidden_dim), workdir,
               str(firsts[i]), str(sizes[i]), output, mode, '--arch', 'opt', '--swiglu-table', 'relu-table.bin']
        if args.prover_mode == 'op':
            cmd += ['--op-range', str(OPS_PER_LAYER * firsts[i]), str(OPS_PER_LAYER * sizes[i])]
        return cmd

    def early_witgen_cmd():
        # Forward pass up to the last partition's start, dumping every partition input on the way. The end
        # state equals the last dump, so carry_end is deleted (clear_end).
        cmd = ['./partition', args.input_file, str(args.seq_len), str(embed_dim), str(hidden_dim), workdir,
               '0', str(firsts[P - 1]), 'carry_end', 'witgen', '--arch', 'opt', '--swiglu-table', 'relu-table.bin',
               '--dump-cuts', ','.join(str(cut_of(i)) for i in range(1, P)), '--dump-prefix', 'carry_']
        if args.prover_mode == 'op':
            cmd += ['--op-range', '0', str(cut_of(P - 1))]
        return cmd

    def clear_carries():
        # A stale .ready marker would launch a prover on the previous run's input.
        for fn in [f for i in range(1, P) for f in (inputs[i], inputs[i] + '.ready', inputs[i] + '.tmp')] + \
                  [f for f in os.listdir('.') if f.startswith('carry_end')]:
            if os.path.exists(fn):
                os.remove(fn)

    def clear_end():
        for fn in [f for f in os.listdir('.') if f.startswith('carry_end')]:
            os.remove(fn)

    def carry_matches():
        # The early carry must equal what the producing prover recomputed for its own output.
        if not args.early_carry:
            return True
        ok = True
        for i in range(P - 1):
            try:
                same = open(inputs[i + 1], 'rb').read() == open(f'part{i}_prove_out.bin', 'rb').read()
            except FileNotFoundError:
                same = False
            if not same:
                print(f'early carry {inputs[i + 1]} differs from partition {i}\'s prover output')
            ok &= same
        return ok

    def com_path(i, side):
        # The op path commits each live boundary tensor by name; at a layer boundary that is just x.
        return f'part{i}-{side}-x-commitment.bin' if args.prover_mode == 'op' else f'part{i}-{side}-commitment.bin'

    def env_for(i):
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': devices[i % len(devices)]}

    def prove_cmd(i):
        return base_cmd(i, 'prove', f'part{i}_prove_out.bin') + [
            '--part-idx', str(i), '--act-pp', 'act-pp.bin',
            '--commit-input', str(int(i > 0)), '--commit-output', str(int(i < P - 1)),
            '--prefix', f'part{i}', '--eps', repr(eps)]

    def collect(i, proc, results):
        out, _ = proc.communicate()
        sys.stdout.write(out)
        fields = parse_partition_line(out) or {}
        fields.update(idx=i, exit_code=proc.returncode, device=devices[i % len(devices)])
        results.append(fields)
        return proc.returncode == 0 and fields.get('status') == 'OK'

    def gpu_free_mb(dev):
        q = subprocess.run(['nvidia-smi', '-i', str(dev), '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout
        return int(q)

    def gpu_total_mb(dev):
        q = subprocess.run(['nvidia-smi', '-i', str(dev), '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout
        return int(q)

    def proc_mem_mb():
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True).stdout
        mem = {}
        for line in out.strip().splitlines():
            pid, used = line.split(',')
            mem[int(pid)] = int(used)
        return mem

    if args.pipeline:
        # Witgen chain stays serial; partition i's prover launches as soon as its input exists and the GPU
        # has room for one more prover (memory gate), never more than W at once.
        W = args.concurrent or P
        results, running = [], {}
        state = {'ok': True}
        t_start = time.time()
        ends = {}
        launches = {}

        def reap():
            for j in [j for j, p in running.items() if p.poll() is not None]:
                state['ok'] &= collect(j, running.pop(j), results)
                ends[j] = time.time() - t_start

        WITGEN_MARGIN_MB = args.witgen_margin_mb
        need = args.mem_per_proc_mb

        def reserved_mb():
            # Each running prover is assumed to grow to its peak; reserve what it has not allocated yet.
            used = proc_mem_mb()
            return sum(max(0, need - used.get(p.pid, 0)) for p in running.values())

        def launch(i, margin):
            dev = devices[i % len(devices)]
            while True:
                reap()
                if len(running) < W and gpu_free_mb(dev) - reserved_mb() >= need + margin:
                    break
                time.sleep(0.2)
            running[i] = subprocess.Popen(prove_cmd(i), env=env_for(i), stdout=subprocess.PIPE, text=True)
            launches[i] = time.time() - t_start

        # The witgen chain runs in its own thread, always ahead: partition i's input is ready once
        # witgen(i - 1) has finished, independently of how many provers are running. Provers launch
        # in order as their inputs appear and the memory gate allows; while the chain is still
        # running, each launch leaves WITGEN_MARGIN_MB free for it. If prover + witgen exceed the
        # GPU outright, the chain runs to completion first and provers are gated on their own need.
        # Partition P-1's witgen output (final_out.bin) is never consumed by anything:
        # its own prove step recomputes and saves the identical forward pass anyway.
        overlap = need + WITGEN_MARGIN_MB <= gpu_total_mb(devices[0])
        if not overlap:
            print(f'pipeline: prover ({need} MB) + witgen ({WITGEN_MARGIN_MB} MB) exceed the GPU; running witgen chain first')
        ready = [threading.Event() for _ in range(P)]
        ready[0].set()
        chain = {'done_s': 0.0, 'failed': False}

        def run_chain_early():
            clear_carries()
            if P == 1:
                return
            wg = subprocess.Popen(early_witgen_cmd(), env=env_for(0))
            while True:
                rc = wg.poll()
                for i in range(1, P):
                    if not ready[i].is_set() and os.path.exists(inputs[i] + '.ready'):
                        ready[i].set()
                if rc is not None:
                    break
                time.sleep(0.05)
            if rc != 0 or not all(e.is_set() for e in ready):
                print(f'early-carry witgen failed (exit {rc})')
                chain['failed'] = True
                for e in ready:
                    e.set()
                return
            chain['done_s'] = time.time() - t_start
            clear_end()

        def run_chain():
            if args.early_carry:
                return run_chain_early()
            for i in range(P - 1):
                r = subprocess.run(base_cmd(i, 'witgen', outputs[i]), env=env_for(0))
                if r.returncode != 0:
                    print(f'witgen of partition {i} failed (exit {r.returncode})')
                    chain['failed'] = True
                    for e in ready:
                        e.set()
                    return
                ready[i + 1].set()
            chain['done_s'] = time.time() - t_start

        chain_thread = threading.Thread(target=run_chain, daemon=True)
        chain_thread.start()
        if not overlap:
            chain_thread.join()
        for i in range(P):
            while not ready[i].wait(0.2):
                reap()
            if chain['failed']:
                exit(1)
            launch(i, WITGEN_MARGIN_MB if chain_thread.is_alive() else 0)
        chain_thread.join()
        witgen_chain_s = chain['done_s']
        for j, p in running.items():
            state['ok'] &= collect(j, p, results)
            ends[j] = time.time() - t_start
        status_ok = state['ok']
        e2e_s = time.time() - t_start
        results.sort(key=lambda f: f['idx'])
        seq_sum_s = sum(f.get('prover_total_s', 0.0) for f in results)
        pow2 = lambda n: 1 << (n - 1).bit_length()
        expected_len = pow2(args.seq_len * embed_dim) // pow2(embed_dim) * 144
        chain_ok = True
        for i in range(P - 1):
            try:
                a = open(com_path(i, 'out'), 'rb').read()
                b = open(com_path(i + 1, 'in'), 'rb').read()
                chain_ok &= (a == b) and len(a) == expected_len
            except FileNotFoundError:
                chain_ok = False
        carry_ok = carry_matches()
        status = 'OK' if status_ok and chain_ok and carry_ok else 'FAIL'
        print(f'PIPELINE_SUMMARY parts={P} concurrent={W} witgen_chain_s={witgen_chain_s:.3f} e2e_s={e2e_s:.3f} '
              f'prover_sum_s={seq_sum_s:.3f} prover_max_s={max(f.get("prover_total_s", 0.0) for f in results):.3f} '
              f'early_carry={args.early_carry} chain_ok={chain_ok} carry_ok={carry_ok} status={status}')
        with open(args.out_json, 'w') as f:
            json.dump({'config': {'model': model_card, 'seq_len': args.seq_len, 'embed_dim': embed_dim, 'hidden_dim': hidden_dim,
                                  'num_layers': num_layers, 'sizes': sizes, 'devices': devices, 'concurrent': W, 'pipeline': True,
                                  'early_carry': args.early_carry},
                       'partitions': results, 'witgen_chain_s': witgen_chain_s, 'e2e_s': e2e_s, 'prover_sum_s': seq_sum_s,
                       'partition_launch_s': launches, 'partition_end_s': ends, 'chain_ok': chain_ok, 'carry_ok': carry_ok,
                       'status': status}, f, indent=2)
        exit(0 if status == 'OK' else 1)

    # Serial witness-generation chain: each partition's input is the previous partition's output.
    prep_s = 0.0
    t_start = time.time()
    if not args.skip_witgen and args.early_carry and P > 1:
        t0 = time.time()
        clear_carries()
        r = subprocess.run(early_witgen_cmd(), env=env_for(0))
        if r.returncode != 0:
            print(f'early-carry witgen failed (exit {r.returncode})')
            exit(1)
        clear_end()
        prep_s = time.time() - t0
    elif not args.skip_witgen:
        t0 = time.time()
        for i in range(P - 1):
            # Partition P-1's witgen output (final_out.bin) is never consumed by anything:
            # its own prove step recomputes and saves the identical forward pass anyway.
            r = subprocess.run(base_cmd(i, 'witgen', outputs[i]), env=env_for(0))
            if r.returncode != 0:
                print(f'witgen of partition {i} failed (exit {r.returncode})')
                exit(1)
        prep_s = time.time() - t0

    # Prove partitions in waves of W concurrent processes, one GPU each.
    W = 1 if args.sequential else (args.concurrent or P)
    results = []
    makespan_s = wall_makespan_s = seq_sum_s = 0.0
    status_ok = True
    for wave_start in range(0, P, W):
        wave = list(range(wave_start, min(wave_start + W, P)))
        t0 = time.time()
        procs = {}
        for i in wave:
            cmd = base_cmd(i, 'prove', f'part{i}_prove_out.bin') + [
                '--part-idx', str(i), '--act-pp', 'act-pp.bin',
                '--commit-input', str(int(i > 0)), '--commit-output', str(int(i < P - 1)),
                '--prefix', f'part{i}', '--eps', repr(eps), '--witgen-first', str(int(args.witgen_first))]
            procs[i] = subprocess.Popen(cmd, env=env_for(i), stdout=subprocess.PIPE, text=True)
        wave_max = 0.0
        for i in wave:
            out, _ = procs[i].communicate()
            sys.stdout.write(out)
            fields = parse_partition_line(out) or {}
            fields.update(idx=i, exit_code=procs[i].returncode, device=devices[i % len(devices)])
            results.append(fields)
            if procs[i].returncode != 0 or fields.get('status') != 'OK':
                status_ok = False
            else:
                seq_sum_s += fields['prover_total_s']
                wave_max = max(wave_max, fields['prover_total_s'])
        wall_makespan_s += time.time() - t0
        makespan_s += wave_max

    # Cross-partition binding: the producer's output commitment must equal the consumer's input commitment.
    expected_len = pow2(args.seq_len * embed_dim) // pow2(embed_dim) * 144
    chain_ok = True
    for i in range(P - 1):
        try:
            a = open(com_path(i, 'out'), 'rb').read()
            b = open(com_path(i + 1, 'in'), 'rb').read()
            chain_ok &= (a == b) and len(a) == expected_len
        except FileNotFoundError:
            chain_ok = False
    carry_ok = carry_matches()
    status = 'OK' if status_ok and chain_ok and carry_ok else 'FAIL'

    e2e_s = time.time() - t_start
    # Witgen-first: the record pass's wall time, its witgen compute and store (device->host) times, the host memory
    # it peaked at, and the replay's load (host->device) time.
    wf = ''
    if args.witgen_first and results:
        r0 = results[0]
        wf = (f'wf_record_s={r0.get("record_s", 0):.3f} wf_witgen_s={r0.get("witgen_s", 0):.3f} wf_store_s={r0.get("ws_store_s", 0):.3f} '
              f'wf_load_s={r0.get("ws_load_s", 0):.3f} wf_peak_gb={r0.get("ws_peak_gb", 0):.3f} '
              f'wf_raw_tensors={r0.get("ws_raw_tensors", 0)} wf_pageable_tensors={r0.get("ws_pageable_tensors", 0)} ')
    print(f'SPLIT_SUMMARY parts={P} prep_s={prep_s:.3f} makespan_s={makespan_s:.3f} wall_makespan_s={wall_makespan_s:.3f} e2e_s={e2e_s:.3f} '
          f'seq_sum_s={seq_sum_s:.3f} {wf}witgen_first={bool(args.witgen_first)} early_carry={args.early_carry} '
          f'chain_ok={chain_ok} carry_ok={carry_ok} status={status}')
    with open(args.out_json, 'w') as f:
        json.dump({
            'config': {'model_size': args.model_size, 'seq_len': args.seq_len, 'embed_dim': embed_dim, 'hidden_dim': hidden_dim,
                       'num_layers': num_layers, 'sizes': sizes, 'devices': devices, 'wave_width': W, 'eps': eps,
                       'early_carry': args.early_carry},
            'partitions': results, 'witgen_first': args.witgen_first,
            'prep_s': prep_s, 'makespan_s': makespan_s, 'wall_makespan_s': wall_makespan_s, 'e2e_s': e2e_s,
            'seq_sum_s': seq_sum_s, 'chain_ok': chain_ok, 'carry_ok': carry_ok, 'status': status,
        }, f, indent=2)
    exit(0 if status == 'OK' else 1)
