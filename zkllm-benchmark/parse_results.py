"""Parse run_experiments.sh's summary.log (and the per-run JSON files) into a CSV table.

Usage: python parse_results.py [results_dir] [--all]

summary.log is append-only across sweeps into the same directory; by default only the latest
row per (model, config) is kept. --all keeps every row.
"""
import re, sys, csv, json, glob, os

keep_all = '--all' in sys.argv
argv = [a for a in sys.argv[1:] if a != '--all']
out_dir = argv[0] if argv else 'results'
summary_path = os.path.join(out_dir, 'summary.log')

FIELD_RE = re.compile(r'(\w+)=([^\s]+)')

rows = []
with open(summary_path) as f:
    for line in f:
        if not line.startswith('RESULT'):
            continue
        fields = dict(FIELD_RE.findall(line))
        rows.append(fields)

if not keep_all:
    latest = {}
    for r in rows:
        latest[(r.get('model'), r.get('tag'))] = r
    rows = list(latest.values())

cols = ['model', 'tag', 'rc', 'wall_s', 'peak_mem_MiB', 'parts', 'prep_s', 'witgen_chain_s',
        'makespan_s', 'wall_makespan_s', 'e2e_s', 'prover_max_s', 'prover_sum_s', 'early_carry', 'chain_ok', 'carry_ok', 'status']
out_csv = os.path.join(out_dir, 'summary.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f'wrote {out_csv} ({len(rows)} rows)')

# Compact end-to-end table: for each model, zkLLM baseline vs each zkMod config.
print()
header = f'{"model":10s} {"config":14s} {"end_to_end_s":>13s} {"peak_mem_GB":>12s} {"status":>8s}'
print(header)
print('-' * len(header))
for r in rows:
    e2e = r.get('e2e_s')
    if not e2e and r.get('wall_makespan_s'):
        # Older SPLIT_SUMMARY lines lack e2e_s; wall_makespan_s excludes the witgen chain (prep_s).
        e2e = float(r['wall_makespan_s']) + float(r.get('prep_s', 0))
    e2e = e2e or r.get('wall_s') or ''
    try:
        e2e = f'{float(e2e):.1f}'
    except ValueError:
        pass
    try:
        mem_gb = f'{int(r["peak_mem_MiB"]) / 1024:.1f}'
    except (KeyError, ValueError):
        mem_gb = ''
    # No SPLIT/PIPELINE summary line means the harness died before proving (e.g. an argparse error).
    status = r.get('status') or 'CRASH'
    print(f'{r.get("model",""):10s} {r.get("tag",""):14s} {e2e:>13s} {mem_gb:>12s} {status:>8s}')
