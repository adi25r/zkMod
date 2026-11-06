#!/bin/bash
# Build an allocation-tracing copy of ./partition in $1 (default /tmp/zkllm-memtrace), leaving
# the real build untouched. Every cudaMalloc/cudaFree is recorded (memtrace.hpp); run the
# traced binary with MEMTRACE_OUT=<prefix> to get <prefix>_peak.csv and <prefix>_timeline.csv.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
OUT="${1:-/tmp/zkllm-memtrace}"
mkdir -p "$OUT"
cp "$SRC"/*.cu "$SRC"/*.cuh "$SRC"/*.hpp "$SRC"/*.cpp "$SRC"/Makefile "$HERE/memtrace.hpp" "$OUT/"
cd "$OUT"
python3 - <<'PY'
s = open('bls12-381.cuh').read()
def patch(a, b):
    global s
    assert a in s, 'memtrace build: pattern not found in bls12-381.cuh: ' + a
    s = s.replace(a, b, 1)
patch('struct DeviceAllocStats {', '#include "memtrace.hpp"\nstruct DeviceAllocStats {')
patch('    st.live += n;\n', '    st.live += n;\n    mt_on_alloc(*p, n);\n')
patch('    if (it != st.sizes.end()) { st.live -= it->second; st.sizes.erase(it); }\n',
      '    if (it != st.sizes.end()) { st.live -= it->second; st.sizes.erase(it); }\n    mt_on_free(p);\n')
open('bls12-381.cuh', 'w').write(s)
m = open('Makefile').read()
m = m.replace('NVCC_FLAGS := -arch=$(ARCH) -std=c++17',
              'NVCC_FLAGS := -arch=$(ARCH) -std=c++17 -g -Xcompiler -rdynamic -Xcompiler -fno-omit-frame-pointer')
open('Makefile', 'w').write(m)
PY
make ARCH="${ARCH:-sm_86}" partition
echo "traced binary: $OUT/partition"
