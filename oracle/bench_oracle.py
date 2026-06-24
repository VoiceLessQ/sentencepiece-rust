#!/usr/bin/env python3
"""Throughput of the reference Python `sentencepiece` (C++ core), for comparison
with `cargo run --release --example bench`.

Usage:
    python oracle/bench_oracle.py <model> <corpus> [iters]

Per-line encode/decode (matching examples/bench.rs), so the Python call overhead
is included — this is a rough cross-check, not a tuned benchmark.
"""
import sys
import time


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: bench_oracle.py <model> <corpus> [iters]\n")
        return 2
    import sentencepiece as spm

    model, corpus = sys.argv[1], sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    sp = spm.SentencePieceProcessor(model_file=model)
    lines = [ln.rstrip("\n") for ln in open(corpus, encoding="utf-8") if ln.strip()]
    total_bytes = sum(len(ln.encode("utf-8")) for ln in lines)

    for ln in lines:
        sp.encode(ln)

    t = time.perf_counter()
    for _ in range(iters):
        for ln in lines:
            sp.encode(ln)
    enc = time.perf_counter() - t

    encoded = [sp.encode(ln) for ln in lines]
    t = time.perf_counter()
    for _ in range(iters):
        for ids in encoded:
            sp.decode(ids)
    dec = time.perf_counter() - t

    n = len(lines) * iters
    mb = total_bytes * iters / 1e6
    print(f"model:  {model}")
    print(f"corpus: {corpus} ({len(lines)} lines, {total_bytes} bytes) x{iters} iters")
    print(f"encode: {enc:.3f}s  {n / enc:.0f} lines/s  {mb / enc:.1f} MB/s")
    print(f"decode: {dec:.3f}s  {n / dec:.0f} lines/s  {mb / dec:.1f} MB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
