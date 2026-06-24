# sentencepiece-rs

A from-scratch, pure-Rust port of [SentencePiece](https://github.com/google/sentencepiece)
**inference** (encode/decode), differentially verified against the upstream
C++/Python implementation.

This is **not** a binding. The existing `sentencepiece` crate wraps Google's C++
library; this crate reimplements the tokenisation algorithms in Rust from the
reference source, the same way the author's other library ports (`urlparse-rs`,
`robotparser-rs`, `fnmatch-rs`, …) reimplement their Python originals and verify
against them.

Training is out of scope — a trained `.model` is loaded and used to tokenise.

## Correctness Goal

The aim of this crate is to be **provably faithful** to upstream SentencePiece —
not merely "close enough". That goal drives everything else:

- Every module is a line-by-line port of the reference file it names in its
  header, so behaviour can be checked against the original.
- Output is **differentially verified** against the Python `sentencepiece`
  oracle over thousands of real-text lines, both encode *and* decode — see
  [The Rust tokenizer test](#the-rust-tokenizer-test) and
  [Verification breadth](#verification-breadth). A divergence from the oracle is
  treated as a bug until proven to be an equally-optimal tie (the one documented
  precision fence).

The other properties are consequences of that discipline, not goals competing
with it:

- **Zero dependencies** — there is nothing to audit but this crate and `std`.
- **Speed** — a welcome side effect, *measured* (see
  [Speed](#speed)), but never traded against matching the oracle.

## Why

The scary parts of the upstream repo dont apply to inference:

- The 138k-line `normalization_rule.h` is compile-time charsmap *generation*; a
  trained model embeds its own `precompiled_charsmap`, which we just replay.
- The suffix-array trainer, `builder`, and `esaxx` are training-only.
- The `.proto` is tiny, so a hand-written proto2 reader replaces `prost`/`protoc`
  and keeps the crate dependency-free.

## Milestones

| Version | Scope | Verified against |
|--------:|-------|------------------|
| v0.1 | BPE segmentation; ASCII / whitespace normalisation; byte-fallback encode | Python `sentencepiece`, ASCII corpora |
| v0.2 | Darts charsmap normaliser → full Unicode input | + Unicode corpora (full-width, ligatures, CJK, …) |
| **v0.3** *(current)* | Unigram Viterbi segmentation; byte-fallback + byte-piece decode | 3 models (BPE, Unigram, BPE+byte_fallback), encode *and* decode |

Inference is feature-complete for both BPE and Unigram models, with faithful
decode (byte-piece reassembly, unk surface, dummy-prefix handling). Both
directions are differentially verified against the Python oracle.

### Verification breadth

Beyond the small committed corpora, the encoders were run over the upstream
training corpora (`botchan.txt`, 4.3k English lines; `wagahaiwa_nekodearu.txt`,
2.3k Japanese lines) through all three models — **13,264 lines, encode and
decode**. Result: **13,258 exact, 6 equal-score reorderings, 0 real mismatches.**

The 6 differences are all runs of repeated punctuation (`.......`, long `----`)
on the Unigram model: a *same-multiset* reordering of pieces, i.e. an
equally-optimal Viterbi path with identical total score. This is a documented
**precision fence** — the compiled oracle binary's optimised float arithmetic
resolves these exact ties differently than portable `f32`; both segmentations
are optimal. See the note in [src/unigram.rs](src/unigram.rs). The test counts
these separately and only fails on a genuine (different-piece) mismatch.

To reproduce the broad run, point `gen_oracle.py` at a larger corpus instead of
the small `corpus_*.txt` files — e.g. `botchan.txt` and `wagahaiwa_nekodearu.txt`
from the upstream SentencePiece repo's
[`data/`](https://github.com/google/sentencepiece/tree/master/data) directory.
Those corpora are not vendored here; download them from the original source.

## Layout

```
src/
  proto.rs       hand-written proto2 wire reader (no deps)
  model.rs       ModelProto -> typed struct (pieces, flags, normalizer spec)
  vocab.rs       piece <-> id maps, scores, byte-fallback ids
  normalizer.rs  whitespace/▁ normalisation + Darts charsmap replay
  bpe.rs         BPE merge encoder
  unigram.rs     Unigram Viterbi best-path (+ piece prefix-trie)
  trie.rs        Darts double-array (common_prefix_search / traverse)
  processor.rs   SentencePieceProcessor: load + encode/decode orchestration
examples/inspect.rs   load a model and dump a summary
tests/oracle.rs       differential test vs the Python oracle
oracle/gen_oracle.py  generates oracle/cases.tsv from Python sentencepiece
```

Each module names the reference file it ports in its header comment, from the
upstream [google/sentencepiece](https://github.com/google/sentencepiece) source.
That source is not vendored here — during development it sits as a local clone
outside the repo; clone it from the original if you want to follow the ports.

## Try it

```sh
# in-tree toolchain (see global setup); plain `cargo` works too
cargo run --example inspect -- tests/models/botchan_1000_bpe.model
```

## Verify against the oracle

```sh
# reference implementation, used only to produce the oracle
python -m pip install sentencepiece

# For each (model, output) pair, run over the ASCII corpus (>) then the
# Unicode corpus (>>):
#   botchan_1000_bpe.model          -> cases.tsv          (BPE)
#   test_oss_model_unigram.model    -> cases_unigram.tsv  (Unigram)
#   wagahaiwa_2000_bpe_byte.model   -> cases_byte.tsv     (BPE + byte_fallback)
python oracle/gen_oracle.py tests/models/botchan_1000_bpe.model \
    oracle/corpus_ascii.txt > oracle/cases.tsv
python oracle/gen_oracle.py tests/models/botchan_1000_bpe.model \
    oracle/corpus_unicode.txt >> oracle/cases.tsv
# ... likewise for the Unigram and byte-fallback fixtures.

cargo test
```

## The Rust tokenizer test

[tests/oracle.rs](tests/oracle.rs) is a plain `cargo test` integration test — pure
Rust, no Python at run time. For each fixture model it:

1. loads the `.model` with `SentencePieceProcessor` (the same public API a user
   calls);
2. reads the generated `oracle/cases*.tsv` (`hex(text)\t<ids>\thex(decode(ids))`);
3. **encodes** each text and compares the ids to the oracle's, and **decodes** the
   oracle's ids and compares the text — so both directions are checked;
4. classifies any encode difference: a *same-multiset reordering* is an
   equally-optimal Viterbi tie (accepted, counted separately); anything else is a
   genuine mismatch and **fails** the test.

It runs across all three fixtures (BPE, Unigram, BPE+byte_fallback) and prints a
breakdown, e.g.:

```
checked 117 cases against the Python oracle: 116 exact, 1 equal-score reorderings, 0 real mismatches
```

The test **self-skips green** on a fresh checkout until the `oracle/cases*.tsv`
files exist, so `cargo test` never fails just because the oracle hasnt been
generated yet. The oracle (Python `sentencepiece`) is only used to *produce* those
files; the test itself, and the crate, need nothing but Rust.

## Speed

A rough throughput cross-check — [examples/bench.rs](examples/bench.rs) vs
[oracle/bench_oracle.py](oracle/bench_oracle.py), same per-line workload, release
build, single thread (numbers from one machine; run it yourself for yours):

| model / corpus | encode, Rust | encode, Py `sp` | decode, Rust | decode, Py `sp` |
|------------------------|-----------|-----------|---------|----------|
| BPE / English          |  6.7 MB/s |  2.9 MB/s | 28 MB/s | 5.9 MB/s |
| Unigram / English      | 10.0 MB/s |  4.8 MB/s | 26 MB/s | 5.5 MB/s |
| BPE + byte / Japanese  | 24.5 MB/s |  6.2 MB/s | 50 MB/s | 6.9 MB/s |

Read this as "the port runs at a healthy native throughput", **not** "Rust beats
C++". The Python column includes per-call CPython/FFI overhead (the C++ core in
batch mode would be faster), and this crate is straightforward single-threaded
inference tuned for *matching the oracle*, not for speed (e.g. the normaliser
re-runs `common_prefix_search` per position). Even so, native Rust encode is
~2–4× and decode ~5–7× the per-line Python path here, with no native dependency
to link.

```sh
cargo run --release --example bench -- tests/models/botchan_1000_bpe.model <corpus> 50
python oracle/bench_oracle.py        tests/models/botchan_1000_bpe.model <corpus> 50
```

## License

Apache-2.0, matching upstream SentencePiece. The bundled `tests/models/*.model`
fixtures originate from the SentencePiece repository.
