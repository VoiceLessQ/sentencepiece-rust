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

## Why it's tractable

The scary parts of the upstream repo don't apply to inference:

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

To reproduce the broad run, point `gen_oracle.py` at the upstream corpora under
`../Reference/sentencepiece/data/` instead of the small `corpus_*.txt` files.

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

Each module names the reference file it ports in its header comment. The upstream
source lives at `../Reference/sentencepiece` (local, gitignored).

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

`tests/oracle.rs` is skipped (green) until `oracle/cases.tsv` exists, then asserts
our token ids match Python's exactly for every case.

## License

Apache-2.0, matching upstream SentencePiece. The bundled `tests/models/*.model`
fixtures originate from the SentencePiece repository.
