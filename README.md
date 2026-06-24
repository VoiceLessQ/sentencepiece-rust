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
| **v0.3** *(current)* | Unigram Viterbi segmentation (default model type) | + Unigram model; 78 BPE+Unigram cases match |

Inference is now feature-complete for both BPE and Unigram models.

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

# BPE model
python oracle/gen_oracle.py tests/models/botchan_1000_bpe.model \
    oracle/corpus_ascii.txt > oracle/cases.tsv
python oracle/gen_oracle.py tests/models/botchan_1000_bpe.model \
    oracle/corpus_unicode.txt >> oracle/cases.tsv
# Unigram model
python oracle/gen_oracle.py tests/models/test_oss_model_unigram.model \
    oracle/corpus_ascii.txt > oracle/cases_unigram.tsv
python oracle/gen_oracle.py tests/models/test_oss_model_unigram.model \
    oracle/corpus_unicode.txt >> oracle/cases_unigram.tsv

cargo test
```

`tests/oracle.rs` is skipped (green) until `oracle/cases.tsv` exists, then asserts
our token ids match Python's exactly for every case.

## License

Apache-2.0, matching upstream SentencePiece. The bundled `tests/models/*.model`
fixtures originate from the SentencePiece repository.
