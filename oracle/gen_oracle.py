#!/usr/bin/env python3
"""Generate a differential-test oracle from the reference Python sentencepiece.

For each input line we emit one TSV record:

    <hex(utf8(text))> \t <id id id ...>

ids are the canonical correctness signal (they fully determine the segmentation).
Text is hex-encoded so it can carry any bytes without escaping headaches; the
Rust harness (tests/oracle.rs) hex-decodes and compares.

Usage:
    python gen_oracle.py <model.model> <corpus.txt> > cases.tsv

Requires the reference implementation:
    pip install sentencepiece

This is the oracle, by design a different implementation than the crate under
test — exactly the pattern used by the author's other differential ports.
"""
import sys


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: gen_oracle.py <model.model> <corpus.txt>\n")
        return 2

    try:
        import sentencepiece as spm
    except ImportError:
        sys.stderr.write("missing dependency: pip install sentencepiece\n")
        return 1

    model_path, corpus_path = sys.argv[1], sys.argv[2]
    sp = spm.SentencePieceProcessor(model_file=model_path)

    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.rstrip("\n")
            if not text:
                continue
            ids = sp.encode(text, out_type=int)
            hex_text = text.encode("utf-8").hex()
            sys.stdout.write(hex_text + "\t" + " ".join(map(str, ids)) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
