//! Unigram segmentation — optimised Viterbi best-path.
//!
//! Port reference: `unigram::Model::EncodeOptimized` in
//! `Reference/sentencepiece/src/unigram_model.cc` (lines ~913-1045).
//!
//! Forward DP over UTF-8 positions: for each start position, traverse the vocab
//! trie to find every piece beginning there, relax `best_path_ends_at`, fall back
//! to a single-character UNK when no single-char piece matches, then backtrack.
//!
//! Depends on [`crate::trie`]; deferred to the v0.3 milestone (see README.md).
//! BPE (v0.1) is the first verified path.

#![allow(dead_code)]

use crate::bpe::Span;
use crate::vocab::Vocab;

pub fn encode(_norm: &[u8], _vocab: &Vocab) -> Vec<Span> {
    unimplemented!("Unigram Viterbi is a v0.3 milestone; v0.1 ships BPE only")
}
