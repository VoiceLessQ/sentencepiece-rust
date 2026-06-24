//! Darts-clone double-array trie — `common_prefix_search` / `traverse`.
//!
//! Needed by two later milestones:
//!   * the charsmap normaliser (`normalizer.cc` replays a trie embedded in
//!     `precompiled_charsmap`), and
//!   * Unigram segmentation (`unigram_model.cc` traverses a trie built over the
//!     vocabulary to enumerate pieces starting at each position).
//!
//! Port reference: `Reference/sentencepiece/third_party/darts_clone/darts.h`
//! (inference needs only `traverse` / `commonPrefixSearch`, not the builder).
//!
//! Stub: not implemented in v0.1 (BPE inference does not use the trie).

#![allow(dead_code)]

/// A read-only view over a Darts double-array built elsewhere (the charsmap
/// blob, or — later — a trie we build over the vocabulary).
pub struct DoubleArray<'a> {
    _units: &'a [u32],
}

impl<'a> DoubleArray<'a> {
    pub fn from_units(units: &'a [u32]) -> Self {
        DoubleArray { _units: units }
    }

    /// Incrementally walk one key position. Returns the value at the node, or a
    /// sentinel for "no value" / "no such transition".
    ///
    /// TODO(v0.2): port `DoubleArray::traverse` from darts.h.
    pub fn traverse(&self, _key: &[u8], _node_pos: &mut usize, _key_pos: &mut usize) -> i32 {
        unimplemented!("Darts trie traverse is a v0.2 milestone")
    }
}
