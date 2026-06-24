//! A from-scratch, pure-Rust port of [SentencePiece] **inference**
//! (encode/decode), differentially verified against the upstream C++/Python
//! implementation — the same harness used across the author's other library
//! ports.
//!
//! This is **not** a wrapper around the C++ library (that already exists as the
//! `sentencepiece` crate). The algorithms here are reimplemented in Rust from
//! the reference source in `projects/Reference/sentencepiece`.
//!
//! # Status
//! Inference is feature-complete for BPE and Unigram models, verified against
//! the Python `sentencepiece` oracle:
//! - **v0.1:** BPE segmentation + ASCII/whitespace normalisation.
//! - **v0.2:** Darts charsmap normaliser → full Unicode input (full-width
//!   folding, ligatures, CJK, …).
//! - **v0.3 (current):** Unigram Viterbi segmentation (the default model type),
//!   plus faithful decode (byte-piece reassembly, unk surface, dummy-prefix).
//!
//! Encode *and* decode are verified against the oracle across BPE, Unigram, and
//! byte-fallback models (117 cases). Training is intentionally out of scope —
//! this crate loads a trained `.model` and tokenises with it.
//!
//! # Example
//! ```no_run
//! use sentencepiece_rust::SentencePieceProcessor;
//! let sp = SentencePieceProcessor::open("tests/models/botchan_1000_bpe.model")?;
//! let ids = sp.encode("hello world")?;
//! let text = sp.decode(&ids)?;
//! # Ok::<(), sentencepiece_rust::Error>(())
//! ```
//!
//! [SentencePiece]: https://github.com/google/sentencepiece

mod bpe;
mod error;
mod model;
mod normalizer;
mod processor;
mod proto;
mod trie;
mod unigram;
mod vocab;

pub use error::{Error, Result};
pub use model::{ModelType, PieceType};
pub use processor::SentencePieceProcessor;
pub use vocab::Vocab;
