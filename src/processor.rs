//! `SentencePieceProcessor` — load a model and encode/decode text.
//!
//! Orchestration mirrors `sentencepiece_processor.cc`: normalise, run the
//! segmentation model, then map pieces to ids applying byte-fallback.

use std::path::Path;

use crate::bpe;
use crate::error::{Error, Result};
use crate::model::{ModelProto, ModelType};
use crate::normalizer::{Normalizer, SPACE_SYMBOL};
use crate::vocab::Vocab;

pub struct SentencePieceProcessor {
    model: ModelProto,
    vocab: Vocab,
    normalizer: Normalizer,
}

impl SentencePieceProcessor {
    /// Load a serialised `.model` file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Load a model from raw `.model` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let model = ModelProto::parse(bytes)?;
        let vocab = Vocab::from_model(&model);
        let normalizer = Normalizer::new(&model.normalizer, model.treat_whitespace_as_suffix);
        Ok(SentencePieceProcessor {
            model,
            vocab,
            normalizer,
        })
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub fn model_type(&self) -> ModelType {
        self.model.model_type
    }

    pub fn piece_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn id_to_piece(&self, id: i32) -> Option<&str> {
        self.vocab.id_to_piece(id)
    }

    /// Encode `text` into a sequence of piece ids.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        // v0.1 guard: charsmap folding for non-ASCII isn't implemented yet, so
        // refuse rather than emit wrong tokens. (Pure-ASCII input is exact.)
        if self.normalizer.needs_charsmap() && !text.is_ascii() {
            return Err(Error::Unsupported(
                "non-ASCII input requires the charsmap normaliser (v0.2)",
            ));
        }

        let spans = match self.model.model_type {
            ModelType::Bpe => {
                let norm = self.normalizer.normalize(text);
                if norm.is_empty() {
                    return Ok(Vec::new());
                }
                bpe::encode(norm.as_bytes(), &self.vocab)
                    .into_iter()
                    .map(|s| (norm.as_bytes()[s.start..s.start + s.len].to_vec(), s.id))
                    .collect::<Vec<_>>()
            }
            ModelType::Unigram => {
                return Err(Error::Unsupported("Unigram model (v0.3)"));
            }
            ModelType::Word | ModelType::Char => {
                return Err(Error::Unsupported("Word/Char models"));
            }
        };

        // Post-process, mirroring PopulateSentencePieceText in
        // sentencepiece_processor.cc: byte-fallback decomposition, and merging a
        // continuous run of unknown pieces into a single <unk>.
        let mut ids = Vec::with_capacity(spans.len());
        let mut prev_unk = false;
        for (bytes, id) in spans {
            let is_unk = id == self.vocab.unk_id;
            if is_unk && self.vocab.byte_fallback {
                // Decompose the unknown piece into its raw UTF-8 bytes.
                for &b in &bytes {
                    let bid = self.vocab.byte_id(b);
                    ids.push(if bid >= 0 { bid } else { self.vocab.unk_id });
                }
            } else if is_unk && prev_unk {
                // Continuous unknown run: merged into the previous <unk>.
            } else {
                ids.push(id);
            }
            prev_unk = is_unk;
        }
        Ok(ids)
    }

    /// Decode a sequence of ids back into text.
    ///
    /// v0.1 covers the common path (normal pieces + `▁` -> space). Byte-piece
    /// reassembly (`<0xXX>` runs -> UTF-8) is a follow-up once byte-fallback
    /// encode is verified.
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let mut out = String::new();
        for &id in ids {
            match self.vocab.id_to_piece(id) {
                Some(piece) => out.push_str(piece),
                None => return Err(Error::Model(format!("id {id} out of range"))),
            }
        }
        let detok = out.replace(SPACE_SYMBOL, " ");
        Ok(detok.trim_start().to_string())
    }
}
