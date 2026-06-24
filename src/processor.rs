//! `SentencePieceProcessor` — load a model and encode/decode text.
//!
//! Orchestration mirrors `sentencepiece_processor.cc`: normalise, run the
//! segmentation model, then map pieces to ids applying byte-fallback.

use std::path::Path;

use crate::bpe;
use crate::error::{Error, Result};
use crate::model::{ModelProto, ModelType, PieceType};
use crate::normalizer::{Normalizer, SPACE_SYMBOL};
use crate::unigram::Unigram;
use crate::vocab::Vocab;

pub struct SentencePieceProcessor {
    model: ModelProto,
    vocab: Vocab,
    normalizer: Normalizer,
    unigram: Option<Unigram>,
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
        let unigram = match model.model_type {
            ModelType::Unigram => Some(Unigram::from_vocab(&vocab)),
            _ => None,
        };
        Ok(SentencePieceProcessor {
            model,
            vocab,
            normalizer,
            unigram,
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
        let norm = self.normalizer.normalize(text);
        if norm.is_empty() {
            return Ok(Vec::new());
        }
        let nb = norm.as_bytes();

        let spans = match self.model.model_type {
            ModelType::Bpe => bpe::encode(nb, &self.vocab),
            ModelType::Unigram => {
                let u = self
                    .unigram
                    .as_ref()
                    .ok_or(Error::Model("unigram model not initialised".into()))?;
                u.encode(nb, &self.vocab)
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
        for span in spans {
            let id = span.id;
            let is_unk = id == self.vocab.unk_id;
            if is_unk && self.vocab.byte_fallback {
                // Decompose the unknown piece into its raw UTF-8 bytes.
                for &b in &nb[span.start..span.start + span.len] {
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
    /// Faithful port of `SentencePieceProcessor::Decode`: control symbols vanish,
    /// `<unk>` becomes the unk surface, the single dummy-prefix `▁` is stripped,
    /// remaining `▁` become spaces, and runs of `<0xXX>` byte pieces are
    /// reassembled into UTF-8 (invalid sequences -> U+FFFD).
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let piece_size = self.vocab.len() as i32;
        let mut text = String::new();
        let mut byte_run: Vec<u8> = Vec::new();
        let mut is_bos_ws = true;
        let mut bos_ws_seen = false;

        for &id in ids {
            if id < 0 || id >= piece_size {
                return Err(Error::Model(format!("id {id} out of range")));
            }
            let kind = self.vocab.kind(id).unwrap_or(PieceType::Normal);
            let piece = self.vocab.id_to_piece(id).unwrap_or("");

            if kind == PieceType::Byte {
                if let Some(b) = crate::vocab::piece_to_byte(piece) {
                    byte_run.push(b);
                }
                continue;
            }

            // Flush any pending byte run as UTF-8 before handling this piece.
            flush_bytes(&mut text, &mut byte_run);
            if bos_ws_seen || !text.is_empty() {
                is_bos_ws = false;
            }
            let (decoded, new_bos) = self.decode_piece(piece, kind, is_bos_ws);
            bos_ws_seen = new_bos;
            text.push_str(&decoded);
        }
        flush_bytes(&mut text, &mut byte_run);
        Ok(text)
    }

    /// Surface for one non-byte piece, plus whether it consumed a bos whitespace.
    /// Ports the `DecodeSentencePiece` lambda.
    fn decode_piece(&self, piece: &str, kind: PieceType, is_bos_ws: bool) -> (String, bool) {
        match kind {
            PieceType::Control => return (String::new(), false),
            PieceType::Unknown => return (self.model.unk_surface.clone(), false),
            _ => {}
        }

        let mut p = piece;
        let mut has_bos_ws = false;
        let ns = &self.model.normalizer;
        if is_bos_ws && (ns.add_dummy_prefix || ns.remove_extra_whitespaces) {
            if let Some(stripped) = p.strip_prefix(SPACE_SYMBOL) {
                p = stripped;
                has_bos_ws = true;
            }
            if ns.remove_extra_whitespaces {
                has_bos_ws = false;
            }
        }
        (p.replace(SPACE_SYMBOL, " "), has_bos_ws)
    }
}

/// Append a run of raw bytes to `text` as UTF-8, mapping invalid sequences to
/// U+FFFD (mirrors `ProcessBytePieces`).
fn flush_bytes(text: &mut String, run: &mut Vec<u8>) {
    if !run.is_empty() {
        text.push_str(&String::from_utf8_lossy(run));
        run.clear();
    }
}
