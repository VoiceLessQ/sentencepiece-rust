//! Vocabulary: piece <-> id maps and the accessors the encoders need.
//!
//! Mirrors the lookup surface of `ModelInterface` in
//! `Reference/sentencepiece/src/model_interface.{h,cc}`.

use std::collections::HashMap;

use crate::model::{ModelProto, PieceType};

pub struct Vocab {
    pieces: Vec<(String, f32, PieceType)>,
    piece_to_id: HashMap<Vec<u8>, i32>,
    /// byte value (0..=255) -> id of its `<0xXX>` piece, when `byte_fallback`.
    byte_to_id: [i32; 256],
    pub unk_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    pub byte_fallback: bool,
    min_score: f32,
}

impl Vocab {
    pub fn from_model(model: &ModelProto) -> Vocab {
        let mut piece_to_id = HashMap::with_capacity(model.pieces.len());
        let mut byte_to_id = [-1i32; 256];
        let mut pieces = Vec::with_capacity(model.pieces.len());
        let mut min_score = f32::INFINITY;

        for (id, p) in model.pieces.iter().enumerate() {
            let id = id as i32;
            // Last write wins, matching the C++ flat_hash_map insertion order.
            piece_to_id
                .entry(p.piece.as_bytes().to_vec())
                .or_insert(id);
            if p.kind == PieceType::Byte {
                if let Some(b) = piece_to_byte(&p.piece) {
                    byte_to_id[b as usize] = id;
                }
            }
            if p.kind == PieceType::Normal && p.score < min_score {
                min_score = p.score;
            }
            pieces.push((p.piece.clone(), p.score, p.kind));
        }

        Vocab {
            pieces,
            piece_to_id,
            byte_to_id,
            unk_id: model.unk_id,
            bos_id: model.bos_id,
            eos_id: model.eos_id,
            pad_id: model.pad_id,
            byte_fallback: model.byte_fallback,
            min_score: if min_score.is_finite() { min_score } else { 0.0 },
        }
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Lowest score among normal pieces — used by the Unigram unknown penalty.
    pub fn min_score(&self) -> f32 {
        self.min_score
    }

    /// Returns the id for an exact piece, or `None` if absent.
    pub fn piece_to_id(&self, piece: &[u8]) -> Option<i32> {
        self.piece_to_id.get(piece).copied()
    }

    pub fn score(&self, id: i32) -> f32 {
        self.pieces
            .get(id as usize)
            .map(|p| p.1)
            .unwrap_or(0.0)
    }

    pub fn kind(&self, id: i32) -> Option<PieceType> {
        self.pieces.get(id as usize).map(|p| p.2)
    }

    pub fn id_to_piece(&self, id: i32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.0.as_str())
    }

    /// Id of the `<0xXX>` piece for a raw byte, when byte_fallback is enabled.
    pub fn byte_id(&self, b: u8) -> i32 {
        self.byte_to_id[b as usize]
    }

    /// Pieces that participate in segmentation (NORMAL + USER_DEFINED), as
    /// `(bytes, id)`. Used to build the Unigram trie; mirrors the membership of
    /// the C++ `pieces_` map minus UNUSED (which is skipped on match anyway).
    pub fn segmentable_pieces(&self) -> impl Iterator<Item = (&[u8], i32)> {
        self.pieces.iter().enumerate().filter_map(|(i, (p, _, k))| {
            matches!(k, PieceType::Normal | PieceType::UserDefined)
                .then_some((p.as_bytes(), i as i32))
        })
    }
}

/// `"<0x3A>"` -> `Some(0x3A)`; anything else -> `None`.
/// Matches `PieceToByte` in `model_interface.cc`.
pub(crate) fn piece_to_byte(piece: &str) -> Option<u8> {
    let bytes = piece.as_bytes();
    if bytes.len() == 6 && &bytes[..3] == b"<0x" && bytes[5] == b'>' {
        let hi = (bytes[3] as char).to_digit(16)?;
        let lo = (bytes[4] as char).to_digit(16)?;
        Some((hi * 16 + lo) as u8)
    } else {
        None
    }
}
