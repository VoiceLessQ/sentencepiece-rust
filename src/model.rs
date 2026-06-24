//! Parsed `ModelProto` — the subset of fields needed for inference.
//!
//! Field numbers and defaults mirror
//! `Reference/sentencepiece/src/sentencepiece_model.proto`.

use crate::error::{Error, Result};
use crate::proto::Reader;

/// Segmentation algorithm stored in the model (`TrainerSpec.model_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Unigram,
    Bpe,
    Word,
    Char,
}

/// Piece category (`ModelProto.SentencePiece.Type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Unused,
}

impl PieceType {
    fn from_i32(v: i32) -> PieceType {
        match v {
            1 => PieceType::Normal,
            2 => PieceType::Unknown,
            3 => PieceType::Control,
            4 => PieceType::UserDefined,
            6 => PieceType::Byte,
            5 => PieceType::Unused,
            _ => PieceType::Normal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SentencePiece {
    pub piece: String,
    pub score: f32,
    pub kind: PieceType,
}

/// `NormalizerSpec` — text normalisation parameters.
#[derive(Debug, Clone)]
pub struct NormalizerSpec {
    /// Compiled Darts double-array charsmap (empty for identity normalisation).
    pub precompiled_charsmap: Vec<u8>,
    pub add_dummy_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
}

impl Default for NormalizerSpec {
    fn default() -> Self {
        NormalizerSpec {
            precompiled_charsmap: Vec::new(),
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

/// The decoded model: vocabulary pieces plus the flags inference depends on.
#[derive(Debug, Clone)]
pub struct ModelProto {
    pub pieces: Vec<SentencePiece>,
    pub model_type: ModelType,
    pub byte_fallback: bool,
    pub treat_whitespace_as_suffix: bool,
    pub unk_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    pub normalizer: NormalizerSpec,
}

impl ModelProto {
    /// Parse a serialised `.model` file.
    pub fn parse(bytes: &[u8]) -> Result<ModelProto> {
        let mut model = ModelProto {
            pieces: Vec::new(),
            model_type: ModelType::Unigram,
            byte_fallback: false,
            treat_whitespace_as_suffix: false,
            unk_id: 0,
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            normalizer: NormalizerSpec::default(),
        };

        let mut r = Reader::new(bytes);
        while let Some((number, value)) = r.next()? {
            match number {
                // repeated SentencePiece pieces = 1;
                1 => {
                    if let Some(b) = value.as_bytes() {
                        model.pieces.push(parse_piece(b)?);
                    }
                }
                // optional TrainerSpec trainer_spec = 2;
                2 => {
                    if let Some(b) = value.as_bytes() {
                        parse_trainer_spec(b, &mut model)?;
                    }
                }
                // optional NormalizerSpec normalizer_spec = 3;
                3 => {
                    if let Some(b) = value.as_bytes() {
                        model.normalizer = parse_normalizer_spec(b)?;
                    }
                }
                _ => {} // self_test_data, denormalizer_spec, extensions: ignored
            }
        }

        if model.pieces.is_empty() {
            return Err(Error::Model("model contains no pieces".into()));
        }
        Ok(model)
    }
}

fn parse_piece(bytes: &[u8]) -> Result<SentencePiece> {
    let mut piece = String::new();
    let mut score = 0.0f32;
    let mut kind = PieceType::Normal;
    let mut r = Reader::new(bytes);
    while let Some((number, value)) = r.next()? {
        match number {
            1 => piece = value.as_str().unwrap_or_default(),
            2 => score = value.as_f32().unwrap_or(0.0),
            3 => kind = PieceType::from_i32(value.as_i32().unwrap_or(1)),
            _ => {}
        }
    }
    Ok(SentencePiece { piece, score, kind })
}

fn parse_normalizer_spec(bytes: &[u8]) -> Result<NormalizerSpec> {
    let mut spec = NormalizerSpec::default();
    let mut r = Reader::new(bytes);
    while let Some((number, value)) = r.next()? {
        match number {
            2 => spec.precompiled_charsmap = value.as_bytes().unwrap_or_default().to_vec(),
            3 => spec.add_dummy_prefix = value.as_bool().unwrap_or(true),
            4 => spec.remove_extra_whitespaces = value.as_bool().unwrap_or(true),
            5 => spec.escape_whitespaces = value.as_bool().unwrap_or(true),
            _ => {}
        }
    }
    Ok(spec)
}

fn parse_trainer_spec(bytes: &[u8], model: &mut ModelProto) -> Result<()> {
    let mut r = Reader::new(bytes);
    while let Some((number, value)) = r.next()? {
        match number {
            3 => {
                model.model_type = match value.as_i32().unwrap_or(1) {
                    2 => ModelType::Bpe,
                    3 => ModelType::Word,
                    4 => ModelType::Char,
                    _ => ModelType::Unigram,
                }
            }
            24 => model.treat_whitespace_as_suffix = value.as_bool().unwrap_or(false),
            35 => model.byte_fallback = value.as_bool().unwrap_or(false),
            40 => model.unk_id = value.as_i32().unwrap_or(0),
            41 => model.bos_id = value.as_i32().unwrap_or(1),
            42 => model.eos_id = value.as_i32().unwrap_or(2),
            43 => model.pad_id = value.as_i32().unwrap_or(-1),
            _ => {}
        }
    }
    Ok(())
}
