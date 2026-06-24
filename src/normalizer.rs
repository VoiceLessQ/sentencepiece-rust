//! Text normalisation.
//!
//! Port reference: `Reference/sentencepiece/src/normalizer.cc`
//! (`Normalize`, `NormalizePrefix`, `DecodePrecompiledCharsMap`).
//!
//! When the model carries a `precompiled_charsmap` we replay its Darts trie to
//! fold Unicode (NFKC/NMT rules), exactly as upstream does; otherwise prefixes
//! pass through one UTF-8 character at a time. Whitespace handling (dummy prefix,
//! space collapsing, `▁` escaping) is layered on top in both cases.
//!
//! Known limitation: the user-defined-symbol `PrefixMatcher` that protects custom
//! pieces from normalisation is not yet ported (models without user_defined
//! symbols — e.g. the test fixture — are unaffected).

use std::borrow::Cow;

use crate::model::NormalizerSpec;
use crate::trie::DoubleArray;

/// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's visible space.
pub const SPACE_SYMBOL: &str = "\u{2581}";

/// A decoded `precompiled_charsmap`: a Darts trie plus the blob of
/// null-terminated replacement strings its values index into.
struct Charsmap {
    trie: DoubleArray,
    normalized: Vec<u8>,
}

impl Charsmap {
    /// Decode the `<u32 trie_size><trie><normalized strings>` blob.
    /// Ports `DecodePrecompiledCharsMap`.
    fn decode(blob: &[u8]) -> Option<Charsmap> {
        if blob.len() <= 4 {
            return None;
        }
        let size = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        if size >= blob.len() - 4 || size % 4 != 0 {
            return None;
        }
        let trie = DoubleArray::from_trie_bytes(&blob[4..4 + size]);
        let normalized = blob[4 + size..].to_vec();
        Some(Charsmap { trie, normalized })
    }

    /// Longest-match normalisation of the prefix of `input`.
    /// Returns `(replacement, consumed_input_bytes)`. Ports `NormalizePrefix`.
    fn normalize_prefix<'a>(&'a self, input: &'a [u8]) -> (Cow<'a, [u8]>, usize) {
        let mut longest_len = 0usize;
        let mut longest_val = 0usize;
        for (val, len) in self.trie.common_prefix_search(input) {
            if longest_len == 0 || len > longest_len {
                longest_len = len;
                longest_val = val as usize;
            }
        }

        if longest_len == 0 || longest_len > input.len() || longest_val >= self.normalized.len() {
            // No rule: emit one UTF-8 character unchanged. (Input originates from
            // a Rust &str, so it is always valid UTF-8; the malformed-byte path
            // from the C++ is unreachable here.)
            let len = utf8_len(input[0]).min(input.len());
            (Cow::Borrowed(&input[..len]), len)
        } else {
            let start = longest_val;
            let end = self.normalized[start..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| start + p)
                .unwrap_or(self.normalized.len());
            (Cow::Borrowed(&self.normalized[start..end]), longest_len)
        }
    }
}

pub struct Normalizer {
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
    charsmap: Option<Charsmap>,
}

impl Normalizer {
    pub fn new(spec: &NormalizerSpec, treat_whitespace_as_suffix: bool) -> Normalizer {
        let charsmap = if spec.precompiled_charsmap.is_empty() {
            None
        } else {
            Charsmap::decode(&spec.precompiled_charsmap)
        };
        Normalizer {
            add_dummy_prefix: spec.add_dummy_prefix,
            remove_extra_whitespaces: spec.remove_extra_whitespaces,
            escape_whitespaces: spec.escape_whitespaces,
            treat_whitespace_as_suffix,
            charsmap,
        }
    }

    /// Normalise one prefix: charsmap rule if present, else identity per char.
    fn normalize_prefix<'a>(&'a self, input: &'a [u8]) -> (Cow<'a, [u8]>, usize) {
        match &self.charsmap {
            Some(cm) => cm.normalize_prefix(input),
            None => {
                let len = utf8_len(input[0]).min(input.len());
                (Cow::Borrowed(&input[..len]), len)
            }
        }
    }

    /// Normalise `text` into the form fed to the segmentation model.
    /// Ports the byte-level `Normalizer::Normalize` loop.
    pub fn normalize(&self, text: &str) -> String {
        let input = text.as_bytes();
        let space: &[u8] = if self.escape_whitespaces {
            SPACE_SYMBOL.as_bytes()
        } else {
            b" "
        };

        let mut out: Vec<u8> = Vec::with_capacity(input.len() * 2);

        // Leading dummy whitespace.
        if !self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            out.extend_from_slice(space);
        }

        let mut is_prev_space = self.remove_extra_whitespaces;
        let mut pos = 0;
        while pos < input.len() {
            let (sp_cow, consumed) = self.normalize_prefix(&input[pos..]);
            let mut sp: &[u8] = sp_cow.as_ref();

            // Drop leading spaces if the previous piece ended in whitespace.
            while is_prev_space {
                match sp.first() {
                    Some(&b' ') => sp = &sp[1..],
                    _ => break,
                }
            }

            if !sp.is_empty() {
                for &b in sp {
                    if b == b' ' {
                        out.extend_from_slice(space);
                    } else {
                        out.push(b);
                    }
                }
                is_prev_space = sp.last() == Some(&b' ');
            }

            pos += consumed.max(1);
            if !self.remove_extra_whitespaces {
                is_prev_space = false;
            }
        }

        // Trim trailing space symbols.
        if self.remove_extra_whitespaces {
            while out.ends_with(space) {
                out.truncate(out.len() - space.len());
            }
        }

        // Trailing dummy whitespace.
        if self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            out.extend_from_slice(space);
        }

        String::from_utf8(out).unwrap_or_default()
    }
}

/// Byte length of a UTF-8 sequence from its leading byte.
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1,
    }
}
