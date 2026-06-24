//! Text normalisation.
//!
//! Port reference: `Reference/sentencepiece/src/normalizer.cc`.
//!
//! The full normaliser replays a precompiled Darts charsmap (`precompiled_charsmap`)
//! to fold Unicode, then handles whitespace. v0.1 implements only the
//! **whitespace half** (identity charsmap), which is exact for ASCII / already-NFKC
//! input. Non-ASCII normalisation is deferred until [`crate::trie`] lands — see
//! the milestone roadmap in README.md.

use crate::model::NormalizerSpec;

/// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's visible space.
pub const SPACE_SYMBOL: &str = "\u{2581}";

pub struct Normalizer {
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
    has_charsmap: bool,
}

impl Normalizer {
    pub fn new(spec: &NormalizerSpec, treat_whitespace_as_suffix: bool) -> Normalizer {
        Normalizer {
            add_dummy_prefix: spec.add_dummy_prefix,
            remove_extra_whitespaces: spec.remove_extra_whitespaces,
            escape_whitespaces: spec.escape_whitespaces,
            treat_whitespace_as_suffix,
            has_charsmap: !spec.precompiled_charsmap.is_empty(),
        }
    }

    /// True when this model needs charsmap folding we don't yet implement.
    /// The processor uses this to refuse non-ASCII input loudly rather than
    /// silently producing wrong tokens.
    pub fn needs_charsmap(&self) -> bool {
        self.has_charsmap
    }

    /// Normalise `text` into the form fed to the segmentation model:
    /// collapsed/escaped whitespace with the `▁` meta symbol.
    pub fn normalize(&self, text: &str) -> String {
        // 1. Collapse runs of ASCII space and trim, if requested.
        let collapsed = if self.remove_extra_whitespaces {
            collapse_spaces(text)
        } else {
            text.to_string()
        };

        if collapsed.is_empty() {
            return String::new();
        }

        // 2. Dummy prefix/suffix so "world" and " world" tokenise alike.
        let with_dummy = if self.add_dummy_prefix {
            if self.treat_whitespace_as_suffix {
                format!("{collapsed} ")
            } else {
                format!(" {collapsed}")
            }
        } else {
            collapsed
        };

        // 3. Escape spaces with the meta symbol.
        if self.escape_whitespaces {
            with_dummy.replace(' ', SPACE_SYMBOL)
        } else {
            with_dummy
        }
    }
}

/// Collapse runs of ASCII space to a single space and trim leading/trailing.
fn collapse_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true; // leading-trim
    for ch in text.chars() {
        if ch == ' ' {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}
