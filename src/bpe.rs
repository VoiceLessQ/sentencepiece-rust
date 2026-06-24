//! BPE segmentation.
//!
//! Faithful port of the deterministic path (alpha = 0) of
//! `bpe::Model::SampleEncode` in `Reference/sentencepiece/src/bpe_model.cc`.
//!
//! Algorithm:
//!   1. Split the normalised bytes into symbols. The reference does a longest
//!      prefix match against USER_DEFINED/UNUSED pieces; for a standard trained
//!      model there are none, so v0.1 splits per UTF-8 character. (USER_DEFINED
//!      matching needs the Darts trie and is tracked for a later milestone.)
//!   2. Keep symbols in a doubly linked list. Seed a max-heap with every
//!      adjacent pair whose concatenation is a known piece, keyed by piece score.
//!   3. Repeatedly pop the best pair, skip if stale, merge it, then push the two
//!      new neighbouring pairs. Tie-break: higher score, then smaller left index.
//!   4. Walk the surviving list, mapping each symbol's bytes back to an id
//!      (`unk_id` when absent — byte-fallback is applied later by the processor).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::vocab::Vocab;

/// One output symbol: a byte range `[start, start+len)` into the normalised
/// input and the vocabulary id it maps to.
pub struct Span {
    pub start: usize,
    pub len: usize,
    pub id: i32,
}

struct Symbol {
    start: usize,
    len: usize, // 0 marks a symbol that was merged away
    prev: i64,
    next: i64,
}

struct Pair {
    score: f32,
    left: usize,
    right: usize,
    size: usize, // expected combined byte length, for staleness detection
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pair {}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: "greater" = popped first.
        // Higher score wins; on a tie the smaller left index wins.
        match self.score.total_cmp(&other.score) {
            Ordering::Equal => other.left.cmp(&self.left),
            ord => ord,
        }
    }
}

pub fn encode(norm: &[u8], vocab: &Vocab) -> Vec<Span> {
    if norm.is_empty() {
        return Vec::new();
    }

    // 1. Per-character symbols.
    let mut symbols: Vec<Symbol> = Vec::new();
    let mut i = 0;
    while i < norm.len() {
        let clen = utf8_len(norm[i]).min(norm.len() - i);
        let idx = symbols.len() as i64;
        symbols.push(Symbol {
            start: i,
            len: clen,
            prev: idx - 1,
            next: -1, // patched below
        });
        i += clen;
    }
    let n = symbols.len();
    for (k, sym) in symbols.iter_mut().enumerate() {
        sym.next = if k + 1 < n { (k + 1) as i64 } else { -1 };
    }

    let mut agenda: BinaryHeap<Pair> = BinaryHeap::new();

    // Try to register the pair (left, right) if their concatenation is a piece.
    let maybe_add = |symbols: &[Symbol], agenda: &mut BinaryHeap<Pair>, left: i64, right: i64| {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (left as usize, right as usize);
        let start = symbols[l].start;
        let size = symbols[l].len + symbols[r].len;
        if let Some(id) = vocab.piece_to_id(&norm[start..start + size]) {
            if id != vocab.unk_id {
                agenda.push(Pair {
                    score: vocab.score(id),
                    left: l,
                    right: r,
                    size,
                });
            }
        }
    };

    // 2. Seed with adjacent pairs.
    for k in 0..n.saturating_sub(1) {
        maybe_add(&symbols, &mut agenda, k as i64, (k + 1) as i64);
    }

    // 3. Merge loop.
    while let Some(top) = agenda.pop() {
        let (l, r) = (top.left, top.right);
        // Stale if either side was consumed or sizes no longer match.
        if symbols[l].len == 0 || symbols[r].len == 0 {
            continue;
        }
        if symbols[l].len + symbols[r].len != top.size || symbols[l].next != r as i64 {
            continue;
        }

        // Merge r into l.
        symbols[l].len += symbols[r].len;
        let r_next = symbols[r].next;
        symbols[l].next = r_next;
        if r_next >= 0 {
            symbols[r_next as usize].prev = l as i64;
        }
        symbols[r].len = 0;

        let l_prev = symbols[l].prev;
        maybe_add(&symbols, &mut agenda, l_prev, l as i64);
        maybe_add(&symbols, &mut agenda, l as i64, r_next);
    }

    // 4. Walk surviving symbols, resolving ids.
    let mut out = Vec::new();
    let mut idx: i64 = 0;
    while idx >= 0 {
        let s = &symbols[idx as usize];
        let bytes = &norm[s.start..s.start + s.len];
        let id = vocab.piece_to_id(bytes).unwrap_or(vocab.unk_id);
        out.push(Span {
            start: s.start,
            len: s.len,
            id,
        });
        idx = s.next;
    }
    out
}

/// Byte length of a UTF-8 sequence from its leading byte.
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1, // continuation/invalid: treat as single byte
    }
}
