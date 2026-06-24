//! Unigram segmentation — optimised Viterbi best-path.
//!
//! Faithful port of `unigram::Model::EncodeOptimized` in
//! `Reference/sentencepiece/src/unigram_model.cc` (~lines 913-1045).
//!
//! Forward DP over UTF-8 positions: for each start position, enumerate every
//! vocab piece beginning there, relax `best_path_ends_at`, fall back to a
//! single-character `<unk>` when no single-char piece matches, then backtrack.
//!
//! The reference walks a Darts trie incrementally; we use an equivalent plain
//! byte prefix-trie over the segmentable pieces (NORMAL + USER_DEFINED — UNUSED
//! is in the C++ trie but skipped on match, so omitting it is identical, and
//! CONTROL/UNKNOWN/BYTE live in the reserved map, never the trie). Matches are
//! emitted in increasing length, matching the reference's tie-break order.

use std::collections::HashMap;

use crate::bpe::Span;
use crate::model::PieceType;
use crate::vocab::Vocab;

const UNK_PENALTY: f32 = 10.0;

#[derive(Default)]
struct Node {
    children: HashMap<u8, usize>,
    id: i32, // -1 = no piece ends here
}

struct PieceTrie {
    nodes: Vec<Node>,
}

impl PieceTrie {
    fn new() -> PieceTrie {
        PieceTrie {
            nodes: vec![Node {
                children: HashMap::new(),
                id: -1,
            }],
        }
    }

    fn insert(&mut self, key: &[u8], id: i32) {
        let mut cur = 0usize;
        for &b in key {
            cur = match self.nodes[cur].children.get(&b).copied() {
                Some(n) => n,
                None => {
                    let n = self.nodes.len();
                    self.nodes.push(Node {
                        children: HashMap::new(),
                        id: -1,
                    });
                    self.nodes[cur].children.insert(b, n);
                    n
                }
            };
        }
        self.nodes[cur].id = id;
    }

    /// Push `(id, prefix_len)` for every piece that is a prefix of `input`,
    /// in increasing length.
    fn prefixes(&self, input: &[u8], out: &mut Vec<(i32, usize)>) {
        out.clear();
        let mut cur = 0usize;
        for (i, &b) in input.iter().enumerate() {
            match self.nodes[cur].children.get(&b) {
                Some(&n) => {
                    cur = n;
                    if self.nodes[n].id >= 0 {
                        out.push((self.nodes[n].id, i + 1));
                    }
                }
                None => break,
            }
        }
    }
}

pub struct Unigram {
    trie: PieceTrie,
}

impl Unigram {
    pub fn from_vocab(vocab: &Vocab) -> Unigram {
        let mut trie = PieceTrie::new();
        for (bytes, id) in vocab.segmentable_pieces() {
            trie.insert(bytes, id);
        }
        Unigram { trie }
    }

    pub fn encode(&self, norm: &[u8], vocab: &Vocab) -> Vec<Span> {
        if norm.is_empty() {
            return Vec::new();
        }
        let size = norm.len();
        let unk_score = vocab.min_score() - UNK_PENALTY;

        #[derive(Clone)]
        struct BestPathNode {
            id: i32,
            score: f32,
            starts_at: i64, // -1 = unfilled / start sentinel
        }
        let mut bp = vec![
            BestPathNode {
                id: -1,
                score: 0.0,
                starts_at: -1,
            };
            size + 1
        ];

        let mut matches: Vec<(i32, usize)> = Vec::new();
        let mut starts_at = 0usize;
        while starts_at < size {
            let best_till = bp[starts_at].score;
            let mblen = utf8_len(norm[starts_at]).min(size - starts_at);
            let mut has_single = false;

            self.trie.prefixes(&norm[starts_at..], &mut matches);
            for &(id, length) in &matches {
                let score = if matches!(vocab.kind(id), Some(PieceType::UserDefined)) {
                    // GetUserDefinedScore: 0.1 * (length - 1), computed in double.
                    (0.1_f64 * (length as f64 - 1.0)) as f32
                } else {
                    vocab.score(id)
                };
                let cand = score + best_till;
                let t = &mut bp[starts_at + length];
                // Tie-break: `>=`, not the upstream source's strict `>`. On an
                // exact-score tie (e.g. "-----" = {-,--,--} vs {--,--,-}) the
                // installed `sentencepiece` oracle keeps the *later* path; the
                // two differ only on such ties. Verified across both models.
                if t.starts_at == -1 || cand >= t.score {
                    t.score = cand;
                    t.starts_at = starts_at as i64;
                    t.id = id;
                }
                if !has_single && length == mblen {
                    has_single = true;
                }
            }

            if !has_single {
                let cand = unk_score + best_till;
                let t = &mut bp[starts_at + mblen];
                if t.starts_at == -1 || cand > t.score {
                    t.score = cand;
                    t.starts_at = starts_at as i64;
                    t.id = vocab.unk_id;
                }
            }

            starts_at += mblen;
        }

        // Backtrack.
        let mut out = Vec::new();
        let mut ends_at = size as i64;
        while ends_at > 0 {
            let node = &bp[ends_at as usize];
            let s = node.starts_at as usize;
            out.push(Span {
                start: s,
                len: ends_at as usize - s,
                id: node.id,
            });
            ends_at = node.starts_at;
        }
        out.reverse();
        out
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
