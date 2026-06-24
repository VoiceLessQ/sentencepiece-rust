//! Darts-clone double-array trie — read-only `common_prefix_search` / `traverse`.
//!
//! Port reference: `Reference/sentencepiece/third_party/darts_clone/darts.h`
//! (we load a prebuilt array, so only the search side is ported, not `build`).
//!
//! Used by the charsmap normaliser (`common_prefix_search`, v0.2) and by Unigram
//! segmentation (`traverse`, v0.3).
//!
//! Unit bit-layout (`DoubleArrayUnit` in darts.h):
//!   * `has_leaf` = bit 8
//!   * `value`    = low 31 bits (leaf units only)
//!   * `label`    = bit 31 | low 8 bits
//!   * `offset`   = (unit >> 10) << ((unit & (1<<9)) >> 6)

#[inline]
fn has_leaf(u: u32) -> bool {
    (u >> 8) & 1 == 1
}
#[inline]
fn value(u: u32) -> i32 {
    (u & 0x7fff_ffff) as i32
}
#[inline]
fn label(u: u32) -> u32 {
    u & (0x8000_0000 | 0xff)
}
#[inline]
fn offset(u: u32) -> u32 {
    (u >> 10) << ((u & (1 << 9)) >> 6)
}

/// A read-only Darts double-array, owning its units.
pub struct DoubleArray {
    units: Vec<u32>,
}

impl DoubleArray {
    /// Build from the raw little-endian trie blob (length must be a multiple of 4).
    pub fn from_trie_bytes(bytes: &[u8]) -> DoubleArray {
        let n = bytes.len() / 4;
        let mut units = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * 4;
            units.push(u32::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
            ]));
        }
        DoubleArray { units }
    }

    #[inline]
    fn unit(&self, i: usize) -> Option<u32> {
        self.units.get(i).copied()
    }

    /// All `(value, prefix_len)` pairs where a key prefix matches `key`.
    /// Ports `commonPrefixSearch` (explicit-length branch). Bounds-checked so a
    /// malformed model returns no matches rather than panicking.
    pub fn common_prefix_search(&self, key: &[u8]) -> Vec<(i32, usize)> {
        let mut results = Vec::new();
        let mut node_pos = 0usize;
        let mut unit = match self.unit(node_pos) {
            Some(u) => u,
            None => return results,
        };
        node_pos ^= offset(unit) as usize;
        for (i, &k) in key.iter().enumerate() {
            node_pos ^= k as usize;
            unit = match self.unit(node_pos) {
                Some(u) => u,
                None => return results,
            };
            if label(unit) != k as u32 {
                return results;
            }
            node_pos ^= offset(unit) as usize;
            if has_leaf(unit) {
                if let Some(leaf) = self.unit(node_pos) {
                    results.push((value(leaf), i + 1));
                }
            }
        }
        results
    }

    /// Incrementally walk `key[key_pos..length]` from `node_pos`. Returns the
    /// value at the reached node, `-1` (no value), or `-2` (no such transition).
    /// Ports `traverse`; used by Unigram (v0.3).
    #[allow(dead_code)]
    pub fn traverse(
        &self,
        key: &[u8],
        node_pos: &mut usize,
        key_pos: &mut usize,
        length: usize,
    ) -> i32 {
        let mut id = *node_pos;
        let mut unit = match self.unit(id) {
            Some(u) => u,
            None => return -2,
        };
        while *key_pos < length {
            let k = key[*key_pos] as usize;
            id ^= offset(unit) as usize ^ k;
            unit = match self.unit(id) {
                Some(u) => u,
                None => return -2,
            };
            if label(unit) != k as u32 {
                return -2;
            }
            *node_pos = id;
            *key_pos += 1;
        }
        if !has_leaf(unit) {
            return -1;
        }
        match self.unit(id ^ offset(unit) as usize) {
            Some(u) => value(u),
            None => -1,
        }
    }
}
