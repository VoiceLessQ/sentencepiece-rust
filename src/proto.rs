//! Minimal proto2 wire-format reader.
//!
//! SentencePiece `.model` files are serialised `ModelProto` messages
//! (`Reference/sentencepiece/src/sentencepiece_model.proto`). We only need a
//! handful of fields, so instead of depending on `prost` + `protoc` we walk the
//! wire format directly. The wire format is tiny and stable:
//!
//!   record   := key value
//!   key      := varint  (field_number << 3 | wire_type)
//!   wire_type 0 = varint, 1 = 64-bit, 2 = length-delimited, 5 = 32-bit
//!
//! See <https://protobuf.dev/programming-guides/encoding/>.

use crate::error::{Error, Result};

/// A single decoded field value.
#[derive(Debug)]
#[allow(dead_code)] // Fixed64 completes the wire format; no field we read uses it.
pub enum Value<'a> {
    Varint(u64),
    Fixed32([u8; 4]),
    Fixed64([u8; 8]),
    Len(&'a [u8]),
}

impl<'a> Value<'a> {
    /// Interpret a varint field as a bool (proto2 `optional bool`).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Varint(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Interpret a varint field as an `int32` (two's-complement, may be negative).
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Value::Varint(v) => Some(*v as i64 as i32),
            _ => None,
        }
    }

    /// Interpret a `fixed32` field as an IEEE-754 `float`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::Fixed32(b) => Some(f32::from_le_bytes(*b)),
            _ => None,
        }
    }

    /// Borrow a length-delimited field as raw bytes.
    pub fn as_bytes(&self) -> Option<&'a [u8]> {
        match self {
            Value::Len(b) => Some(b),
            _ => None,
        }
    }

    /// Decode a length-delimited field as a UTF-8 string (lossy is not used:
    /// SentencePiece pieces are valid UTF-8 or `<0xXX>` byte tokens).
    pub fn as_str(&self) -> Option<String> {
        self.as_bytes()
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }
}

/// A streaming reader over a serialised protobuf message.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.buf.len() {
                return Err(Error::Proto("truncated varint".into()));
            }
            if shift >= 64 {
                return Err(Error::Proto("varint overflow".into()));
            }
            let byte = self.buf[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.buf.len())
            .ok_or_else(|| Error::Proto("length-delimited field out of bounds".into()))?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Read the next `(field_number, value)` pair, or `Ok(None)` at end of message.
    pub fn next(&mut self) -> Result<Option<(u32, Value<'a>)>> {
        if self.is_empty() {
            return Ok(None);
        }
        let key = self.varint()?;
        let number = (key >> 3) as u32;
        let wire = key & 7;
        let value = match wire {
            0 => Value::Varint(self.varint()?),
            1 => {
                let b = self.take(8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                Value::Fixed64(a)
            }
            2 => {
                let len = self.varint()? as usize;
                Value::Len(self.take(len)?)
            }
            5 => {
                let b = self.take(4)?;
                let mut a = [0u8; 4];
                a.copy_from_slice(b);
                Value::Fixed32(a)
            }
            other => {
                return Err(Error::Proto(format!("unsupported wire type {other}")));
            }
        };
        Ok(Some((number, value)))
    }
}
