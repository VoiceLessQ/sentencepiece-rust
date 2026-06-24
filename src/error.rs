//! Error type for the crate.

use std::fmt;

/// Errors produced while loading a model or encoding/decoding text.
#[derive(Debug)]
pub enum Error {
    /// The model file/bytes could not be parsed as a `ModelProto`.
    Proto(String),
    /// The model parsed but is missing data required for inference.
    Model(String),
    /// A feature is recognised but not yet implemented in this port.
    Unsupported(&'static str),
    /// Underlying I/O error while reading a model file.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Proto(m) => write!(f, "proto parse error: {m}"),
            Error::Model(m) => write!(f, "invalid model: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
