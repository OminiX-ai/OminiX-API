//! OpenAI-compatible request/response types
//!
//! Some fields are deserialized for API compatibility but not yet used by
//! the inference backends (e.g., `top_p`, `response_format`, `speed`).

mod audio;
mod chat;
mod download;
mod error;
mod image;
mod training;
mod vlm;
mod voice;

pub use audio::*;
pub use chat::*;
pub use download::*;
pub use error::*;
pub use image::*;
pub use training::*;
pub use vlm::*;
pub use voice::*;
