//! Utility modules for the Pipecat framework.
//!
//! This module provides foundational utilities including unique identification,
//! per-type instance counting, and the [`BaseObject`] type that supplies event
//! handling and lifecycle management to all Pipecat components.

pub mod base_object;

pub use base_object::{BaseObject, EventHandler, EventHandlerEntry};
