// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Frame processing pipeline infrastructure for Pipecat.
//!
//! This module provides the core processing system that enables building
//! audio/video processing pipelines using channel-based task isolation.

pub mod aggregators;
pub mod audio;
// pub mod filters; // Legacy FrameProcessor — commented out until migrated
pub mod metrics;
pub mod processor;
pub use processor::{Processor, ProcessorContext, ProcessorWeight};

/// Implement `Debug` and `Display` for a type that contains a `base: BaseProcessor` field.
///
/// The `Debug` impl prints `TypeName(name)` and the `Display` impl prints just the
/// processor name obtained from `self.base.name()`.
///
/// # Examples
///
/// ```ignore
/// impl_base_debug_display!(MyProcessor);
/// ```
#[macro_export]
macro_rules! impl_base_debug_display {
    ($struct_name:ident) => {
        impl std::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($struct_name), self.base.name())
            }
        }

        impl std::fmt::Display for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.base.name())
            }
        }
    };
}

/// Implement `Debug` and `Display` for a processor struct with `id: u64` and `name: String` fields.
///
/// Unlike [`impl_base_debug_display!`], which requires a `base: BaseProcessor` field, this macro
/// works with any struct that has `id` and `name` fields directly. The `Debug` output uses
/// `debug_struct` with `..` non-exhaustive notation; the `Display` output prints just the name.
///
/// # Examples
///
/// ```ignore
/// struct MyProcessor {
///     id: u64,
///     name: String,
///     // ... other fields
/// }
///
/// impl_processor!(MyProcessor);
/// ```
///
/// **Note:** This macro only supports non-generic structs. For generic
/// structs, implement `Debug` and `Display` manually.
#[macro_export]
macro_rules! impl_processor {
    ($ty:ident) => {
        impl ::std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.debug_struct(stringify!($ty))
                    .field("id", &self.id)
                    .field("name", &self.name)
                    .finish_non_exhaustive()
            }
        }

        impl ::std::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, "{}", self.name)
            }
        }
    };
}

/// Implement only `Display` for a type that contains a `base: BaseProcessor` field.
///
/// Use this when the type needs a custom `Debug` implementation (e.g. to show
/// extra fields) but the standard `Display` that prints `self.base.name()`.
///
/// # Examples
///
/// ```ignore
/// impl_base_display!(CartesiaTTSService);
/// ```
#[macro_export]
macro_rules! impl_base_display {
    ($struct_name:ident) => {
        impl std::fmt::Display for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.base.name())
            }
        }
    };
}

/// Direction of frame flow in the processing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDirection {
    /// Frames flowing from input to output.
    Downstream,
    /// Frames flowing back from output to input.
    Upstream,
}

#[cfg(test)]
mod macro_tests {
    /// Minimal struct with `id` and `name` fields for testing `impl_processor!`.
    struct TestProc {
        id: u64,
        name: String,
    }

    impl_processor!(TestProc);

    #[test]
    fn debug_contains_struct_name_and_id() {
        let p = TestProc {
            id: 42,
            name: "my-proc".to_string(),
        };
        let debug = format!("{:?}", p);
        assert!(
            debug.contains("TestProc"),
            "Debug output should contain struct name, got: {debug}"
        );
        assert!(
            debug.contains("42"),
            "Debug output should contain the id, got: {debug}"
        );
    }

    #[test]
    fn display_equals_name() {
        let p = TestProc {
            id: 1,
            name: "hello-world".to_string(),
        };
        assert_eq!(format!("{}", p), "hello-world");
    }
}
