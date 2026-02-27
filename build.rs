// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Build script that compiles `.proto` files into Rust code via `prost-build`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(&["proto/frames.proto"], &["proto/"])?;
    Ok(())
}
