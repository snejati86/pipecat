// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! DTMF (Dual-Tone Multi-Frequency) detection and generation.

/// DTMF keypad entries for phone system integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeypadEntry {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Pound,
    Star,
}

impl std::fmt::Display for KeypadEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeypadEntry::Zero => write!(f, "0"),
            KeypadEntry::One => write!(f, "1"),
            KeypadEntry::Two => write!(f, "2"),
            KeypadEntry::Three => write!(f, "3"),
            KeypadEntry::Four => write!(f, "4"),
            KeypadEntry::Five => write!(f, "5"),
            KeypadEntry::Six => write!(f, "6"),
            KeypadEntry::Seven => write!(f, "7"),
            KeypadEntry::Eight => write!(f, "8"),
            KeypadEntry::Nine => write!(f, "9"),
            KeypadEntry::Pound => write!(f, "#"),
            KeypadEntry::Star => write!(f, "*"),
        }
    }
}
