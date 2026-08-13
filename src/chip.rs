//! Apple silicon generation detection.
//!
//! Some acceleration choices are only correct on particular generations, so the
//! server identifies the host chip at startup rather than assuming.
//!
//! The distinction that matters here is the Neural Engine. Every Apple silicon
//! Mac has one, but reaching it means a Core ML port — MLX has no ANE backend on
//! any generation. Through M4 that port is not worth making for ASR: the ANE is
//! not fast enough at transformer decode to beat the GPU path, and running it
//! alongside buys little. M5 changes the arithmetic because its GPU gained
//! Neural Accelerators, so GPU and ANE became genuinely separate units worth
//! running concurrently.
//!
//! So ANE use is opt-in on M5 and above, and off elsewhere.

use std::process::Command;

/// Apple silicon generation, as reported by `machdep.cpu.brand_string`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleChip {
    M1,
    M2,
    M3,
    M4,
    M5,
    /// Newer than we know about — treated as at least M5-capable.
    Newer(u8),
    /// Intel, or a string we could not parse.
    Unknown,
}

impl AppleChip {
    /// Read the CPU brand string and classify it.
    pub fn detect() -> Self {
        let brand = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();
        Self::from_brand(&brand)
    }

    /// Classify a brand string such as `"Apple M5 Max"`.
    pub fn from_brand(brand: &str) -> Self {
        let brand = brand.trim();
        let Some(rest) = brand.strip_prefix("Apple M") else {
            return AppleChip::Unknown;
        };
        // Take the leading digits: "5 Max" -> 5, "3" -> 3.
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        match digits.parse::<u8>() {
            Ok(1) => AppleChip::M1,
            Ok(2) => AppleChip::M2,
            Ok(3) => AppleChip::M3,
            Ok(4) => AppleChip::M4,
            Ok(5) => AppleChip::M5,
            Ok(n) if n > 5 => AppleChip::Newer(n),
            _ => AppleChip::Unknown,
        }
    }

    /// Generation number, or 0 when unknown.
    pub fn generation(&self) -> u8 {
        match self {
            AppleChip::M1 => 1,
            AppleChip::M2 => 2,
            AppleChip::M3 => 3,
            AppleChip::M4 => 4,
            AppleChip::M5 => 5,
            AppleChip::Newer(n) => *n,
            AppleChip::Unknown => 0,
        }
    }

    /// Whether the GPU cores carry Neural Accelerators (matrix units reached
    /// through Metal 4 tensor ops). Introduced with M5.
    pub fn has_gpu_neural_accelerators(&self) -> bool {
        self.generation() >= 5
    }

    /// Whether offloading ASR encoder work to the Neural Engine is worth doing.
    ///
    /// M5 and later only. On earlier generations the ANE is not a useful second
    /// unit for this workload, and a Core ML port would cost more than it
    /// returns — so the server does not attempt it.
    pub fn ane_useful_for_asr(&self) -> bool {
        self.generation() >= 5
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AppleChip::M1 => "M1",
            AppleChip::M2 => "M2",
            AppleChip::M3 => "M3",
            AppleChip::M4 => "M4",
            AppleChip::M5 => "M5",
            AppleChip::Newer(_) => "M6+",
            AppleChip::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_brand_strings() {
        assert_eq!(AppleChip::from_brand("Apple M3"), AppleChip::M3);
        assert_eq!(AppleChip::from_brand("Apple M3 Pro"), AppleChip::M3);
        assert_eq!(AppleChip::from_brand("Apple M4 Max"), AppleChip::M4);
        assert_eq!(AppleChip::from_brand("Apple M5 Max\n"), AppleChip::M5);
        assert_eq!(AppleChip::from_brand("Apple M5 Ultra"), AppleChip::M5);
        assert_eq!(AppleChip::from_brand("Apple M12"), AppleChip::Newer(12));
        assert_eq!(AppleChip::from_brand("Intel Core i9"), AppleChip::Unknown);
        assert_eq!(AppleChip::from_brand(""), AppleChip::Unknown);
    }

    #[test]
    fn ane_is_m5_and_later_only() {
        for chip in [AppleChip::M1, AppleChip::M2, AppleChip::M3, AppleChip::M4] {
            assert!(!chip.ane_useful_for_asr(), "{:?} must not use the ANE", chip);
            assert!(!chip.has_gpu_neural_accelerators());
        }
        assert!(AppleChip::M5.ane_useful_for_asr());
        assert!(AppleChip::Newer(6).ane_useful_for_asr());
        // An unrecognised host stays conservative.
        assert!(!AppleChip::Unknown.ane_useful_for_asr());
    }
}
