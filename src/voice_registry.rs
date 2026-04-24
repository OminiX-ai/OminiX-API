//! Voice registry for validating TTS voice names.
//!
//! Collects the set of voices the server will actually synthesize for:
//! the built-in Qwen3-TTS preset speakers plus any custom cloned voices
//! declared in `~/.OminiX/models/voices.json`. Used to reject
//! `POST /v1/audio/tts/qwen3` requests whose `voice` field does not match
//! a registered voice, instead of silently falling back to a default.
//!
//! An empty/`"default"` voice is considered valid at the handler level
//! (the handler substitutes the documented default via `normalize_voice`
//! before looking up here).

use std::collections::HashSet;

/// Qwen3-TTS CustomVoice preset speaker names (built into the model).
///
/// Kept in lockstep with `PRESET_SPEAKERS` in `handlers::training`; any
/// preset added to the model must be added to both. A future refactor
/// can unify the two copies without changing behavior.
const PRESET_SPEAKERS: &[&str] = &[
    "vivian",
    "serena",
    "ryan",
    "aiden",
    "uncle_fu",
    "chinese_woman",
    "chinese_man",
    "dialect",
    "english_man",
];

/// Default path of the custom-voice registry file.
const DEFAULT_VOICES_JSON: &str = "~/.OminiX/models/voices.json";

/// Environment variable that overrides `DEFAULT_VOICES_JSON`. Primarily
/// for tests and operators who store the voices file elsewhere.
const VOICES_JSON_ENV: &str = "OMINIX_VOICES_JSON";

/// Registry of voices the TTS backend will accept.
#[derive(Debug, Clone)]
pub struct VoiceRegistry {
    /// Canonical voice names — preset speakers plus custom names declared in
    /// `voices.json`. Ordering: presets first (registry order), then customs
    /// in JSON-iteration order.
    canonical_names: Vec<String>,
    /// Lookup keys: every canonical name plus every alias. Case-sensitive
    /// to match the existing model behavior (Qwen3-TTS preset names are
    /// lowercase by convention).
    valid_keys: HashSet<String>,
}

impl VoiceRegistry {
    /// Load the registry from the default location
    /// (`~/.OminiX/models/voices.json`, or the path in the
    /// `OMINIX_VOICES_JSON` env var when set). Missing or malformed JSON
    /// is tolerated — the registry still includes the preset speakers.
    pub fn load() -> Self {
        let path = std::env::var(VOICES_JSON_ENV)
            .unwrap_or_else(|_| crate::utils::expand_tilde(DEFAULT_VOICES_JSON));
        Self::load_from_path(&path)
    }

    /// Load the registry from an explicit path. Exposed for tests.
    pub fn load_from_path(path: &str) -> Self {
        let custom_json = std::fs::read_to_string(path)
            .ok()
            .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok());
        Self::from_json(custom_json.as_ref())
    }

    /// Build the registry from optional custom-voice JSON (the raw
    /// contents of `voices.json`).
    fn from_json(custom: Option<&serde_json::Value>) -> Self {
        let mut canonical_names: Vec<String> =
            PRESET_SPEAKERS.iter().map(|s| s.to_string()).collect();
        let mut valid_keys: HashSet<String> = canonical_names.iter().cloned().collect();

        if let Some(config) = custom {
            if let Some(voices) = config.get("voices").and_then(|v| v.as_object()) {
                for (name, voice) in voices {
                    canonical_names.push(name.clone());
                    valid_keys.insert(name.clone());
                    if let Some(aliases) = voice.get("aliases").and_then(|a| a.as_array()) {
                        for alias in aliases.iter().filter_map(|v| v.as_str()) {
                            valid_keys.insert(alias.to_string());
                        }
                    }
                }
            }
        }

        Self {
            canonical_names,
            valid_keys,
        }
    }

    /// Return `true` when `voice` matches a registered canonical name or
    /// alias. Case-sensitive.
    pub fn contains(&self, voice: &str) -> bool {
        self.valid_keys.contains(voice)
    }

    /// Canonical list of voices for client-facing error messages.
    pub fn available_voices(&self) -> &[String] {
        &self.canonical_names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with_custom(json: serde_json::Value) -> VoiceRegistry {
        VoiceRegistry::from_json(Some(&json))
    }

    #[test]
    fn should_accept_preset_speakers_when_no_custom_voices() {
        let reg = VoiceRegistry::from_json(None);
        assert!(reg.contains("vivian"));
        assert!(reg.contains("serena"));
        assert!(reg.contains("english_man"));
    }

    #[test]
    fn should_reject_unknown_voice_when_not_registered() {
        let reg = VoiceRegistry::from_json(None);
        assert!(!reg.contains("yangmi"));
        assert!(!reg.contains("does_not_exist"));
    }

    #[test]
    fn should_accept_custom_voice_when_declared_in_json() {
        let reg = registry_with_custom(serde_json::json!({
            "voices": {
                "yangmi": {"aliases": []}
            }
        }));
        assert!(reg.contains("yangmi"));
    }

    #[test]
    fn should_accept_alias_when_declared_in_json() {
        let reg = registry_with_custom(serde_json::json!({
            "voices": {
                "vivian_clone": {"aliases": ["vivian2", "vivian-new"]}
            }
        }));
        assert!(reg.contains("vivian_clone"));
        assert!(reg.contains("vivian2"));
        assert!(reg.contains("vivian-new"));
    }

    #[test]
    fn should_be_case_sensitive_when_matching_voice() {
        let reg = VoiceRegistry::from_json(None);
        assert!(reg.contains("vivian"));
        assert!(!reg.contains("Vivian"));
        assert!(!reg.contains("VIVIAN"));
    }

    #[test]
    fn should_list_presets_first_in_available_voices() {
        let reg = registry_with_custom(serde_json::json!({
            "voices": {"custom_a": {"aliases": []}}
        }));
        let names = reg.available_voices();
        assert_eq!(names[0], "vivian");
        assert!(names.iter().any(|n| n == "custom_a"));
        // Customs appear after all presets.
        let custom_idx = names.iter().position(|n| n == "custom_a").unwrap();
        assert!(custom_idx >= PRESET_SPEAKERS.len());
    }

    #[test]
    fn should_tolerate_malformed_voices_json_when_loading() {
        // Point at a file that does not exist — should still list presets.
        let reg = VoiceRegistry::load_from_path("/nonexistent/voices.json");
        assert!(reg.contains("vivian"));
        assert!(!reg.contains("yangmi"));
    }

    #[test]
    fn should_tolerate_voices_without_aliases_field() {
        let reg = registry_with_custom(serde_json::json!({
            "voices": {
                "no_alias_voice": {}
            }
        }));
        assert!(reg.contains("no_alias_voice"));
    }
}
