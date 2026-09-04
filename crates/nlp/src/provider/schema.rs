//! The shape every backend produces, regardless of how it got there.
//!
//! All three providers — heuristic, local LLM, remote LLM — return the same
//! `Extraction` struct. The original supersede heuristic only consumed
//! `entities`; the semantic path reads `topic` + `intent` + `operation` to do
//! semantic supersede instead of string-Jaccard.

use serde::{Deserialize, Deserializer, Serialize};

/// Read an enum-valued field without letting an unrecognised value discard the
/// rest of the extraction. Models return synonyms — "note" for update, "modify"
/// for replace — and a strict parse threw away the topic and the entities along
/// with the word it could not place. `Intent::Unknown` exists for exactly this
/// case; an operation it cannot name is simply absent.
fn lenient<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let raw = Option::<serde_json::Value>::deserialize(d)?;
    Ok(raw.and_then(|v| serde_json::from_value(v).ok()))
}

fn lenient_intent<'de, D: Deserializer<'de>>(d: D) -> Result<Intent, D::Error> {
    Ok(lenient::<D, Intent>(d)?.unwrap_or_default())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    /// User is asking about state, no update implied.
    Question,
    /// User is changing or asserting a fact / state.
    Update,
    /// User is choosing among alternatives or settling a debate.
    Decision,
    /// User is creating a todo / something to follow up on.
    Task,
    /// User is offering an opinion / preference.
    Opinion,
    /// User is referring to an existing thing without changing it.
    Reference,
    /// Default / unknown — heuristic backend returns this when it can't classify.
    Unknown,
}

impl Default for Intent {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// Adds something new.
    Add,
    /// Removes / deletes / drops something.
    Remove,
    /// Replaces an existing value with a new one (the classic supersede case).
    Replace,
    /// Re-states the existing value (no semantic change; useful for confidence).
    Reaffirm,
    /// Asserts something that contradicts a known prior claim.
    Contradict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extraction {
    /// Canonical topic phrase for this memory. e.g. "deploy target",
    /// "auth middleware". Used for semantic supersede.
    #[serde(default)]
    pub topic: Option<String>,

    #[serde(default, deserialize_with = "lenient_intent")]
    pub intent: Intent,

    #[serde(default, deserialize_with = "lenient")]
    pub operation: Option<Operation>,

    /// The cognitive register this turn belongs to (e.g. "code", "general",
    /// "journal", "research"). Auto-classified by an LLM provider when the
    /// caller didn't pin a mode; `None` when unclear (the project default then
    /// wins). The heuristic provider always returns `None`.
    #[serde(default)]
    pub mode: Option<String>,

    /// Multi-word noun phrases. Always populated, even by the heuristic.
    /// Kept for back-compat with the original entity-Jaccard fingerprint.
    #[serde(default)]
    pub entities: Vec<String>,

    /// What `operation` acts on, if any. e.g. for `operation=replace,
    /// action_target="deploy target"`, the deploy target is what's being
    /// replaced.
    #[serde(default)]
    pub action_target: Option<String>,

    /// Free-text claim being contradicted, if `operation=contradict`.
    /// Surfaced later via a /conflicts endpoint.
    #[serde(default)]
    pub contradicts_hint: Option<String>,

    /// Provider's self-reported confidence in this extraction, 0.0–1.0.
    /// Heuristic always returns 0.5. LLMs are asked to self-rate.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    0.5
}

impl Extraction {
    /// Empty extraction with default values — used when extraction is skipped
    /// or fails.
    pub fn empty() -> Self {
        Self {
            topic: None,
            intent: Intent::Unknown,
            operation: None,
            mode: None,
            entities: Vec::new(),
            action_target: None,
            contradicts_hint: None,
            confidence: 0.0,
        }
    }
}

/// An episodic memory referenced by the consolidation worker. Lightweight —
/// the worker pulls these in batches and feeds them to `distill_facts()`.
#[derive(Debug, Clone, Serialize)]
pub struct EpisodeRef {
    pub id: uuid::Uuid,
    pub content: String,
    pub topic: Option<String>,
    pub entities: Vec<String>,
    /// When the episode happened — an RFC 3339 instant, or `start..end` for a
    /// span. Serialized into the distill prompt so the model can weigh recency
    /// when episodes disagree; without it, "which claim is newer" is unknowable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
}

/// A semantic fact distilled from one or more episodic memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledFact {
    /// The distilled claim, written in the third person, present tense.
    pub content: String,
    /// Optional canonical topic phrase shared by the source episodes.
    #[serde(default)]
    pub topic: Option<String>,
    /// IDs of the episodic memories this fact was derived from.
    #[serde(default)]
    pub source_episode_ids: Vec<uuid::Uuid>,
    /// Provider's self-reported confidence.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// Wrapper used as a JSON-schema target so the LLM returns a typed array.
#[derive(Debug, Clone, Deserialize)]
pub struct DistillResponse {
    pub facts: Vec<DistilledFact>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_the_enum_does_not_have_costs_only_that_word() {
        let e: Extraction = serde_json::from_str(
            r#"{"topic":"backups","intent":"note","operation":"modify","entities":["backup drive"]}"#,
        )
        .expect("one unrecognised word must not discard the extraction");
        assert_eq!(e.topic.as_deref(), Some("backups"));
        assert_eq!(e.entities, vec!["backup drive".to_string()]);
        assert_eq!(e.intent, Intent::Unknown);
        assert!(e.operation.is_none());
    }

    #[test]
    fn a_word_the_enum_does_have_is_still_read() {
        let e: Extraction =
            serde_json::from_str(r#"{"intent":"decision","operation":"replace"}"#).unwrap();
        assert_eq!(e.intent, Intent::Decision);
        assert!(matches!(e.operation, Some(Operation::Replace)));
    }

    #[test]
    fn a_field_of_the_wrong_shape_entirely_is_also_survivable() {
        let e: Extraction =
            serde_json::from_str(r#"{"topic":"t","intent":7,"operation":["a"]}"#).unwrap();
        assert_eq!(e.topic.as_deref(), Some("t"));
        assert_eq!(e.intent, Intent::Unknown);
        assert!(e.operation.is_none());
    }

    #[test]
    fn an_absent_field_is_the_default() {
        let e: Extraction = serde_json::from_str("{}").unwrap();
        assert_eq!(e.intent, Intent::Unknown);
        assert!(e.operation.is_none());
        assert!(e.topic.is_none());
        assert!(e.entities.is_empty());
    }
}
