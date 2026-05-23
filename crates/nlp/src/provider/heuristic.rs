//! Rule-based AiProvider. Always available; no models, no network, no LLM.
//!
//! Produces `entities` via the existing noun-phrase extractor and best-effort
//! `intent` + `operation` via regex pattern matching. Conservatively low
//! `confidence` so semantic-supersede consumers know to back off when this
//! provider is the source.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;

use super::{AiProvider, Capabilities, ExtractContext, Extraction, Intent, Operation, ProviderError};
use crate::extract_entities;

pub struct HeuristicProvider;

#[async_trait]
impl AiProvider for HeuristicProvider {
    fn name(&self) -> &'static str {
        "heuristic"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            extraction: true,
            summarization: false,
            fact_distillation: false,
            typical_latency_ms: 1,
            context_window: 1_000_000,
        }
    }

    async fn extract(
        &self,
        text: &str,
        _ctx: &ExtractContext,
    ) -> Result<Extraction, ProviderError> {
        let entities = extract_entities(text);

        // Best-effort intent classification via patterns.
        let intent = classify_intent(text);
        let operation = classify_operation(text);

        // For topic, take the longest multi-word entity (usually the most
        // semantically loaded). It's a heuristic; not as good as an LLM picking.
        let topic = entities
            .iter()
            .filter(|e| e.contains(' '))
            .max_by_key(|e| e.split_whitespace().count())
            .cloned();

        let action_target = match operation {
            Some(_) => topic.clone(),
            None => None,
        };

        Ok(Extraction {
            topic,
            intent,
            operation,
            entities,
            action_target,
            contradicts_hint: None,
            confidence: 0.5,
        })
    }
}

fn classify_intent(text: &str) -> Intent {
    let lower = text.to_lowercase();
    if QUESTION_RE.is_match(&lower) {
        Intent::Question
    } else if TASK_RE.is_match(&lower) {
        Intent::Task
    } else if DECISION_RE.is_match(&lower) {
        Intent::Decision
    } else if OPINION_RE.is_match(&lower) {
        Intent::Opinion
    } else if UPDATE_RE.is_match(&lower) {
        Intent::Update
    } else {
        Intent::Unknown
    }
}

fn classify_operation(text: &str) -> Option<Operation> {
    let lower = text.to_lowercase();
    if CONTRADICT_RE.is_match(&lower) {
        Some(Operation::Contradict)
    } else if REPLACE_RE.is_match(&lower) {
        Some(Operation::Replace)
    } else if REMOVE_RE.is_match(&lower) {
        Some(Operation::Remove)
    } else if ADD_RE.is_match(&lower) {
        Some(Operation::Add)
    } else {
        None
    }
}

// Intent patterns. Conservative — when nothing matches we return Unknown
// rather than guess.
static QUESTION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\?|^what|^how|^why|^when|^where|^who|^which|^can\b|^should\b|^could\b|^is\b|^are\b").unwrap());
static TASK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bi need to\b|\bi should\b|\bwe need to\b|\btodo:|\btodo\b|\bdon'?t forget\b|\bnext step\b|\bfollow up\b").unwrap());
static DECISION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\blet'?s\b|\bwe'?ll\b|\bdecided\b|\bgoing with\b|\bgo with\b|\bchose\b").unwrap());
static OPINION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bi think\b|\bi feel\b|\bimo\b|\bin my opinion\b|\bi prefer\b|\bi like\b|\bi don'?t like\b").unwrap());
static UPDATE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bactually\b|\bnow\b|\bswitched\b|\bchanged\b|\bupdated\b|\bmoved to\b|\bis now\b|\bare now\b").unwrap());

// Operation patterns.
static ADD_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\badd(?:ed|ing)?\b|\bappend(?:ed|ing)?\b|\binclude(?:d|ing)?\b").unwrap());
static REMOVE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bremove(?:d|ing)?\b|\bdelete(?:d|ing)?\b|\bdrop(?:ped|ping)?\b|\bcross(?:ed|ing)? off\b").unwrap());
static REPLACE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bswitch(?:ed|ing)?\b|\bchang(?:ed|ing)?\b|\breplaced?\b|\bmoved? to\b|\bis now\b|\bare now\b").unwrap());
static CONTRADICT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\bactually\b|\bcorrection\b|\bwrong\b|\bnot true\b|\bnope\b").unwrap());

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn smoke_test_supersede_turns() {
        let p = HeuristicProvider;
        let ctx = ExtractContext::default();

        let t1 = p.extract(
            "We are using Postgres with pgvector. The deploy target is staging.",
            &ctx,
        )
        .await
        .unwrap();
        let t2 = p.extract(
            "Actually we switched the deploy target to production today.",
            &ctx,
        )
        .await
        .unwrap();

        // Both should agree on the topic.
        assert_eq!(t1.topic.as_deref(), Some("deploy target"), "t1: {:?}", t1);
        assert_eq!(t2.topic.as_deref(), Some("deploy target"), "t2: {:?}", t2);

        // Turn 2 should be flagged as a replacement.
        assert!(matches!(t2.operation, Some(Operation::Replace) | Some(Operation::Contradict)),
                "t2 op: {:?}", t2.operation);
    }

    #[tokio::test]
    async fn classifies_questions() {
        let p = HeuristicProvider;
        let ctx = ExtractContext::default();
        let r = p.extract("What deploy target are we using?", &ctx).await.unwrap();
        assert_eq!(r.intent, Intent::Question, "got {:?}", r);
    }

    #[tokio::test]
    async fn classifies_tasks() {
        let p = HeuristicProvider;
        let ctx = ExtractContext::default();
        let r = p.extract("I need to update the rate limiter config.", &ctx).await.unwrap();
        assert_eq!(r.intent, Intent::Task, "got {:?}", r);
    }
}
