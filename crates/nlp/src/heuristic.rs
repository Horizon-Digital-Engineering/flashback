//! Rule-based entity extraction.
//!
//! The Phase 1 supersede heuristic depended on spaCy NER, which only catches
//! named entities (proper nouns, organizations, dates). It missed everyday
//! domain noun-phrases like "deploy target", "auth middleware", "the
//! migration" — exactly the things people refer back to across turns.
//!
//! This Rust replacement extracts:
//!
//! 1. **Capitalized standalone words** (likely proper nouns: `Postgres`,
//!    `Anthropic`, `Wisconsin`).
//! 2. **Multi-word non-stopword sequences** (likely domain noun phrases:
//!    `deploy target`, `auth middleware`, `pricing tier`).
//!
//! It's intentionally simple — no POS tagger, no ML model. The goal is good
//! *fingerprints* for entity-Jaccard supersede detection, not full NER. Phase
//! 2b layers an LLM-based provider on top of this for richer extraction.

use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;

/// Extract a deduplicated list of entity strings from free text.
///
/// All entities are lowercased so Jaccard comparisons are case-insensitive.
pub fn extract_entities(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let normalized = normalize(text);
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for cap in TOKEN_RE.captures_iter(&normalized) {
        if let Some(m) = cap.get(0) {
            let raw = m.as_str();
            if is_proper_noun_candidate(raw) {
                let lower = raw.to_lowercase();
                // Skip stopwords AND verbs — common verbs at sentence start
                // ("Got", "Noted", "Switched") look like proper nouns but
                // aren't entities.
                if lower.len() >= 2
                    && !STOPWORDS.contains(lower.as_str())
                    && !VERB_LIKE.contains(lower.as_str())
                    && seen.insert(lower.clone())
                {
                    out.push(lower);
                }
            }
        }
    }

    let phrases = extract_phrases(&normalized);
    for p in phrases {
        if seen.insert(p.clone()) {
            out.push(p);
        }
    }

    out
}

fn normalize(text: &str) -> String {
    // Collapse whitespace; strip soft-line-wraps; trim.
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_proper_noun_candidate(word: &str) -> bool {
    word.len() >= 2
        && word.chars().next().is_some_and(|c| c.is_uppercase())
        && word.chars().any(|c| c.is_lowercase())
}

/// Walk the tokenized stream, grouping runs of "content tokens" (alphanumeric,
/// not stopwords). Any run of length ≥ 2 is a candidate noun phrase.
fn extract_phrases(text: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut current: Vec<String> = Vec::new();

    for raw_token in text.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
        let token = raw_token.trim();
        if token.is_empty() {
            flush(&mut current, &mut phrases, &mut seen);
            continue;
        }
        let lower = token.to_lowercase();

        // Stopword OR pronoun OR verb-ish breaks the phrase.
        if STOPWORDS.contains(lower.as_str()) || VERB_LIKE.contains(lower.as_str()) {
            flush(&mut current, &mut phrases, &mut seen);
            continue;
        }
        // Numbers alone don't anchor a phrase.
        if lower.chars().all(|c| c.is_numeric()) {
            flush(&mut current, &mut phrases, &mut seen);
            continue;
        }
        current.push(lower);
    }
    flush(&mut current, &mut phrases, &mut seen);
    phrases
}

fn flush(current: &mut Vec<String>, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    if current.len() >= 2 {
        let phrase = current.join(" ");
        if !seen.contains(&phrase) {
            seen.insert(phrase.clone());
            out.push(phrase);
        }
    }
    current.clear();
}

static TOKEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\w[\w'-]*\b").unwrap());

/// English stopword set. Trimmed to function words + filler — kept small so
/// it doesn't eat real content (e.g. "production", "staging" are NOT stopwords).
static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "for", "to", "from",
        "in", "on", "at", "by", "with", "as", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "our", "their", "mine", "yours",
        "ours", "theirs", "do", "does", "did", "have", "has", "had", "having", "not", "no",
        "yes", "ok", "okay", "maybe", "should", "would", "could", "can", "may", "might", "must",
        "shall", "will", "going", "want", "wants", "wanted", "really", "actually", "just",
        "very", "so", "too", "also", "even", "still", "any", "some", "all", "more", "most",
        "few", "many", "much", "less", "lots", "lot", "such", "what", "when", "where", "why",
        "how", "who", "whom", "which", "whose", "than", "because", "though", "although", "while",
        "after", "before", "during", "until", "via", "about", "around", "into", "onto", "out",
        "over", "under", "between", "through", "up", "down", "off", "back", "again", "now",
        "today", "tomorrow", "yesterday", "soon", "later", "earlier", "first", "last", "next",
        "thanks", "please", "yeah", "yep", "nope", "well", "hi", "hello", "hey", "ok", "alright",
        // Conversation scaffolding labels — not real entities.
        "user", "assistant", "system",
    ]
    .into_iter()
    .collect()
});

/// Tokens that are unambiguously verbs (and so end a noun phrase). Domain
/// action words like "deploy", "build", "ship", "switch", "add", "remove" are
/// deliberately EXCLUDED from their base form — they can be modifiers in
/// compounds like "deploy target", "build script", "switch case". Only their
/// -ed and -ing inflections are listed (those forms are almost always verbal).
static VERB_LIKE: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // Abstract action verbs — all forms are verbal in conversation.
        "use", "uses", "used", "using",
        "make", "makes", "made", "making",
        "go", "goes", "went", "going", "gone",
        "say", "says", "said", "saying",
        "see", "sees", "saw", "seen", "seeing",
        "tell", "tells", "told", "telling",
        "think", "thinks", "thought", "thinking",
        "know", "knows", "knew", "known", "knowing",
        "take", "takes", "took", "taken", "taking",
        "give", "gives", "gave", "given", "giving",
        "find", "finds", "found", "finding",
        "try", "tries", "tried", "trying",
        "look", "looks", "looked", "looking",
        "feel", "feels", "felt", "feeling",
        "need", "needs", "needed", "needing",
        "want", "wants", "wanted",
        "come", "comes", "came", "coming",
        // Domain action words — only inflected forms are verbal. The base
        // form (deploy, build, switch, etc.) can be a noun/modifier in
        // compounds, so it's NOT in this list.
        "added", "adding",
        "removed", "removing",
        "dropped", "dropping",
        "deleted", "deleting",
        "deployed", "deploying",
        "built", "building",
        "shipped", "shipping",
        "switched", "switching",
        "ran", "running",
        "moved", "moving",
        "rolled", "rolling",
        "broke", "broken", "breaking",
        "fixed", "fixing",
        "configured", "configuring",
        "merged", "merging",
        "pushed", "pushing",
        "pulled", "pulling",
        // Get/note/etc — pure verbs even at base form.
        "get", "gets", "got", "gotten", "getting",
        "note", "notes", "noted", "noting",
    ]
    .into_iter()
    .collect()
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test_turn_pair() {
        let t1 = "We are using Postgres with pgvector. The deploy target is staging.";
        let t2 = "Actually we switched the deploy target to production today.";
        let e1 = extract_entities(t1);
        let e2 = extract_entities(t2);

        // Both turns should pull "deploy target" — the supersede heuristic
        // depended on this and Phase 1 spaCy NER missed it.
        assert!(e1.iter().any(|e| e == "deploy target"), "t1 entities: {:?}", e1);
        assert!(e2.iter().any(|e| e == "deploy target"), "t2 entities: {:?}", e2);

        // Postgres should be caught as a proper noun.
        assert!(e1.iter().any(|e| e == "postgres"), "t1 entities: {:?}", e1);
    }

    #[test]
    fn stopwords_dont_dominate() {
        let e = extract_entities("the the the the the");
        assert!(e.is_empty(), "stopwords leaked: {:?}", e);
    }

    #[test]
    fn empty_text() {
        assert!(extract_entities("").is_empty());
        assert!(extract_entities("   ").is_empty());
    }
}
