//! Shared prompt template for the extraction call.
//!
//! The same prompt feeds OpenRouter / Anthropic / OpenAI / local LLM —
//! they all produce the same `Extraction` JSON shape. Each backend wraps
//! it in its own message-format conventions but the instruction is shared.

const SYSTEM_PROMPT: &str = r#"You extract structured information from a single conversation turn or note.

Output ONLY a JSON object matching this schema. No prose, no markdown fences, no commentary.

Schema:
{
  "topic":            string | null,
  "intent":           "question" | "update" | "decision" | "task" | "opinion" | "reference" | "unknown",
  "operation":        "add" | "remove" | "replace" | "reaffirm" | "contradict" | null,
  "entities":         string[],
  "action_target":    string | null,
  "contradicts_hint": string | null,
  "confidence":       number
}

Field semantics:
- `topic` — a 1-4 word canonical phrase summarizing what this turn is about. Lowercase. Example: "deploy target", "auth middleware", "Q3 plan". Null if the turn has no clear topic.
- `intent` — the speaker's communicative goal. Use "update" when state is changing, "question" when asking, "decision" when settling something, "task" when creating a follow-up, "opinion" for preference statements, "reference" when re-mentioning without changing.
- `operation` — what the speaker is doing to the topic. "replace" supersedes a prior value. "contradict" asserts something opposite to a prior claim. Null if no state change.
- `entities` — 0-8 key noun phrases mentioned, lowercase, deduplicated. Multi-word phrases preferred over single tokens.
- `action_target` — the entity the operation acts on, if any.
- `contradicts_hint` — free-text claim being contradicted, if intent is contradict.
- `confidence` — your self-assessed confidence in this extraction, 0.0 to 1.0.

Be conservative. Return null / "unknown" / empty array when the signal is weak."#;

pub fn build_system_prompt() -> &'static str {
    SYSTEM_PROMPT
}

const DISTILL_SYSTEM_PROMPT: &str = r#"You distill a cluster of related conversation memories into a small set of factual claims.

Input: a JSON array of episodic memories that share a topic.

Output: ONLY a JSON object of the form
{ "facts": [ { "content": string, "topic": string | null, "source_episode_ids": string[], "confidence": number } ] }

Rules:
- Each `content` is a single declarative claim about the user, the system, or the project. Third person, present tense. Concise (1-2 sentences).
- Prefer fewer, denser facts (1-3 for most clusters) over many small ones.
- `topic` is the shared canonical topic phrase.
- `source_episode_ids` MUST be a subset of the input episodes' `id` field — list which episodes each fact was derived from.
- `confidence` is your self-rating, 0.0-1.0. Higher when multiple episodes agree.
- If the cluster is too noisy or inconsistent to distill, return { "facts": [] }.
- No prose, no markdown fences. JSON only."#;

pub fn build_distill_system_prompt() -> &'static str {
    DISTILL_SYSTEM_PROMPT
}

pub fn build_distill_user_prompt(episodes_json: &str) -> String {
    format!("Episodes:\n{episodes_json}")
}

pub fn build_user_prompt(text: &str, recent: &[String]) -> String {
    let mut out = String::new();
    if !recent.is_empty() {
        out.push_str("Recent prior turns for coreference resolution (do NOT extract these — context only):\n");
        for r in recent.iter().take(5) {
            out.push_str("- ");
            out.push_str(r);
            out.push('\n');
        }
        out.push('\n');
    }
    out.push_str("Turn to extract:\n");
    out.push_str(text);
    out
}
