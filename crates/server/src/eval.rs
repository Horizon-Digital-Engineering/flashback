//! `flashback eval` — score the pipeline before trusting it with a corpus.
//!
//! Feeds a JSONL slice of memories through the REAL configured pipeline —
//! ingest, curation, distillation, assembly — under an isolated eval user,
//! then asks the slice's questions and scores what came back. The output is
//! a number instead of a vibe, which is the whole point: the field's memory
//! systems measurably DROP facts during consolidation, and the only defense
//! is measuring your own before an import makes the database precious.
//!
//! Slice format, one JSON object per line:
//!
//!   {"memory": "took 5mg lisinopril", "when": "2025-11-02"}
//!   {"question": "what medication?", "expect": ["lisinopril"], "reject": ["ibuprofen"]}
//!
//! `when` backdates a memory (noon UTC). `expect` substrings must appear in
//! the served context — counted separately for the synthesis feed (did a FACT
//! carry it?) and for anything served at all (did retrieval find it?).
//! `reject` substrings must NOT appear; each hit is counted as noise.
//!
//! Everything runs under its own user id, so the eval store is fully isolated
//! from real data by the same tenancy wall every request obeys. Runs are
//! additive under that user; use a fresh `--user` for a clean slate.

use serde::Deserialize;
use sqlx::PgPool;

use crate::error::AppResult;
use crate::nlp::NlpService;
use crate::routes::records::{assemble_inner, ingest_record, AssembleRequest, IngestRecordRequest};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EvalEntry {
    Memory {
        memory: String,
        #[serde(default)]
        when: Option<String>,
    },
    Question {
        question: String,
        expect: Vec<String>,
        #[serde(default)]
        reject: Vec<String>,
    },
}

/// Parse a JSONL slice. Blank lines and `#` comments are skipped; a malformed
/// line is an error naming its number, not a silent drop.
pub fn parse_slice(text: &str) -> Result<Vec<EvalEntry>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let entry: EvalEntry =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e} — in {line:?}", i + 1))?;
        out.push(entry);
    }
    Ok(out)
}

#[derive(Debug, Default)]
pub struct EvalOutcome {
    pub memories: usize,
    pub questions: usize,
    /// Questions whose every `expect` appeared somewhere in the served context.
    pub retrieval_hits: usize,
    /// Questions whose every `expect` appeared in the synthesis feed itself.
    pub synthesis_hits: usize,
    /// Total `reject` substrings that appeared anywhere served.
    pub noise: usize,
    pub promoted: i64,
    pub distilled: i64,
    pub superseded: i64,
    pub clusters_failed: i64,
    pub skipped_distill: bool,
    /// Per-question lines for the report.
    pub detail: Vec<String>,
}

impl EvalOutcome {
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "seeded {} memories · promoted {} · distilled {} · superseded {} · failed clusters {}\n",
            self.memories, self.promoted, self.distilled, self.superseded, self.clusters_failed
        ));
        if self.skipped_distill {
            out.push_str("WARNING: provider cannot distill — synthesis scores measure nothing.\n");
        }
        for d in &self.detail {
            out.push_str(d);
            out.push('\n');
        }
        let pct = |n: usize| {
            if self.questions == 0 {
                0.0
            } else {
                100.0 * n as f64 / self.questions as f64
            }
        };
        out.push_str(&format!(
            "retrieval {}/{} ({:.0}%) · synthesis {}/{} ({:.0}%) · noise {}\n",
            self.retrieval_hits,
            self.questions,
            pct(self.retrieval_hits),
            self.synthesis_hits,
            self.questions,
            pct(self.synthesis_hits),
            self.noise
        ));
        out
    }
}

/// Run a parsed slice: seed every memory, run one curation pass, ask every
/// question. Pure library shape so tests drive it with stubs; the CLI wraps it.
pub async fn run_slice(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    entries: &[EvalEntry],
) -> AppResult<EvalOutcome> {
    let mut out = EvalOutcome::default();

    for e in entries {
        if let EvalEntry::Memory { memory, when } = e {
            let event_time = when.as_deref().and_then(|d| {
                chrono::NaiveDate::parse_from_str(d.trim(), "%Y-%m-%d")
                    .ok()
                    .and_then(|nd| nd.and_hms_opt(12, 0, 0))
                    .map(|dt| chrono::DateTime::from_naive_utc_and_offset(dt, chrono::Utc))
            });
            ingest_record(
                pool,
                nlp,
                user_id,
                IngestRecordRequest {
                    r#type: "document".into(),
                    content: memory.clone(),
                    event_time,
                    source: "eval:seed".into(),
                    source_ref: None,
                    project_id: None,
                    container_id: None,
                    mode: None,
                    importance: None,
                    supersedes: None,
                    payload: None,
                },
            )
            .await?;
            out.memories += 1;
        }
    }

    let stats = crate::curation::curate(pool, nlp, user_id).await?;
    out.promoted = stats.promoted;
    out.distilled = stats.distilled;
    out.superseded = stats.superseded;
    out.clusters_failed = stats.clusters_failed;
    out.skipped_distill = stats.skipped_distill;

    for e in entries {
        let EvalEntry::Question {
            question,
            expect,
            reject,
        } = e
        else {
            continue;
        };
        out.questions += 1;
        let res = assemble_inner(
            pool,
            nlp,
            user_id,
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: None,
                modes: None,
                exclude_container_id: None,
                query: Some(question.clone()),
                limit: Some(12),
            },
        )
        .await?;
        let synth_text = res
            .synthesis
            .iter()
            .map(|s| s.content.to_lowercase())
            .collect::<Vec<_>>()
            .join("\n");
        let all_text = format!(
            "{synth_text}\n{}",
            res.records
                .iter()
                .map(|r| r.content.to_lowercase())
                .collect::<Vec<_>>()
                .join("\n")
        );

        let retrieval_ok = expect.iter().all(|x| all_text.contains(&x.to_lowercase()));
        let synthesis_ok = expect
            .iter()
            .all(|x| synth_text.contains(&x.to_lowercase()));
        let noise: Vec<&String> = reject
            .iter()
            .filter(|x| all_text.contains(&x.to_lowercase()))
            .collect();

        if retrieval_ok {
            out.retrieval_hits += 1;
        }
        if synthesis_ok {
            out.synthesis_hits += 1;
        }
        out.noise += noise.len();
        out.detail.push(format!(
            "{} retrieval · {} synthesis{} — {question}",
            if retrieval_ok { "ok  " } else { "MISS" },
            if synthesis_ok { "ok  " } else { "MISS" },
            if noise.is_empty() {
                String::new()
            } else {
                format!(" · NOISE {noise:?}")
            },
        ));
    }
    Ok(out)
}

/// CLI entry: real config, real pool, real provider — measure the deployment.
pub async fn run(args: Vec<String>) -> anyhow::Result<()> {
    let file = args
        .first()
        .ok_or_else(|| anyhow::anyhow!("usage: flashback eval <slice.jsonl> [--user=<id>]"))?;
    let user = args
        .iter()
        .find_map(|a| a.strip_prefix("--user="))
        .unwrap_or("eval-harness")
        .to_string();

    let text = std::fs::read_to_string(file)?;
    let entries = parse_slice(&text).map_err(|e| anyhow::anyhow!("{file}: {e}"))?;

    let cfg = crate::config::Config::from_env()?;
    let pool = crate::db::create_pool(&cfg.database_url).await?;
    crate::db::migrate(&pool).await?;
    let provider_cfg = crate::settings::resolve_from_db(&pool, &cfg.provider).await;
    let nlp = crate::nlp::Nlp::new(
        crate::nlp::Config {
            cache_dir: cfg.fastembed_cache_dir.clone(),
        },
        &provider_cfg,
    )
    .await?;

    let outcome = run_slice(&pool, &nlp, &user, &entries).await?;
    print!("{}", outcome.render());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skips_blanks_and_comments_and_names_bad_lines() {
        let good = "# comment\n\n{\"memory\":\"a\"}\n{\"question\":\"q\",\"expect\":[\"a\"]}\n";
        assert_eq!(parse_slice(good).unwrap().len(), 2);
        let bad = "{\"memory\":\"a\"}\nnot json\n";
        let err = parse_slice(bad).unwrap_err();
        assert!(err.starts_with("line 2:"), "{err}");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn slice_scores_the_real_pipeline(pool: sqlx::PgPool) {
        use crate::error::AppError;
        use async_trait::async_trait;
        use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};
        struct Distilling;
        #[async_trait]
        impl NlpService for Distilling {
            fn provider_name(&self) -> &'static str {
                "test-distill"
            }
            fn provider_can_distill(&self) -> bool {
                true
            }
            fn embedder_model_name(&self) -> &str {
                "sentence-transformers/all-MiniLM-L6-v2"
            }
            fn embedder_dimension(&self) -> usize {
                384
            }
            async fn embed_one(&self, t: &str) -> Result<Vec<f32>, AppError> {
                Ok(crate::curation::bow_embed(t))
            }
            async fn embed_batch(&self, t: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
                Ok(t.iter().map(|x| crate::curation::bow_embed(x)).collect())
            }
            fn extract_entities(&self, t: &str) -> Vec<String> {
                flashback_nlp::extract_entities(t)
            }
            async fn extract_full(&self, _t: &str) -> Result<Extraction, AppError> {
                Ok(Extraction::empty())
            }
            async fn distill_facts(
                &self,
                e: &[EpisodeRef],
            ) -> Result<Vec<DistilledFact>, ProviderError> {
                Ok(vec![DistilledFact {
                    content: "the patient takes lisinopril daily".into(),
                    topic: None,
                    source_episode_ids: e.iter().map(|x| x.id).collect(),
                    confidence: 0.9,
                }])
            }
        }

        let entries = parse_slice(
            "{\"memory\":\"took 5mg lisinopril with breakfast\",\"when\":\"2025-11-02\"}\n\
             {\"memory\":\"note: took lisinopril with breakfast as usual\",\"when\":\"2025-12-01\"}\n\
             {\"question\":\"what medication is taken?\",\"expect\":[\"lisinopril\"],\"reject\":[\"ibuprofen\"]}",
        )
        .unwrap();
        let out = run_slice(&pool, &Distilling, "eval-test", &entries)
            .await
            .unwrap();
        assert_eq!(out.memories, 2);
        assert_eq!(out.questions, 1);
        assert_eq!(out.retrieval_hits, 1, "{}", out.render());
        assert_eq!(out.synthesis_hits, 1, "the fact itself must carry it");
        assert_eq!(out.noise, 0);
    }

    #[test]
    fn render_shows_the_numbers() {
        let o = EvalOutcome {
            memories: 3,
            questions: 2,
            retrieval_hits: 2,
            synthesis_hits: 1,
            noise: 0,
            ..Default::default()
        };
        let r = o.render();
        assert!(r.contains("retrieval 2/2 (100%)"), "{r}");
        assert!(r.contains("synthesis 1/2 (50%)"), "{r}");
    }
}
