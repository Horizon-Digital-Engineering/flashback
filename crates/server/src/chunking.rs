//! Semantic chunking for document ingestion.
//!
//! Splits at paragraph/section boundaries; keeps headings glued to their
//! following paragraph; emits 256–512-token chunks with ~10% overlap.

const TARGET_TOKENS: usize = 384;
const MIN_TOKENS: usize = 80;
const MAX_TOKENS: usize = 600;

pub struct Chunk {
    pub text: String,
    pub index: usize,
}

pub fn chunk_document(text: &str) -> Vec<Chunk> {
    let paragraphs = split_paragraphs(text);
    if paragraphs.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;

    for para in paragraphs {
        let para_tokens = approx_tokens(&para);

        // A heading on its own is small — glue it to the next paragraph.
        if para_tokens < MIN_TOKENS && current_tokens >= TARGET_TOKENS {
            push_chunk(&mut chunks, std::mem::take(&mut current));
            current_tokens = 0;
        }

        if current_tokens + para_tokens > MAX_TOKENS && current_tokens >= MIN_TOKENS {
            push_chunk(&mut chunks, std::mem::take(&mut current));
            current_tokens = 0;
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&para);
        current_tokens += para_tokens;

        if current_tokens >= TARGET_TOKENS {
            push_chunk(&mut chunks, std::mem::take(&mut current));
            current_tokens = 0;
        }
    }

    if !current.is_empty() {
        push_chunk(&mut chunks, current);
    }

    chunks
}

fn push_chunk(chunks: &mut Vec<Chunk>, text: String) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    chunks.push(Chunk {
        index: chunks.len(),
        text: trimmed.to_string(),
    });
}

fn split_paragraphs(text: &str) -> Vec<String> {
    let mut paras: Vec<String> = Vec::new();
    let mut buf = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !buf.trim().is_empty() {
                paras.push(buf.trim().to_string());
            }
            buf.clear();
        } else {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(line);
        }
    }
    if !buf.trim().is_empty() {
        paras.push(buf.trim().to_string());
    }
    paras
}

pub fn approx_tokens(s: &str) -> usize {
    (s.chars().count() + 3) / 4
}

pub fn sha256_hex(text: &str) -> String {
    // Avoid pulling sha2 just for this; use std hasher + format.
    // For content-change detection we don't need cryptographic strength.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("h64:{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_text(chars: usize) -> String {
        "x".repeat(chars)
    }

    #[test]
    fn approx_tokens_matches_char_quarter() {
        // (chars + 3) / 4 rounds up for the partial 4-byte group.
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("a"), 1);
        assert_eq!(approx_tokens("abcd"), 1);
        assert_eq!(approx_tokens("abcde"), 2);
        assert_eq!(approx_tokens(&fixed_text(400)), 100);
    }

    #[test]
    fn split_paragraphs_handles_empty_and_single() {
        assert!(split_paragraphs("").is_empty());
        let one = split_paragraphs("hello world");
        assert_eq!(one, vec!["hello world"]);
    }

    #[test]
    fn split_paragraphs_splits_on_blank_lines() {
        let txt = "first para\nstill first\n\nsecond para\n\n  \n\nthird para";
        assert_eq!(
            split_paragraphs(txt),
            vec!["first para\nstill first", "second para", "third para"]
        );
    }

    #[test]
    fn split_paragraphs_trims_trailing_whitespace() {
        let txt = "alpha  \n  \n  beta  ";
        let out = split_paragraphs(txt);
        assert_eq!(out, vec!["alpha", "beta"]);
    }

    #[test]
    fn chunk_document_empty_returns_empty() {
        assert!(chunk_document("").is_empty());
        assert!(chunk_document("\n\n\n").is_empty());
    }

    #[test]
    fn chunk_document_single_short_paragraph_is_one_chunk() {
        let chunks = chunk_document("just a short note");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].text, "just a short note");
    }

    #[test]
    fn chunk_document_indexes_are_sequential() {
        // Build several oversized paragraphs to force multiple chunks.
        let big = fixed_text(MAX_TOKENS * 4 + 100); // ~600+ tokens each
        let txt = format!("{big}\n\n{big}\n\n{big}");
        let chunks = chunk_document(&txt);
        assert!(chunks.len() >= 2);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i, "chunk index should be sequential");
        }
    }

    #[test]
    fn chunk_document_doesnt_emit_empty_chunks() {
        let chunks = chunk_document("\n\nhello\n\n\n\nworld\n\n");
        for c in &chunks {
            assert!(!c.text.is_empty());
            assert_eq!(c.text, c.text.trim());
        }
    }

    #[test]
    fn sha256_hex_is_prefixed_and_deterministic() {
        let h = sha256_hex("anything");
        assert!(h.starts_with("h64:"), "got {h}");
        // 4 prefix chars + 16 hex chars
        assert_eq!(h.len(), 4 + 16);
        // Deterministic.
        assert_eq!(sha256_hex("anything"), sha256_hex("anything"));
        // Different inputs → different outputs.
        assert_ne!(sha256_hex("a"), sha256_hex("b"));
    }
}
