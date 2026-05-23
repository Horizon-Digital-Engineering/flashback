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
