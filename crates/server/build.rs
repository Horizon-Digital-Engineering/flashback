//! Stamps the binary with the commit it was built from.
//!
//! Without this the only evidence of what is deployed is a file timestamp, and
//! a timestamp records when a file was copied rather than what is inside it —
//! an installer that copies a stale artifact produces a file dated today
//! holding code from weeks ago.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn main() {
    // A build from a source archive has no repository. Say so rather than
    // failing: an unknown commit is still better than a silent wrong one.
    let sha = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".to_string());

    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);

    let commit = if dirty { format!("{sha}-dirty") } else { sha };

    // This string is rendered into HTML without escaping, so it is constrained
    // here rather than at each place it is displayed. Everything legitimate is
    // hex, a dash, or a dot.
    let commit: String = commit
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '.')
        .collect();

    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    println!("cargo:rustc-env=FLASHBACK_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FLASHBACK_BUILD_EPOCH={epoch}");

    // Re-stamp when the checked-out commit changes, otherwise a rebuild after a
    // pull keeps reporting the previous one.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
        if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
            println!("cargo:rerun-if-changed={git_dir}/{head_ref}");
        }
    }
}
