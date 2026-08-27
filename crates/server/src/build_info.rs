//! What this binary actually is, decided at compile time.

use chrono::{DateTime, Utc};

pub const COMMIT: &str = env!("FLASHBACK_GIT_COMMIT");
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The build time as a UTC timestamp. Stamped as epoch seconds by the build
/// script because formatting there would mean a build dependency on a date
/// library for a string that is trivially derived here.
pub fn built_at() -> DateTime<Utc> {
    let secs: i64 = env!("FLASHBACK_BUILD_EPOCH").parse().unwrap_or(0);
    DateTime::from_timestamp(secs, 0).unwrap_or_default()
}

/// One line, for `--version` and for logs.
pub fn summary() -> String {
    format!(
        "flashback {VERSION} ({COMMIT}) built {}",
        built_at().format("%Y-%m-%d %H:%M:%S UTC")
    )
}

/// The same facts as structured data, for `/health` and the admin footer.
pub fn as_json() -> serde_json::Value {
    serde_json::json!({
        "version": VERSION,
        "commit": COMMIT,
        "built_at": built_at().to_rfc3339(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_names_the_version_and_the_commit() {
        let s = summary();
        assert!(s.starts_with("flashback "), "got {s}");
        assert!(s.contains(VERSION), "version missing from {s}");
        assert!(s.contains(COMMIT), "commit missing from {s}");
        assert!(s.contains("built "), "build time missing from {s}");
    }

    #[test]
    fn commit_is_safe_to_render_into_html() {
        // The value reaches an admin page without escaping, so the constraint
        // applied when it is stamped is what keeps that sound.
        assert!(
            COMMIT
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.'),
            "unexpected character in {COMMIT}"
        );
        assert!(!COMMIT.is_empty());
    }

    #[test]
    fn json_carries_the_three_facts() {
        let v = as_json();
        assert_eq!(v["version"], VERSION);
        assert_eq!(v["commit"], COMMIT);
        let built = v["built_at"].as_str().expect("built_at is a string");
        assert!(
            DateTime::parse_from_rfc3339(built).is_ok(),
            "built_at is not rfc3339: {built}"
        );
    }

    #[test]
    fn build_time_is_stamped_not_defaulted() {
        // A zero epoch means the build script did not run or could not read a
        // clock, which would make every binary claim the same build time.
        assert!(
            built_at().timestamp() > 0,
            "build time was never stamped: {}",
            built_at()
        );
    }
}
