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
