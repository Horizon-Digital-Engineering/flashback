//! Ebbinghaus decay + reference weighting over the curated layer.
//!
//! The retention curve is the paper Ebbinghaus formula:
//!
//! ```text
//!     R = exp(-t / S)
//! ```
//!
//! where `t` is the hours since a ref was last accessed and `S` is the
//! half-life (in hours) of that ref's `decay_class`. `R` is in `(0, 1]`: a
//! freshly-accessed ref (`t = 0`) has `R = 1`; a ref left untouched for `S`
//! hours has `R = 1/e ≈ 0.368`; retention keeps falling but never reaches
//! zero. That's the point — decay only *demotes* a ref from the active feed
//! (it lowers its ranking weight), it never deletes anything. Raw is immutable;
//! the whole `ref_weights` table is a derived, rebuildable signal.
//!
//! `ref_id` is polymorphic: it is either a `raw_records.id` or a
//! `curated_nodes.id`, disambiguated by `ref_kind` (`'raw'` | `'curated'`).
//! this layer only weights curated nodes, but the column carries either so the
//! raw backstop can be weighted later without a schema change.

use chrono::Utc;
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::AppResult;

/// Reference kinds stored in `ref_weights.ref_kind`. `ref_id` is polymorphic —
/// a `raw_records.id` (`REF_KIND_RAW`) or a `curated_nodes.id`
/// (`REF_KIND_CURATED`). This layer only weights curated nodes; the raw
/// constant documents the convention for the future raw-backstop weighting.
#[allow(dead_code)]
pub const REF_KIND_RAW: &str = "raw";
pub const REF_KIND_CURATED: &str = "curated";

/// Default decay class when a ref doesn't name one.
pub const DECAY_CLASS_DEFAULT: &str = "default";

/// Half-life S (hours) for a named decay class. This is the fallback used when
/// a `ref_weights` row has no explicit `s_hours` override. Values are chosen so
/// retention decays toward `1/e` over a human-meaningful window per class:
///   - `pinned`    : ~90 days — barely decays; the "keep this around" class.
///   - `default`   : ~14 days — the everyday curated-node curve.
///   - `ephemeral` : ~2 days  — scratch/working material that should fade fast.
///
/// An unknown class falls back to the `default` half-life. Mirrors the CASE in
/// the retrieval ranking SQL (`assemble_inner`); keep the two in sync.
#[allow(dead_code)]
pub fn half_life_hours(decay_class: &str) -> f64 {
    match decay_class {
        "pinned" => 90.0 * 24.0,
        "ephemeral" => 2.0 * 24.0,
        // "default" and anything unrecognized.
        _ => 14.0 * 24.0,
    }
}

/// The Ebbinghaus retention `R = exp(-t / S)`.
///
/// `t_hours` is time since last access; `s_hours` is the half-life. A
/// non-positive `s_hours` is treated as "never decays" (`R = 1`) rather than
/// dividing by zero. The retrieval hot path recomputes this inline in SQL for
/// batch ranking; this is the single-ref helper (tests + ad-hoc callers).
#[allow(dead_code)]
pub fn retention(t_hours: f64, s_hours: f64) -> f64 {
    if s_hours <= 0.0 {
        return 1.0;
    }
    let t = t_hours.max(0.0);
    (-t / s_hours).exp()
}

/// Compute the current decay weight for a ref from its stored `ref_weights`
/// row: `R = exp(-t / S)` with `t = now - last_access` and `S` from the row's
/// `s_hours` (or its `decay_class` default). A ref with no row yet is treated
/// as freshly seen (`R = 1.0`) — absence of a weight never demotes a node. The
/// single-ref helper; retrieval ranking recomputes the same term inline in SQL.
#[allow(dead_code)]
pub async fn decay_weight(pool: &PgPool, ref_id: Uuid) -> AppResult<f64> {
    let row: Option<(f64, chrono::DateTime<Utc>, String, Option<f64>)> = sqlx::query_as(
        "SELECT weight::float8, last_access, decay_class, s_hours \
         FROM ref_weights WHERE ref_id = $1",
    )
    .bind(ref_id)
    .fetch_optional(pool)
    .await?;

    let Some((_weight, last_access, decay_class, s_hours)) = row else {
        return Ok(1.0);
    };
    let s = s_hours.unwrap_or_else(|| half_life_hours(&decay_class));
    let t_hours = (Utc::now() - last_access).num_seconds() as f64 / 3600.0;
    Ok(retention(t_hours, s))
}

/// Bump `last_access` to now for a set of refs and recompute their stored
/// `weight` to the fresh retention (`R = 1.0` at `t = 0`). Called on retrieval
/// for every returned node so accessing a node resets its decay clock. Upserts:
/// a ref with no row yet gets one at full weight. Never deletes.
pub async fn touch_refs(pool: &PgPool, ref_kind: &str, ref_ids: &[Uuid]) -> AppResult<()> {
    if ref_ids.is_empty() {
        return Ok(());
    }
    let now = Utc::now();
    // One upsert per ref keeps the runtime API simple and stays within the
    // per-retrieval limit (a page of results). ON CONFLICT resets the clock.
    for id in ref_ids {
        sqlx::query(
            "INSERT INTO ref_weights (ref_id, ref_kind, weight, last_access, decay_class) \
             VALUES ($1, $2, 1.0, $3, $4) \
             ON CONFLICT (ref_id) DO UPDATE \
             SET weight = 1.0, last_access = EXCLUDED.last_access, ref_kind = EXCLUDED.ref_kind",
        )
        .bind(id)
        .bind(ref_kind)
        .bind(now)
        .bind(DECAY_CLASS_DEFAULT)
        .execute(pool)
        .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_at_zero_is_one() {
        assert!((retention(0.0, 336.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn retention_at_one_half_life_is_inverse_e() {
        // R(S, S) = e^-1 ≈ 0.36787944117
        let r = retention(336.0, 336.0);
        assert!((r - std::f64::consts::E.recip()).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn retention_known_values() {
        // t = 2S -> e^-2; t = 3S -> e^-3.
        let s = 100.0;
        assert!((retention(200.0, s) - (-2.0_f64).exp()).abs() < 1e-9);
        assert!((retention(300.0, s) - (-3.0_f64).exp()).abs() < 1e-9);
    }

    #[test]
    fn retention_is_monotonic_decreasing_in_t() {
        let s = 50.0;
        let mut prev = retention(0.0, s);
        for t in [1.0, 10.0, 25.0, 50.0, 100.0, 1000.0] {
            let r = retention(t, s);
            assert!(r < prev, "R must fall as t grows: {r} !< {prev} at t={t}");
            assert!(r > 0.0, "R never reaches zero: {r} at t={t}");
            prev = r;
        }
    }

    #[test]
    fn non_positive_half_life_never_decays() {
        assert_eq!(retention(1000.0, 0.0), 1.0);
        assert_eq!(retention(1000.0, -5.0), 1.0);
    }

    #[test]
    fn half_life_classes_ordered_pinned_gt_default_gt_ephemeral() {
        assert!(half_life_hours("pinned") > half_life_hours("default"));
        assert!(half_life_hours("default") > half_life_hours("ephemeral"));
        // Unknown class falls back to default.
        assert_eq!(half_life_hours("nonsense"), half_life_hours("default"));
    }
}
