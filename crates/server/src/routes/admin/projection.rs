//! Initial positions + edges for the mind-map view.
//!
//! Server-side we:
//!   1. Mean-center the embedding set and compute the first two principal
//!      components via power iteration. Projecting each memory onto these
//!      two axes gives an initial (x, y) that already has clusters roughly
//!      separated. The client then runs a few hundred force-directed
//!      iterations to clean up overlaps and emphasize structure.
//!   2. Emit edges in three kinds:
//!         - `supersede`  : explicit chain links (strong)
//!         - `entity`     : Jaccard ≥ 0.4 on entities (medium)
//!         - `session`    : same session_id (weak)
//!
//! The client picks rendering weights per kind.

use std::collections::HashMap;

use uuid::Uuid;

pub struct GraphInput {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub entities: Vec<String>,
    pub session_id: Option<String>,
    pub supersedes: Option<Uuid>,
}

pub struct GraphEdge {
    pub source: Uuid,
    pub target: Uuid,
    pub kind: &'static str, // "supersede" | "entity" | "session"
    pub weight: f32,
}

pub struct GraphLayout {
    pub coords: HashMap<Uuid, (f32, f32)>,
    pub coords_3d: HashMap<Uuid, (f32, f32, f32)>,
    pub edges: Vec<GraphEdge>,
}

pub fn build_graph(items: &[GraphInput]) -> GraphLayout {
    let coords = pca_layout(items);
    // PCA-3D as initial seed, then refine with UMAP-style SGD.
    let pca_seed = pca_layout_3d(items);
    let coords_3d = umap_refine_3d(items, &pca_seed);
    let edges = build_edges(items);
    GraphLayout {
        coords,
        coords_3d,
        edges,
    }
}

/// UMAP-style refinement on top of PCA seed positions. ~250 LOC implementing
/// the McInnes-2018 sketch with the simplifications:
///   - cosine distance instead of full local-bandwidth fitting
///   - PCA-3D seed instead of spectral init (faster, similar quality)
///   - fixed k-NN size (15) + fixed negative sample count (5)
///   - SGD with the same attractive + negative-sample-repulsive scheme UMAP uses
///
/// The output preserves local neighborhood structure dramatically better than
/// raw PCA for the mind-map view — clusters that overlap in PCA space get
/// pulled apart by the SGD.
fn umap_refine_3d(
    items: &[GraphInput],
    seed: &HashMap<Uuid, (f32, f32, f32)>,
) -> HashMap<Uuid, (f32, f32, f32)> {
    let n = items.len();
    if n < 4 {
        // Not enough data for UMAP to do anything useful — just return seed.
        return seed.clone();
    }
    // Collect embeddings + ids in stable order.
    let mut ids: Vec<Uuid> = Vec::with_capacity(n);
    let mut embs: Vec<&[f32]> = Vec::with_capacity(n);
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    for it in items {
        if it.embedding.is_empty() {
            // Skip embedding-less rows from UMAP; keep their seed positions.
            continue;
        }
        ids.push(it.id);
        embs.push(&it.embedding);
        let p = seed.get(&it.id).copied().unwrap_or((0.0, 0.0, 0.0));
        positions.push([p.0, p.1, p.2]);
    }
    let m = ids.len();
    if m < 4 {
        return seed.clone();
    }

    // 1. Build k-NN graph on cosine distance. k=min(15, m-1).
    let k = std::cmp::min(15, m - 1);
    let knn = knn_cosine(&embs, k);

    // 2. Fuzzy simplicial set — for each neighbor edge, compute a weight
    //    p_ij = exp(-(d_ij - rho_i) / sigma_i). For simplicity we set
    //    sigma_i = mean(d for neighbors of i) and rho_i = min(d) (so the
    //    nearest neighbor gets weight ≈ 1). Then symmetrize via
    //    p_ij ∪ p_ji = p_ij + p_ji - p_ij*p_ji.
    let mut edges: HashMap<(usize, usize), f32> = HashMap::new();
    for i in 0..m {
        let neighbors = &knn[i];
        if neighbors.is_empty() {
            continue;
        }
        let rho = neighbors[0].1.max(1e-6);
        let sigma = (neighbors.iter().map(|(_, d)| *d).sum::<f32>() / (neighbors.len() as f32))
            .max(1e-6);
        for &(j, d) in neighbors {
            let w = ((-(d - rho).max(0.0) / sigma).exp()).clamp(0.0, 1.0);
            let key = if i < j { (i, j) } else { (j, i) };
            // OR-combine when edge already exists (from the other direction).
            let combined = edges.get(&key).copied().unwrap_or(0.0);
            let merged = combined + w - combined * w;
            edges.insert(key, merged);
        }
    }

    // 3. SGD over edges. Standard UMAP a/b are ~1.93 / 0.79 for min_dist=0.1;
    //    we use slightly looser values to keep clusters visually distinct.
    let edge_list: Vec<(usize, usize, f32)> =
        edges.into_iter().map(|((i, j), w)| (i, j, w)).collect();
    let n_epochs = 200usize;
    let initial_lr = 1.0f32;
    let a = 1.8f32;
    let b = 0.8f32;
    let neg_samples = 5usize;
    // PCG-style cheap RNG seed. Deterministic so the map doesn't dance
    // between page loads.
    let mut rng_state: u64 = 0x9E3779B97F4A7C15;

    for epoch in 0..n_epochs {
        let lr = initial_lr * (1.0 - (epoch as f32 / n_epochs as f32));
        for &(i, j, weight) in &edge_list {
            // Attractive force.
            let mut d2 = 0.0f32;
            for k in 0..3 {
                let diff = positions[i][k] - positions[j][k];
                d2 += diff * diff;
            }
            // grad of cross-entropy term wrt distance² ≈ -2ab d² / (1 + a d²ᵇ)
            let grad_coef = (-2.0 * a * b * d2.powf(b - 1.0)) / (1.0 + a * d2.powf(b));
            for k in 0..3 {
                let diff = positions[i][k] - positions[j][k];
                let g = (grad_coef * diff * weight).clamp(-4.0, 4.0);
                positions[i][k] += g * lr;
                positions[j][k] -= g * lr;
            }

            // Negative sampling: push i away from neg_samples random others.
            for _ in 0..neg_samples {
                let rj = next_rand(&mut rng_state) as usize % m;
                if rj == i {
                    continue;
                }
                let mut d2 = 0.0f32;
                for k in 0..3 {
                    let diff = positions[i][k] - positions[rj][k];
                    d2 += diff * diff;
                }
                if d2 < 1e-6 {
                    continue;
                }
                let grad_coef = (2.0 * b) / ((0.001 + d2) * (1.0 + a * d2.powf(b)));
                for k in 0..3 {
                    let diff = positions[i][k] - positions[rj][k];
                    let g = (grad_coef * diff).clamp(-4.0, 4.0);
                    positions[i][k] += g * lr;
                }
            }
        }
    }

    // 4. Re-normalize to [-1, 1] cube so the map renderer doesn't have to
    //    adjust camera.
    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for k in 0..3 {
            mins[k] = mins[k].min(p[k]);
            maxs[k] = maxs[k].max(p[k]);
        }
    }
    let mut out: HashMap<Uuid, (f32, f32, f32)> = seed.clone();
    for (id, p) in ids.iter().zip(positions.iter()) {
        let rx = (maxs[0] - mins[0]).max(1e-6);
        let ry = (maxs[1] - mins[1]).max(1e-6);
        let rz = (maxs[2] - mins[2]).max(1e-6);
        let x = ((p[0] - mins[0]) / rx) * 2.0 - 1.0;
        let y = ((p[1] - mins[1]) / ry) * 2.0 - 1.0;
        let z = ((p[2] - mins[2]) / rz) * 2.0 - 1.0;
        out.insert(*id, (x, y, z));
    }
    out
}

/// k-nearest neighbors by cosine DISTANCE (1 - cosine similarity). Returns
/// neighbor index + distance, sorted ascending. O(N²) — bounded by our 500
/// cap upstream.
fn knn_cosine(embs: &[&[f32]], k: usize) -> Vec<Vec<(usize, f32)>> {
    let n = embs.len();
    // Pre-compute norms.
    let norms: Vec<f32> = embs
        .iter()
        .map(|e| e.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9))
        .collect();
    let mut out: Vec<Vec<(usize, f32)>> = vec![Vec::with_capacity(k); n];
    for i in 0..n {
        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut dot = 0.0f32;
            for (a, b) in embs[i].iter().zip(embs[j]) {
                dot += a * b;
            }
            let sim = dot / (norms[i] * norms[j]);
            let dist = (1.0 - sim).max(0.0);
            dists.push((j, dist));
        }
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        out[i] = dists;
    }
    out
}

/// xorshift-style fast PRNG. Deterministic seed → same layout for same data.
fn next_rand(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// PCA initial layout
// ---------------------------------------------------------------------------

fn pca_layout(items: &[GraphInput]) -> HashMap<Uuid, (f32, f32)> {
    let mut out = HashMap::with_capacity(items.len());
    if items.is_empty() {
        return out;
    }

    let with_emb: Vec<&GraphInput> = items.iter().filter(|i| !i.embedding.is_empty()).collect();
    if with_emb.len() < 2 {
        for it in items {
            out.insert(it.id, (0.0, 0.0));
        }
        return out;
    }

    let d = with_emb[0].embedding.len();
    let n = with_emb.len();

    // Mean-center.
    let mut mean = vec![0.0f32; d];
    for it in &with_emb {
        for (i, x) in it.embedding.iter().enumerate() {
            mean[i] += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f32;
    }
    let centered: Vec<Vec<f32>> = with_emb
        .iter()
        .map(|it| it.embedding.iter().zip(&mean).map(|(x, m)| x - m).collect())
        .collect();

    let pc1 = power_iteration(&centered, d, 50, None);
    let deflated: Vec<Vec<f32>> = centered
        .iter()
        .map(|row| {
            let proj = dot(row, &pc1);
            row.iter().zip(&pc1).map(|(x, p)| x - proj * p).collect()
        })
        .collect();
    let pc2 = power_iteration(&deflated, d, 50, Some(&pc1));

    let mut coords: Vec<(f32, f32)> = centered
        .iter()
        .map(|row| (dot(row, &pc1), dot(row, &pc2)))
        .collect();

    // Range-normalize to [-1, 1].
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for &(x, y) in &coords {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let rx = (max_x - min_x).max(1e-6);
    let ry = (max_y - min_y).max(1e-6);
    for c in coords.iter_mut() {
        c.0 = ((c.0 - min_x) / rx) * 2.0 - 1.0;
        c.1 = ((c.1 - min_y) / ry) * 2.0 - 1.0;
    }

    for (it, c) in with_emb.iter().zip(coords.iter()) {
        out.insert(it.id, *c);
    }
    // Items with no embedding (state_object, etc.) get random-ish positions.
    let mut next_angle: f32 = 0.0;
    for it in items {
        out.entry(it.id).or_insert_with(|| {
            next_angle += 1.5;
            (next_angle.cos() * 0.8, next_angle.sin() * 0.8)
        });
    }
    out
}

/// Same as `pca_layout` but emits three principal components for the 3D
/// canvas renderer. Same input embeddings, same axes 1 & 2 — adds axis 3.
fn pca_layout_3d(items: &[GraphInput]) -> HashMap<Uuid, (f32, f32, f32)> {
    let mut out = HashMap::with_capacity(items.len());
    if items.is_empty() {
        return out;
    }

    let with_emb: Vec<&GraphInput> = items.iter().filter(|i| !i.embedding.is_empty()).collect();
    if with_emb.len() < 3 {
        for it in items {
            out.insert(it.id, (0.0, 0.0, 0.0));
        }
        return out;
    }

    let d = with_emb[0].embedding.len();
    let n = with_emb.len();

    let mut mean = vec![0.0f32; d];
    for it in &with_emb {
        for (i, x) in it.embedding.iter().enumerate() {
            mean[i] += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f32;
    }
    let centered: Vec<Vec<f32>> = with_emb
        .iter()
        .map(|it| it.embedding.iter().zip(&mean).map(|(x, m)| x - m).collect())
        .collect();

    let pc1 = power_iteration(&centered, d, 50, None);
    let deflated_1: Vec<Vec<f32>> = centered
        .iter()
        .map(|row| {
            let proj = dot(row, &pc1);
            row.iter().zip(&pc1).map(|(x, p)| x - proj * p).collect()
        })
        .collect();
    let pc2 = power_iteration(&deflated_1, d, 50, Some(&pc1));
    let deflated_2: Vec<Vec<f32>> = deflated_1
        .iter()
        .map(|row| {
            let proj = dot(row, &pc2);
            row.iter().zip(&pc2).map(|(x, p)| x - proj * p).collect()
        })
        .collect();
    let pc3 = power_iteration(&deflated_2, d, 50, Some(&pc2));

    let mut coords: Vec<(f32, f32, f32)> = centered
        .iter()
        .map(|row| (dot(row, &pc1), dot(row, &pc2), dot(row, &pc3)))
        .collect();

    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f32::INFINITY, f32::NEG_INFINITY);
    for &(x, y, z) in &coords {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    let rx = (max_x - min_x).max(1e-6);
    let ry = (max_y - min_y).max(1e-6);
    let rz = (max_z - min_z).max(1e-6);
    for c in coords.iter_mut() {
        c.0 = ((c.0 - min_x) / rx) * 2.0 - 1.0;
        c.1 = ((c.1 - min_y) / ry) * 2.0 - 1.0;
        c.2 = ((c.2 - min_z) / rz) * 2.0 - 1.0;
    }

    for (it, c) in with_emb.iter().zip(coords.iter()) {
        out.insert(it.id, *c);
    }
    let mut next_angle: f32 = 0.0;
    for it in items {
        out.entry(it.id).or_insert_with(|| {
            next_angle += 1.5;
            (
                next_angle.cos() * 0.6,
                next_angle.sin() * 0.6,
                (next_angle * 0.7).sin() * 0.6,
            )
        });
    }
    out
}

fn power_iteration(rows: &[Vec<f32>], d: usize, iters: usize, deflate: Option<&[f32]>) -> Vec<f32> {
    let mut v = vec![0.0f32; d];
    v[0] = 1.0;
    for i in 0..d.min(8) {
        v[i] += 0.1 * (i as f32);
    }
    normalize(&mut v);

    for _ in 0..iters {
        let mut xv = vec![0.0f32; rows.len()];
        for (k, row) in rows.iter().enumerate() {
            xv[k] = dot(row, &v);
        }
        let mut new_v = vec![0.0f32; d];
        for (k, row) in rows.iter().enumerate() {
            for (i, x) in row.iter().enumerate() {
                new_v[i] += x * xv[k];
            }
        }
        if let Some(prev) = deflate {
            let p = dot(&new_v, prev);
            for (a, b) in new_v.iter_mut().zip(prev) {
                *a -= p * b;
            }
        }
        normalize(&mut new_v);
        v = new_v;
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Edge construction
// ---------------------------------------------------------------------------

fn build_edges(items: &[GraphInput]) -> Vec<GraphEdge> {
    let mut edges = Vec::new();

    // 1. Supersede edges — strong; always included.
    for it in items {
        if let Some(prev) = it.supersedes {
            edges.push(GraphEdge {
                source: prev,
                target: it.id,
                kind: "supersede",
                weight: 1.0,
            });
        }
    }

    // 2. Entity-overlap edges — Jaccard ≥ 0.4. O(n²) but bounded by N ≤ 500
    //    (we cap items in the handler).
    for i in 0..items.len() {
        for j in (i + 1)..items.len() {
            let a = &items[i];
            let b = &items[j];
            if a.entities.is_empty() || b.entities.is_empty() {
                continue;
            }
            let j_score = jaccard(&a.entities, &b.entities);
            if j_score >= 0.4 {
                edges.push(GraphEdge {
                    source: a.id,
                    target: b.id,
                    kind: "entity",
                    weight: j_score,
                });
            }
        }
    }

    // 3. Same-session edges — chronological neighbors only (avoid N² per session).
    //    Bucket by session, sort by id (stable), connect i→i+1.
    let mut by_session: HashMap<&str, Vec<&GraphInput>> = HashMap::new();
    for it in items {
        if let Some(s) = it.session_id.as_deref() {
            by_session.entry(s).or_default().push(it);
        }
    }
    for (_s, members) in &mut by_session {
        members.sort_by_key(|m| m.id);
        for w in members.windows(2) {
            edges.push(GraphEdge {
                source: w[0].id,
                target: w[1].id,
                kind: "session",
                weight: 0.5,
            });
        }
    }

    edges
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    let sa: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let sb: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}
