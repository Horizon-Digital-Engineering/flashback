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
//!         - `session`    : same thread_id (weak)
//!
//! The client picks rendering weights per kind.

use std::collections::HashMap;

use uuid::Uuid;

pub struct GraphInput {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub entities: Vec<String>,
    pub thread_id: Option<String>,
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
    let Some((ids, embs, mut positions)) = collect_umap_inputs(items, seed) else {
        return seed.clone();
    };
    let m = ids.len();

    let k = std::cmp::min(15, m - 1);
    let knn = knn_cosine(&embs, k);
    let edges = build_simplicial_edges(&knn, m);

    optimize_layout_sgd(&edges, &mut positions, m);

    let mut out: HashMap<Uuid, (f32, f32, f32)> = seed.clone();
    for (id, p) in ids.iter().zip(normalize_to_cube(&positions).iter()) {
        out.insert(*id, (p[0], p[1], p[2]));
    }
    out
}

/// Filter to the embedding-bearing subset and pair each one with its PCA-seed
/// position. Returns `None` if fewer than 4 items survive (UMAP needs a
/// minimum to do anything useful; caller falls back to the raw seed).
fn collect_umap_inputs<'a>(
    items: &'a [GraphInput],
    seed: &HashMap<Uuid, (f32, f32, f32)>,
) -> Option<(Vec<Uuid>, Vec<&'a [f32]>, Vec<[f32; 3]>)> {
    let n = items.len();
    if n < 4 {
        return None;
    }
    let mut ids: Vec<Uuid> = Vec::with_capacity(n);
    let mut embs: Vec<&[f32]> = Vec::with_capacity(n);
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    for it in items {
        if it.embedding.is_empty() {
            continue;
        }
        ids.push(it.id);
        embs.push(&it.embedding);
        let p = seed.get(&it.id).copied().unwrap_or((0.0, 0.0, 0.0));
        positions.push([p.0, p.1, p.2]);
    }
    if ids.len() < 4 {
        return None;
    }
    Some((ids, embs, positions))
}

/// Fuzzy simplicial set construction. For each k-NN edge compute
/// `p_ij = exp(-(d_ij - rho_i) / sigma_i)` with `rho_i = nearest-neighbor
/// distance`, `sigma_i = mean(d_neighbors_of_i)`. Symmetrize the directed
/// graph via the probabilistic-OR `p_ij ∪ p_ji = p_ij + p_ji - p_ij*p_ji`.
fn build_simplicial_edges(knn: &[Vec<(usize, f32)>], m: usize) -> HashMap<(usize, usize), f32> {
    let mut edges: HashMap<(usize, usize), f32> = HashMap::new();
    for i in 0..m {
        let neighbors = &knn[i];
        if neighbors.is_empty() {
            continue;
        }
        let rho = neighbors[0].1.max(1e-6);
        let sigma =
            (neighbors.iter().map(|(_, d)| *d).sum::<f32>() / (neighbors.len() as f32)).max(1e-6);
        for &(j, d) in neighbors {
            let w = ((-(d - rho).max(0.0) / sigma).exp()).clamp(0.0, 1.0);
            let key = if i < j { (i, j) } else { (j, i) };
            let combined = edges.get(&key).copied().unwrap_or(0.0);
            let merged = combined + w - combined * w;
            edges.insert(key, merged);
        }
    }
    edges
}

/// Stochastic-gradient descent over the symmetric edge set. Attractive force
/// pulls connected nodes together; negative sampling pushes each node away
/// from `neg_samples` random others. Standard UMAP a/b for min_dist=0.1 are
/// ~1.93 / 0.79; we use slightly looser values (1.8 / 0.8) to keep clusters
/// visually distinct. Mutates `positions` in place.
/// The curve shape and step size the layout is running with. Grouped because
/// they travel together through every force function and mean nothing apart —
/// passing them as loose floats made the call sites unreadable and let two of
/// them be swapped without the compiler noticing.
#[derive(Clone, Copy)]
struct LayoutParams {
    a: f32,
    b: f32,
    lr: f32,
    neg_samples: usize,
}

fn optimize_layout_sgd(edges: &HashMap<(usize, usize), f32>, positions: &mut [[f32; 3]], m: usize) {
    let edge_list: Vec<(usize, usize, f32)> =
        edges.iter().map(|((i, j), w)| (*i, *j, *w)).collect();
    let n_epochs = 200usize;
    let initial_lr = 1.0f32;
    let a = 1.8f32;
    let b = 0.8f32;
    let neg_samples = 5usize;
    // PCG-style cheap RNG seed. Deterministic so the map doesn't dance
    // between page loads.
    let mut rng_state: u64 = 0x9E3779B97F4A7C15;

    for epoch in 0..n_epochs {
        let p = LayoutParams {
            a,
            b,
            lr: initial_lr * (1.0 - (epoch as f32 / n_epochs as f32)),
            neg_samples,
        };
        for &(i, j, weight) in &edge_list {
            apply_attractive_force(positions, i, j, weight, p);
            apply_negative_sampling(positions, i, m, p, &mut rng_state);
        }
    }
}

/// Attractive force: gradient of the cross-entropy term wrt distance²
/// pulls connected nodes closer with strength proportional to edge weight.
fn apply_attractive_force(
    positions: &mut [[f32; 3]],
    i: usize,
    j: usize,
    weight: f32,
    p: LayoutParams,
) {
    let LayoutParams { a, b, lr, .. } = p;
    let d2 = squared_distance(&positions[i], &positions[j]);
    let grad_coef = (-2.0 * a * b * d2.powf(b - 1.0)) / (1.0 + a * d2.powf(b));
    for k in 0..3 {
        let diff = positions[i][k] - positions[j][k];
        let g = (grad_coef * diff * weight).clamp(-4.0, 4.0);
        positions[i][k] += g * lr;
        positions[j][k] -= g * lr;
    }
}

/// Negative sampling: push `i` away from `neg_samples` randomly-chosen other
/// nodes. Skip self-pairs and degenerate near-zero distances.
fn apply_negative_sampling(
    positions: &mut [[f32; 3]],
    i: usize,
    m: usize,
    p: LayoutParams,
    rng_state: &mut u64,
) {
    let LayoutParams {
        a,
        b,
        lr,
        neg_samples,
    } = p;
    for _ in 0..neg_samples {
        let rj = next_rand(rng_state) as usize % m;
        if rj == i {
            continue;
        }
        let d2 = squared_distance(&positions[i], &positions[rj]);
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

fn squared_distance(p: &[f32; 3], q: &[f32; 3]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..3 {
        let d = p[k] - q[k];
        s += d * d;
    }
    s
}

/// Re-scale a 3D point cloud into the [-1, 1] cube on all axes so the map
/// renderer doesn't have to adjust the camera. Returns a new Vec rather than
/// mutating in place to keep the orchestrator's data flow legible.
fn normalize_to_cube(positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            mins[k] = mins[k].min(p[k]);
            maxs[k] = maxs[k].max(p[k]);
        }
    }
    let ranges = [
        (maxs[0] - mins[0]).max(1e-6),
        (maxs[1] - mins[1]).max(1e-6),
        (maxs[2] - mins[2]).max(1e-6),
    ];
    positions
        .iter()
        .map(|p| {
            [
                ((p[0] - mins[0]) / ranges[0]) * 2.0 - 1.0,
                ((p[1] - mins[1]) / ranges[1]) * 2.0 - 1.0,
                ((p[2] - mins[2]) / ranges[2]) * 2.0 - 1.0,
            ]
        })
        .collect()
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

/// Three independent passes over the same items. Kept separate because each
/// answers a different question about a pair of records, and reading one used
/// to mean scrolling past the other two.
fn build_edges(items: &[GraphInput]) -> Vec<GraphEdge> {
    let mut edges = supersede_edges(items);
    edges.extend(entity_overlap_edges(items));
    edges.extend(same_thread_edges(items));
    edges
}

/// Strong and always included: this record replaced that one.
fn supersede_edges(items: &[GraphInput]) -> Vec<GraphEdge> {
    items
        .iter()
        .filter_map(|it| {
            it.supersedes.map(|prev| GraphEdge {
                source: prev,
                target: it.id,
                kind: "supersede",
                weight: 1.0,
            })
        })
        .collect()
}

/// Jaccard >= 0.4 over extracted entities. O(n^2), bounded by the handler's cap
/// of 500 items.
fn entity_overlap_edges(items: &[GraphInput]) -> Vec<GraphEdge> {
    let mut edges = Vec::new();
    for i in 0..items.len() {
        for j in (i + 1)..items.len() {
            let (a, b) = (&items[i], &items[j]);
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
    edges
}

/// Chronological neighbours within a thread only — connecting every pair would
/// be quadratic per thread and would say nothing the ordering does not.
fn same_thread_edges(items: &[GraphInput]) -> Vec<GraphEdge> {
    let mut by_thread: HashMap<&str, Vec<&GraphInput>> = HashMap::new();
    for it in items {
        if let Some(t) = it.thread_id.as_deref() {
            by_thread.entry(t).or_default().push(it);
        }
    }
    let mut edges = Vec::new();
    for members in by_thread.values_mut() {
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
    let sa: std::collections::HashSet<&str> = a.iter().map(String::as_str).collect();
    let sb: std::collections::HashSet<&str> = b.iter().map(String::as_str).collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_distance_same_point_is_zero() {
        assert_eq!(squared_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn squared_distance_axis_aligned() {
        // Unit step on each axis → 1.0.
        assert!((squared_distance(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((squared_distance(&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((squared_distance(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn squared_distance_pythagorean_triple() {
        // (3,4,12) has length 13 → d² = 169.
        assert!((squared_distance(&[0.0; 3], &[3.0, 4.0, 12.0]) - 169.0).abs() < 1e-4);
    }

    #[test]
    fn squared_distance_is_symmetric() {
        let p = [1.5, -2.0, 3.5];
        let q = [-4.0, 0.5, 1.0];
        assert!((squared_distance(&p, &q) - squared_distance(&q, &p)).abs() < 1e-6);
    }

    #[test]
    fn normalize_to_cube_axis_aligned_corners() {
        // Two points on the body diagonal → output corners of [-1,1]³.
        let out = normalize_to_cube(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(out.len(), 2);
        assert!((out[0][0] - (-1.0)).abs() < 1e-6);
        assert!((out[1][0] - 1.0).abs() < 1e-6);
        assert!((out[0][1] - (-1.0)).abs() < 1e-6);
        assert!((out[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_to_cube_handles_degenerate_axis() {
        // All points share the same z value — the z range is 0, and the max
        // clamp to 1e-6 keeps it from dividing by zero. Output z should fall
        // somewhere in the [-1, 1] band without panicking.
        let out = normalize_to_cube(&[[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]]);
        assert_eq!(out.len(), 2);
        for p in &out {
            for axis in p {
                assert!((-1.0..=1.0).contains(axis), "out of cube: {axis}");
            }
        }
    }

    #[test]
    fn normalize_to_cube_preserves_count() {
        let pts = [[0.0; 3], [1.0; 3], [-0.5, 0.5, 0.25], [10.0, -10.0, 0.0]];
        let out = normalize_to_cube(&pts);
        assert_eq!(out.len(), pts.len());
    }

    #[test]
    fn knn_cosine_skips_self_and_returns_k() {
        // 4 points on a circle in 2D. Each point's nearest non-self neighbor
        // is its immediate angular neighbor.
        let a: &[f32] = &[1.0, 0.0];
        let b: &[f32] = &[0.0, 1.0];
        let c: &[f32] = &[-1.0, 0.0];
        let d: &[f32] = &[0.0, -1.0];
        let embs: Vec<&[f32]> = vec![a, b, c, d];
        let knn = knn_cosine(&embs, 2);

        assert_eq!(knn.len(), 4);
        for neighbors in &knn {
            assert_eq!(neighbors.len(), 2, "k=2 should yield 2 neighbors per node");
        }
        // No self-reference in any neighbor list.
        for (i, neighbors) in knn.iter().enumerate() {
            assert!(neighbors.iter().all(|(j, _)| *j != i));
        }
    }

    #[test]
    fn knn_cosine_distance_is_sorted_ascending() {
        let embs: Vec<&[f32]> = vec![
            &[1.0, 0.0, 0.0],
            &[0.9, 0.1, 0.0],  // close to first
            &[0.0, 1.0, 0.0],  // orthogonal to first
            &[-1.0, 0.0, 0.0], // antipodal to first
        ];
        let knn = knn_cosine(&embs, 3);
        // For index 0, neighbors sorted by ascending cosine distance: 1, 2, 3.
        let d0 = &knn[0];
        for w in d0.windows(2) {
            assert!(w[0].1 <= w[1].1, "not sorted: {:?}", d0);
        }
    }

    #[test]
    fn knn_cosine_collapses_to_zero_for_aligned_vectors() {
        // Two parallel vectors → cosine similarity 1, distance 0.
        let embs: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]];
        let knn = knn_cosine(&embs, 1);
        assert!(knn[0][0].1.abs() < 1e-6, "expected ~0 distance");
    }

    #[test]
    fn build_simplicial_edges_symmetrizes() {
        // 3 nodes, each with the other two as neighbors at the same distance.
        // After symmetrization every undirected pair should appear exactly once.
        let knn = vec![
            vec![(1usize, 0.5_f32), (2, 0.5)],
            vec![(0, 0.5), (2, 0.5)],
            vec![(0, 0.5), (1, 0.5)],
        ];
        let edges = build_simplicial_edges(&knn, 3);
        // 3 nodes → at most C(3,2) = 3 undirected pairs.
        assert_eq!(edges.len(), 3);
        for ((i, j), w) in &edges {
            assert!(i < j, "key not canonicalized (i < j): ({i}, {j})");
            assert!(*w > 0.0 && *w <= 1.0, "weight {w} out of (0, 1]");
        }
    }
}
