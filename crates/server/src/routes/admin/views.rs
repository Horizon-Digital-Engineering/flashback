//! HTML rendering helpers. Hand-rolled `format!` calls — no templating crate.
//!
//! Every view renders the canonical world: raw records, the curated layer,
//! references, the catalog, and the proposal queue.

use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::handlers::RawAdminRow;

/// Set once at startup by `main.rs`. Read by `page()` and `login_page()` to
/// render the dev banner — saves threading a `dev_mode` parameter through
/// every per-view function.
static DEV_MODE: OnceLock<bool> = OnceLock::new();

pub fn set_dev_mode(on: bool) {
    let _ = DEV_MODE.set(on);
}

fn dev_mode_on() -> bool {
    *DEV_MODE.get().unwrap_or(&false)
}

/// Page chrome: head + nav + main + footer.
pub fn page(active: &str, user_id: &str, content: &str) -> String {
    let nav = render_nav(active, user_id);
    // On every page rather than only the dashboard: the question "what is
    // running here" is asked from wherever the operator already is. The value
    // is constrained to safe characters where it is stamped, not escaped here.
    let build = crate::build_info::summary();
    let banner = if dev_mode_on() {
        r#"<div class="dev-banner">⚠ DEV MODE — auth bypassed. Every request runs as <code>user_id=dev</code>. Don't expose this server to the internet.</div>"#
    } else {
        ""
    };
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Flashback — {active}</title>
<link rel="stylesheet" href="/admin/style.css" />
</head>
<body>
{banner}
{nav}
<main class="page">
{content}
</main>
<footer class="build">{build}</footer>
{LOCAL_TIME_JS}
</body>
</html>"#
    )
}

/// Rewrite every `<time class="ts">` into the viewer's own timezone. Kept inline
/// and dependency-free; if it never runs the server-rendered "… UTC" text is
/// still correct, just not local.
const LOCAL_TIME_JS: &str = r##"<script>
for (const el of document.querySelectorAll("time.ts")) {
  const d = new Date(el.getAttribute("datetime"));
  if (!isNaN(d)) {
    el.textContent = d.toLocaleString(undefined, {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
    });
    el.title = el.getAttribute("datetime");
  }
}
</script>"##;

fn render_nav(active: &str, user_id: &str) -> String {
    let item = |name: &str, href: &str, label: &str| -> String {
        let cls = if name == active { "active" } else { "" };
        format!(r#"<a href="{href}" class="{cls}">{label}</a>"#)
    };
    format!(
        r#"<nav class="nav">
  <span class="brand">flashback ✦ admin</span>
  {a}
  {b}
  {c}
  {d}
  {e}
  {f}
  {g}
  {h}
  {i}
  {j}
  {k}
  <span class="spacer"></span>
  <span class="user">{user_id}</span>
  <a href="/admin/logout">logout</a>
</nav>"#,
        a = item("dashboard", "/admin", "Dashboard"),
        b = item("records", "/admin/records", "Records"),
        c = item("curated", "/admin/curated", "Curated"),
        d = item("state", "/admin/state", "State"),
        e = item("catalog", "/admin/catalog", "Catalog"),
        f = item("proposals", "/admin/proposals", "Proposals"),
        g = item("map", "/admin/map", "Map"),
        h = item("curate", "/admin/curate", "Curate"),
        i = item("playground", "/admin/playground", "Playground"),
        j = item("settings", "/admin/settings", "Settings"),
        k = item("tokens", "/admin/tokens", "Tokens"),
    )
}

/// Login page (unauthenticated).
pub fn login_page(error: Option<&str>) -> String {
    let err_html = match error {
        Some(msg) => format!(r#"<div class="error">{}</div>"#, esc(msg)),
        None => String::new(),
    };
    let banner = if dev_mode_on() {
        r#"<div class="dev-banner">⚠ DEV MODE — auth bypassed</div>"#
    } else {
        ""
    };
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Flashback — login</title>
<link rel="stylesheet" href="/admin/style.css" />
</head>
<body class="login-page">
  {banner}
  <div class="login-box">
    <h1>flashback ✦</h1>
    <p class="sub">paste a bearer token to access the admin UI</p>
    {err_html}
    <form method="post" action="/admin/login">
      <input type="password" name="token" placeholder="fb_..." autofocus required />
      <button type="submit">Sign in</button>
    </form>
    <p class="muted" style="margin-top:24px;font-size:12px;text-align:center">
      Don't have a token?<br />
      <code>flashback token mint --user=&lt;u&gt; --name=&lt;label&gt;</code>
    </p>
  </div>
</body>
</html>"#
    )
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

pub struct DashboardStats {
    pub records_total: i64,
    pub records_terminal: i64,
    pub state_objects: i64,
    pub curated_nodes: i64,
    pub proposals_pending: i64,
    pub tokens_active: i64,
    pub provider: String,
    pub embedder_model: String,
    pub embedder_dim: usize,
}

pub fn dashboard(user_id: &str, stats: DashboardStats, recent: &[RawAdminRow]) -> String {
    let mut recent_html =
        String::from(r#"<div class="card"><h2 style="margin-top:0">Recent records</h2>"#);
    if recent.is_empty() {
        recent_html
            .push_str(r#"<p class="muted">No records yet. Ingest some via POST /records.</p>"#);
    } else {
        recent_html.push_str(
            "<table><thead><tr><th>type</th><th>content</th><th>event</th></tr></thead><tbody>",
        );
        for r in recent {
            recent_html.push_str(&format!(
                r#"<tr><td>{ty}</td><td><a href="/admin/records/{id}"><div class="content-preview">{content}</div></a></td><td class="mono muted">{when}</td></tr>"#,
                ty = type_pill(&r.r#type),
                id = r.id,
                content = esc(&r.content),
                when = format_when(r.event_time),
            ));
        }
        recent_html.push_str("</tbody></table>");
    }
    recent_html.push_str("</div>");

    let content = format!(
        r#"<h1>Dashboard</h1>
<div class="stat-grid">
  <div class="stat"><div class="label">Records (terminal)</div><div class="value">{term}</div></div>
  <div class="stat"><div class="label">Records (all)</div><div class="value">{total}</div></div>
  <div class="stat"><div class="label">State objects</div><div class="value">{state}</div></div>
  <div class="stat"><div class="label">Curated nodes</div><div class="value">{curated}</div></div>
  <div class="stat"><div class="label">Proposals pending</div><div class="value">{proposals}</div></div>
  <div class="stat"><div class="label">Active tokens</div><div class="value">{tok}</div></div>
</div>
<div class="row">
  <div class="card">
    <h2 style="margin-top:0">Engine</h2>
    <p><span class="muted">Embedder:</span> {embed} ({dim}d)</p>
    <p><span class="muted">Extraction provider:</span> <code>{prov}</code></p>
    <p class="muted" style="margin-top:16px;font-size:12px">
      To swap to a hosted LLM: set <code>PROVIDER=remote</code> + an API key
      and restart the server.
    </p>
  </div>
  <div class="card">
    <h2 style="margin-top:0">Quick links</h2>
    <p><a href="/admin/playground">Playground — watch a turn end to end →</a></p>
    <p><a href="/admin/records">Browse records →</a></p>
    <p><a href="/admin/curated">Curated layer →</a></p>
    <p><a href="/admin/map">Embedding map →</a></p>
    <p><a href="/admin/proposals">Proposal queue →</a></p>
  </div>
</div>
{recent}"#,
        term = stats.records_terminal,
        total = stats.records_total,
        state = stats.state_objects,
        curated = stats.curated_nodes,
        proposals = stats.proposals_pending,
        tok = stats.tokens_active,
        embed = esc(&stats.embedder_model),
        dim = stats.embedder_dim,
        prov = esc(&stats.provider),
        recent = recent_html,
    );

    page("dashboard", user_id, &content)
}

// ---------------------------------------------------------------------------
// Records list
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct RecordsFilter {
    pub r#type: Option<String>,
    pub project_id: Option<String>,
    pub container_id: Option<String>,
    /// Cognitive register (mode) filter — a NATIVE `raw_records.mode` predicate.
    pub mode: Option<String>,
    pub include_superseded: bool,
}

pub fn records_list(
    user_id: &str,
    filter: &RecordsFilter,
    mode_names: &[String],
    records: &[RawAdminRow],
    total: i64,
    payload_keys: &[(String, i64)],
) -> String {
    let q = build_query_string(filter);
    let mut filter_form = String::from(
        r#"<form method="get" action="/admin/records" class="card" style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end">"#,
    );
    filter_form.push_str(&format!(
        r#"<div><label class="muted">Type</label><br />
            <select name="type" style="background:var(--bg-2);color:var(--fg-0);border:1px solid var(--border);border-radius:6px;padding:8px;min-width:140px">
              <option value="" {sel_any}>any</option>
              {opts}
            </select></div>"#,
        sel_any = if filter.r#type.is_none() { "selected" } else { "" },
        opts = ["conversation", "document", "state_object"]
            .iter()
            .map(|t| {
                let sel = if filter.r#type.as_deref() == Some(*t) { "selected" } else { "" };
                format!(r#"<option value="{t}" {sel}>{t}</option>"#)
            })
            .collect::<Vec<_>>()
            .join("\n"),
    ));
    filter_form.push_str(&format!(
        r#"<div><label class="muted">Project</label><br />
            <input type="text" name="project_id" value="{}" placeholder="any" style="min-width:160px" /></div>"#,
        esc(filter.project_id.as_deref().unwrap_or(""))
    ));
    filter_form.push_str(&format!(
        r#"<div><label class="muted">Session</label><br />
            <input type="text" name="container_id" value="{}" placeholder="any" style="min-width:160px" /></div>"#,
        esc(filter.container_id.as_deref().unwrap_or(""))
    ));
    filter_form.push_str(&format!(
        r#"<div><label class="muted">Mode</label><br />
            <select name="mode" style="background:var(--bg-2);color:var(--fg-0);border:1px solid var(--border);border-radius:6px;padding:8px;min-width:140px">
              <option value="" {sel_any}>any</option>
              {opts}
            </select></div>"#,
        sel_any = if filter.mode.is_none() { "selected" } else { "" },
        opts = mode_names
            .iter()
            .map(|m| {
                let sel = if filter.mode.as_deref() == Some(m.as_str()) { "selected" } else { "" };
                format!(r#"<option value="{m}" {sel}>{m}</option>"#, m = esc(m))
            })
            .collect::<Vec<_>>()
            .join("\n"),
    ));
    filter_form.push_str(&format!(
        r#"<div><label><input type="checkbox" name="include_superseded" value="1" {} /> include superseded</label></div>"#,
        if filter.include_superseded { "checked" } else { "" }
    ));
    filter_form.push_str(r#"<div><button type="submit">Apply</button></div>"#);
    filter_form.push_str("</form>");

    let mut table = String::from(
        r#"<table>
<thead><tr>
  <th>type</th>
  <th>content</th>
  <th>entities</th>
  <th>mode</th>
  <th>event</th>
</tr></thead>
<tbody>"#,
    );
    for r in records {
        let topic = r
            .state_key
            .as_deref()
            .map(|k| format!("<strong>{}</strong> &nbsp;", esc(k)))
            .unwrap_or_default();
        let entities = render_entities(&r.entities, 5);
        let mode = r
            .mode
            .as_deref()
            .map(|m| format!(r#"<code>{}</code>"#, esc(m)))
            .unwrap_or_else(|| r#"<span class="muted">—</span>"#.to_string());
        let sup = if r.superseded {
            r#" <span class="pill" style="color:var(--bad)">superseded</span>"#
        } else {
            ""
        };
        table.push_str(&format!(
            r#"<tr><td>{ty}</td><td><a href="/admin/records/{id}">{topic}<div class="content-preview">{content}</div></a>{sup}</td><td>{ents}</td><td>{mode}</td><td class="mono muted">{when}</td></tr>"#,
            ty = type_pill(&r.r#type),
            id = r.id,
            topic = topic,
            content = esc(&r.content),
            sup = sup,
            ents = entities,
            mode = mode,
            when = format_when(r.event_time),
        ));
    }
    if records.is_empty() {
        table.push_str(
            r#"<tr><td colspan="5"><p class="muted" style="text-align:center;padding:16px">No records match the current filter.</p></td></tr>"#,
        );
    }
    table.push_str("</tbody></table>");

    let content = format!(
        r#"<h1>Records</h1>
<p class="muted">{total} total matching this filter. URL state: <code>?{q}</code></p>
{filter_form}
{table}
{keys}"#,
        total = total,
        q = esc(&q),
        filter_form = filter_form,
        table = table,
        keys = payload_key_panel(payload_keys),
    );
    page("records", user_id, &content)
}

/// Census of the open metadata bag. `payload` keeps capture liberal — a writer
/// records every circumstance it knows without waiting for a migration — at the
/// cost of a shape you can't see at a glance. This is the counterweight: it
/// shows what is actually accumulating in there, so promoting a key to a real
/// column becomes an observation rather than a guess.
///
/// Read it as a hint, not a verdict. A key present on every row that nothing
/// ever filters by has earned nothing; a rare key you filter by constantly has
/// earned a column. Frequency is merely the half that can be measured.
fn payload_key_panel(keys: &[(String, i64)]) -> String {
    if keys.is_empty() {
        return String::new();
    }
    let mut rows = String::new();
    for (k, n) in keys {
        rows.push_str(&format!(
            r#"<tr><td class="mono">{k}</td><td class="muted">{n}</td></tr>"#,
            k = esc(k),
            n = n,
        ));
    }
    format!(
        r#"<div class="card" style="margin-top:20px">
  <h2 style="margin-top:0">Capture metadata keys</h2>
  <p class="muted" style="font-size:12px">
    What writers are putting in <code>payload</code>. A key you lean on
    repeatedly is a candidate to become its own column — migrations are
    declarations here, so promoting one later costs a rebuild, not a data loss.
  </p>
  <table><thead><tr><th>key</th><th>records</th></tr></thead><tbody>{rows}</tbody></table>
</div>"#
    )
}

fn build_query_string(f: &RecordsFilter) -> String {
    let mut parts = Vec::new();
    if let Some(t) = &f.r#type {
        parts.push(format!("type={}", t));
    }
    if let Some(p) = &f.project_id {
        parts.push(format!("project_id={}", p));
    }
    if let Some(s) = &f.container_id {
        parts.push(format!("container_id={}", s));
    }
    if let Some(m) = &f.mode {
        parts.push(format!("mode={}", m));
    }
    if f.include_superseded {
        parts.push("include_superseded=1".to_string());
    }
    parts.join("&")
}

fn render_entities(entities: &[String], take: usize) -> String {
    if entities.is_empty() {
        "<span class=\"muted\">—</span>".to_string()
    } else {
        format!(
            r#"<div class="tag-list">{}</div>"#,
            entities
                .iter()
                .take(take)
                .map(|e| format!(r#"<span class="tag">{}</span>"#, esc(e)))
                .collect::<Vec<_>>()
                .join("")
        )
    }
}

// ---------------------------------------------------------------------------
// Record detail
// ---------------------------------------------------------------------------

pub fn record_detail(user_id: &str, r: &RawAdminRow, chain: &[RawAdminRow]) -> String {
    let terminal_id = chain.iter().find(|v| !v.superseded).map(|v| v.id);
    let chain_html = render_chain(chain, terminal_id, r.id);
    let entities_html = render_entities(&r.entities, usize::MAX);

    let payload_html = match &r.payload {
        Some(v) => format!(
            r#"<pre class="json">{}</pre>"#,
            esc(&serde_json::to_string_pretty(v).unwrap_or_default())
        ),
        None => r#"<p class="muted">No structured payload on this record.</p>"#.to_string(),
    };

    let content = format!(
        r#"<p><a href="/admin/records">← back to records</a></p>
<h1>{ty} <span class="mono muted" style="font-size:14px;font-weight:normal">{id}</span></h1>

<div class="row">
  <div class="card">
    <h2 style="margin-top:0">Content</h2>
    <pre class="json">{content}</pre>
    <p class="muted" style="margin-top:12px;font-size:12px">
      event {when} · source <code>{source}</code>{imp}{mode}{sup}
    </p>
    <div style="margin-top:16px">
      <span class="muted">entities:</span> {ents}
    </div>
    {project}
    {session}
  </div>
  <div class="card">
    <h2 style="margin-top:0">Payload</h2>
    {payload_html}
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">Supersede chain ({n} node{plural})</h2>
  {chain_html}
</div>"#,
        ty = type_pill(&r.r#type),
        id = r.id,
        content = esc(&r.content),
        when = format_when(r.event_time),
        source = esc(&r.source),
        imp = r
            .importance
            .map(|i| format!(" · importance {i:.2}"))
            .unwrap_or_default(),
        mode = r
            .mode
            .as_deref()
            .map(|m| format!(" · mode <code>{}</code>", esc(m)))
            .unwrap_or_default(),
        sup = if r.superseded {
            " · <span class=\"pill\" style=\"color:var(--bad)\">superseded</span>".to_string()
        } else {
            String::new()
        },
        ents = entities_html,
        project = r
            .project_id
            .as_deref()
            .map(|p| format!(
                r#"<p class="muted" style="margin-top:6px">project: <code>{}</code></p>"#,
                esc(p)
            ))
            .unwrap_or_default(),
        session = r
            .container_id
            .as_deref()
            .map(|s| format!(r#"<p class="muted">session: <code>{}</code></p>"#, esc(s)))
            .unwrap_or_default(),
        payload_html = payload_html,
        n = chain.len(),
        plural = if chain.len() == 1 { "" } else { "s" },
        chain_html = chain_html,
    );

    page("records", user_id, &content)
}

fn render_chain(chain: &[RawAdminRow], terminal_id: Option<Uuid>, current_id: Uuid) -> String {
    if chain.is_empty() {
        return r#"<p class="muted">No supersede chain.</p>"#.to_string();
    }
    let mut out = String::from(r#"<div class="chain">"#);
    for (i, r) in chain.iter().enumerate() {
        let is_terminal = terminal_id == Some(r.id);
        let is_current = current_id == r.id;
        let cls = if is_terminal {
            "chain-node terminal"
        } else {
            "chain-node"
        };
        let marker = if is_current { " (this record)" } else { "" };
        let terminal_label = if is_terminal {
            r#" <span class="pill t-semantic">current</span>"#
        } else {
            ""
        };
        out.push_str(&format!(
            r#"<div class="{cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
    <div style="flex:1">
      <strong><a href="/admin/records/{id}">{short}</a></strong>{terminal_label} <span class="mono muted">{when}</span>{marker}
      <div class="content-preview" style="margin-top:6px;color:var(--fg-0)">{content}</div>
    </div>
  </div>
</div>"#,
            id = r.id,
            short = short_id(r.id),
            content = esc(&r.content),
            when = format_when(r.event_time),
        ));
        if i < chain.len() - 1 {
            out.push_str(r#"<div class="chain-arrow">↓ superseded by</div>"#);
        }
    }
    out.push_str("</div>");
    out
}

// ---------------------------------------------------------------------------
// Curated layer
// ---------------------------------------------------------------------------

pub struct CuratedNodeView {
    pub kind: String,
    pub content: String,
    pub level: i32,
    pub created_at: DateTime<Utc>,
}

pub fn curated_list(user_id: &str, nodes: &[CuratedNodeView]) -> String {
    let mut body = String::new();
    if nodes.is_empty() {
        body.push_str(r#"<p class="muted">No curated nodes yet. Run a curation pass from the <a href="/admin/curate">Curate</a> page (or wait for the scheduler) to promote + distill raw records.</p>"#);
    } else {
        body.push_str(
            r#"<table><thead><tr><th>kind</th><th>level</th><th>content</th><th>created</th></tr></thead><tbody>"#,
        );
        for n in nodes {
            body.push_str(&format!(
                r#"<tr><td>{kind}</td><td class="mono">{level}</td><td><div class="content-preview">{content}</div></td><td class="mono muted">{when}</td></tr>"#,
                kind = type_pill(&n.kind),
                level = n.level,
                content = esc(&n.content),
                when = format_when(n.created_at),
            ));
        }
        body.push_str("</tbody></table>");
    }
    let content = format!(
        r#"<h1>Curated layer</h1>
<p class="muted">Derived from raw and rebuildable from it: promoted episodic records and distilled semantic facts, plus higher-level summary nodes. Level 0 is raw-adjacent; higher levels are summaries.</p>
{body}"#
    );
    page("curated", user_id, &content)
}

// ---------------------------------------------------------------------------
// State objects (references over raw)
// ---------------------------------------------------------------------------

pub fn state_list(user_id: &str, states: &[RawAdminRow]) -> String {
    let mut body = String::new();
    if states.is_empty() {
        body.push_str(r#"<p class="muted">No state objects yet. Create one via POST /records/state/todo_list (or any other kind).</p>"#);
    } else {
        for s in states {
            let rendered = esc(&s.content);
            let key = esc(s.state_key.as_deref().unwrap_or("?"));
            let kind = esc(s.state_kind.as_deref().unwrap_or("?"));
            body.push_str(&format!(
                r#"<div class="card">
  <h2 style="margin-top:0"><a href="/admin/records/{id}"><code>{kind}/{key}</code></a></h2>
  <pre class="json">{rendered}</pre>
  <p class="muted" style="font-size:12px">updated {when}</p>
</div>"#,
                id = s.id,
                kind = kind,
                key = key,
                rendered = rendered,
                when = format_when(s.event_time),
            ));
        }
    }
    let content = format!(
        r#"<h1>State objects</h1>
<p class="muted">The "heap" — mutable named cells projected from state_object raw records. See <a href="https://github.com/Horizon-Digital-Engineering/flashback/blob/main/docs/REFERENCES.md">docs/REFERENCES.md</a>.</p>
{body}"#
    );
    page("state", user_id, &content)
}

// ---------------------------------------------------------------------------
// Catalog (the store map)
// ---------------------------------------------------------------------------

pub fn catalog_view(user_id: &str, catalog: &crate::catalog::CatalogView) -> String {
    let section = |title: &str, blurb: &str, stores: &[crate::catalog::StoreView]| -> String {
        let mut cards = String::new();
        if stores.is_empty() {
            cards.push_str(r#"<p class="muted">none</p>"#);
        }
        for s in stores {
            let schema_pre = match &s.store.schema {
                Some(v) => format!(
                    r#"<pre class="json">{}</pre>"#,
                    esc(&serde_json::to_string_pretty(v).unwrap_or_default())
                ),
                None => r#"<p class="muted">no declared schema</p>"#.to_string(),
            };
            let synced = match s.store.last_synced_at {
                Some(t) => format!("synced {}", format_when(t)),
                None => "never synced".to_string(),
            };
            cards.push_str(&format!(
                r#"<div class="card">
  <h3 style="margin-top:0"><code>{name}</code> <span class="pill">{kind}</span></h3>
  <p><strong>{count}</strong> records · <span class="muted">{lineage}</span></p>
  <p class="muted" style="font-size:12px">{desc}</p>
  {schema}
  <p class="muted" style="font-size:12px">{synced}</p>
</div>"#,
                name = esc(&s.store.name),
                kind = esc(&s.store.kind),
                count = s.record_count,
                lineage = esc(&s.lineage),
                desc = esc(s.store.description.as_deref().unwrap_or("")),
                schema = schema_pre,
                synced = synced,
            ));
        }
        format!(
            r#"<section style="margin-top:24px">
  <h2>{title}</h2>
  <p class="muted">{blurb}</p>
  {cards}
</section>"#
        )
    };

    let content = format!(
        r#"<h1>Catalog</h1>
<p class="muted">Every store the lake knows about, grouped by kind. The raw + curated layers register themselves with a live schema and record count; operational/external stores you register publish slices into the lake on sync.</p>
{raw}
{curated}
{operational}
{external}"#,
        raw = section("Raw", "The immutable source of truth.", &catalog.raw),
        curated = section(
            "Curated",
            "Derived from raw and rebuildable from it.",
            &catalog.curated
        ),
        operational = section(
            "Operational",
            "Live systems that publish slices into the lake.",
            &catalog.operational
        ),
        external = section(
            "External",
            "Outside sources the lake ingests from.",
            &catalog.external
        ),
    );
    page("catalog", user_id, &content)
}

// ---------------------------------------------------------------------------
// Proposals (the review queue)
// ---------------------------------------------------------------------------

pub fn proposals_view(user_id: &str, proposals: &[crate::proposals::ProposalRow]) -> String {
    let mut body = String::new();
    if proposals.is_empty() {
        body.push_str(
            r#"<p class="muted">No proposals. The lake surfaces a suggestion here — it never acts on its own; you decide.</p>"#,
        );
    }
    for p in proposals {
        let action = p.body.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let rationale = p
            .body
            .get("rationale")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let evidence = p
            .body
            .get("evidence")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let status_pill = match p.status.as_str() {
            "proposed" => r#"<span class="pill" style="color:var(--fg-0)">proposed</span>"#,
            "approved" => r#"<span class="pill" style="color:var(--good)">approved</span>"#,
            "denied" => r#"<span class="pill" style="color:var(--bad)">denied</span>"#,
            "executed" => r#"<span class="pill" style="color:var(--good)">executed</span>"#,
            _ => r#"<span class="pill">?</span>"#,
        };
        // Approve/deny are only offered while the proposal is still 'proposed'.
        // The lake never executes: there is no execute button here — completion
        // is reported by the host, not triggered from this queue.
        let actions = if p.status == "proposed" {
            format!(
                r#"<form method="post" action="/admin/proposals/{id}/approve" class="inline">
     <button type="submit" class="btn">approve</button>
   </form>
   <form method="post" action="/admin/proposals/{id}/deny" class="inline">
     <button type="submit" class="btn danger">deny</button>
   </form>"#,
                id = p.id
            )
        } else {
            let decided = match (&p.decided_by, p.decided_at) {
                (Some(who), Some(at)) => format!("by {} · {}", esc(who), format_when(at)),
                (None, Some(at)) => format_when(at),
                _ => String::new(),
            };
            format!(r#"<span class="muted" style="font-size:12px">{decided}</span>"#)
        };
        body.push_str(&format!(
            r#"<div class="card">
  <h2 style="margin-top:0">{title} <span class="pill">{kind}</span> {status}</h2>
  <p><strong>Action:</strong> {action}</p>
  {rationale_html}
  <p class="muted" style="font-size:12px">{ev} evidence record(s) · created {created}</p>
  <div style="display:flex;gap:8px">{actions}</div>
</div>"#,
            title = esc(&p.title),
            kind = esc(&p.kind),
            status = status_pill,
            action = esc(action),
            rationale_html = if rationale.is_empty() {
                String::new()
            } else {
                format!(r#"<p><strong>Rationale:</strong> {}</p>"#, esc(rationale))
            },
            ev = evidence,
            created = format_when(p.created_at),
            actions = actions,
        ));
    }
    let content = format!(
        r#"<h1>Proposals</h1>
<p class="muted">The lake proposes; you decide. Approving hands the action to the host to carry out — the lake itself never executes.</p>
{body}"#
    );
    page("proposals", user_id, &content)
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

pub struct TokenView {
    pub id: Uuid,
    pub prefix: String,
    pub user_id: String,
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

pub fn tokens_list(user_id: &str, tokens: &[TokenView]) -> String {
    let mut rows = String::new();
    for t in tokens {
        let status = match (t.revoked_at, t.last_used_at) {
            (Some(_), _) => {
                r#"<span class="pill" style="color:var(--bad)">revoked</span>"#.to_string()
            }
            (None, Some(u)) => format!(
                r#"<span class="pill" style="color:var(--good)">active</span> <span class="muted mono">used {}</span>"#,
                format_when(u)
            ),
            (None, None) => r#"<span class="pill">unused</span>"#.to_string(),
        };
        let action = if t.revoked_at.is_none() {
            format!(
                r#"<form method="post" action="/admin/tokens/{id}/revoke" class="inline" onsubmit="return confirm('Revoke this token? Clients using it will get 401.')">
                     <button type="submit" class="btn danger">revoke</button>
                   </form>"#,
                id = t.id
            )
        } else {
            String::new()
        };
        rows.push_str(&format!(
            r#"<tr>
  <td class="mono">{prefix}</td>
  <td>{name}</td>
  <td>{user}</td>
  <td>{status}</td>
  <td class="mono muted">{created}</td>
  <td class="right">{action}</td>
</tr>"#,
            prefix = esc(&t.prefix),
            name = esc(t.name.as_deref().unwrap_or("")),
            user = esc(&t.user_id),
            status = status,
            created = format_when(t.created_at),
            action = action,
        ));
    }
    let content = format!(
        r#"<h1>Tokens</h1>
<p class="muted">All bearer tokens minted on this server. Plaintext is shown only at mint time via the CLI.</p>
<table>
<thead><tr><th>prefix</th><th>name</th><th>user</th><th>status</th><th>created</th><th></th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="muted" style="margin-top:24px;font-size:12px">
  Mint a new token:<br />
  <code>flashback token mint --user=&lt;u&gt; --name=&lt;label&gt;</code>
</p>"#,
        rows = rows,
    );
    page("tokens", user_id, &content)
}

// ---------------------------------------------------------------------------
// Map view (interactive scatterplot)
// ---------------------------------------------------------------------------

pub fn map_view(user_id: &str, node_count: usize, edge_count: usize) -> String {
    let content = format!(
        r#"<h1>Mind map</h1>
<p class="muted">{n} records, {e} edges. <strong>3D scene</strong> rendered in plain canvas2d — hand-rolled perspective projection, depth-sorted, no Three.js / WebGL / framework. Drag to orbit the camera, scroll to zoom, click a node to open. Switch to 2D for the force-directed flat layout.</p>

<div class="map-wrap">
  <canvas id="map-canvas"></canvas>
  <div class="map-legend">
    <div class="key"><span class="swatch" style="background:var(--type-episodic)"></span> episodic</div>
    <div class="key"><span class="swatch" style="background:var(--type-semantic)"></span> semantic</div>
    <div class="key"><span class="swatch" style="background:var(--type-working)"></span> working</div>
    <div class="key"><span class="swatch" style="background:var(--type-document)"></span> document</div>
    <div class="key"><span class="swatch" style="background:var(--type-procedural)"></span> procedural</div>
    <div class="key"><span class="swatch" style="background:var(--type-state)"></span> state_object</div>
    <hr style="margin:8px 0;border:0;border-top:1px solid var(--border)" />
    <div class="key"><span style="width:16px;height:2px;background:var(--bad);display:inline-block"></span> supersede</div>
    <div class="key"><span style="width:16px;height:2px;background:var(--accent);display:inline-block"></span> entity overlap</div>
    <div class="key"><span style="width:16px;height:2px;background:#465070;display:inline-block;opacity:0.7"></span> same session</div>
  </div>
  <div class="map-mode-toggle">
    <button id="mode-3d" class="active">3D orbit</button>
    <button id="mode-2d">2D force</button>
  </div>
  <div id="map-tooltip" class="map-tooltip"></div>
  <div id="map-status" style="position:absolute;left:12px;top:12px;background:rgba(20,22,28,0.92);border:1px solid var(--border);border-radius:6px;padding:8px 10px;font-size:12px;color:var(--fg-2)">loading…</div>
</div>
<p class="map-help">Records at their PCA-projected positions (3 principal components of the 384-dim embedding). Distance ≈ semantic similarity. Edge color = supersede / entity-overlap / same-session.</p>
{js}"#,
        n = node_count,
        e = edge_count,
        js = MAP_JS,
    );
    page("map", user_id, &content)
}

// ---------------------------------------------------------------------------
// Curation status + trigger
// ---------------------------------------------------------------------------

pub fn curate_view(
    user_id: &str,
    counts: &[(String, i64)],
    provider_can_distill: bool,
    provider_name: &str,
) -> String {
    let mut rows_html = String::new();
    for (kind, n) in counts {
        rows_html.push_str(&format!(
            r#"<tr><td>{kind}</td><td class="mono">{n}</td></tr>"#,
            kind = type_pill(kind),
            n = n,
        ));
    }
    if counts.is_empty() {
        rows_html.push_str(
            r#"<tr><td colspan="2" class="muted" style="text-align:center;padding:16px">No curated nodes yet. Run a pass below to promote + distill your raw records.</td></tr>"#,
        );
    }

    let distill_note = if provider_can_distill {
        format!(
            r#"<p class="muted" style="font-size:13px">Distillation uses provider <code>{}</code> — LLM-grade fact extraction is available.</p>"#,
            esc(provider_name)
        )
    } else {
        format!(
            r#"<div class="error" style="background:rgba(255,189,85,0.08);border-color:var(--warn);color:var(--warn)">
                Current provider <code>{}</code> does not implement fact distillation. A curation pass still promotes working → episodic records; semantic distillation is skipped with a warning until you set <code>PROVIDER=remote</code> (or <code>=embedded</code>) and supply credentials/model.
            </div>"#,
            esc(provider_name)
        )
    };

    let content = format!(
        r#"<h1>Curate</h1>
<p class="muted">Run an incremental curation pass: promote new records to <code>episodic</code>, refresh grown conversations, cluster and distill the undistilled into <code>semantic</code> facts, and re-derive summaries when anything changed. Raw is never touched, and an unchanged store costs no model work. The destructive wipe-and-re-derive rebuild lives behind the API, not this button.</p>
{distill_note}
<div class="card" style="display:flex;gap:12px;align-items:center">
  <form method="post" action="/admin/api/curate" class="inline">
    <button type="submit">Run curation now</button>
  </form>
  <span class="muted" style="font-size:12px">manual runs scope to <code>{user_id}</code>; the scheduler iterates all users.</span>
</div>
<h2>Curated nodes by kind</h2>
<table>
<thead><tr><th>kind</th><th>count</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>"#,
        user_id = esc(user_id),
    );

    page("curate", user_id, &content)
}

// ---------------------------------------------------------------------------
// Map renderer — vanilla canvas2d with hand-rolled 3D projection.
// No Three.js, no WebGL, no framework. Toggle between 3D orbit and 2D
// force-directed view in the same scene.
// ---------------------------------------------------------------------------

const MAP_JS: &str = r##"<script>
(async () => {
    const canvas = document.getElementById('map-canvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('map-tooltip');
    const statusEl = document.getElementById('map-status');
    const mode3DBtn = document.getElementById('mode-3d');
    const mode2DBtn = document.getElementById('mode-2d');

    const TYPE_COLOR = {
        episodic:     '#6f9eff',
        semantic:     '#5cd07a',
        working:      '#ffbd55',
        document:     '#b8bcc8',
        procedural:   '#c594ff',
        state_object: '#ff8fbf',
    };
    const EDGE_COLOR = {
        supersede: '#ff7a7a',
        entity:    '#6f9eff',
        session:   '#465070',
    };
    const EDGE_ALPHA = { supersede: 0.85, entity: 0.40, session: 0.20 };
    const EDGE_W     = { supersede: 1.8,  entity: 1.2,  session: 0.8 };

    let data;
    try {
        const r = await fetch('/admin/api/map.json', { credentials: 'same-origin' });
        if (!r.ok) throw new Error('fetch ' + r.status);
        data = await r.json();
    } catch (e) {
        canvas.outerHTML = '<p style="padding:24px;color:#ff7a7a">Failed to load map data: ' + e.message + '</p>';
        return;
    }

    const nodes = data.nodes.map(n => ({ ...n, vx: 0, vy: 0, deg: 0 }));
    const byId = new Map(nodes.map(n => [n.id, n]));
    const edges = data.edges
        .map(e => ({ source: byId.get(e.source), target: byId.get(e.target), kind: e.kind, weight: e.weight }))
        .filter(e => e.source && e.target);
    for (const e of edges) { e.source.deg++; e.target.deg++; }

    // ---- View state ----
    let mode = '3d';
    let rotX = -0.35, rotY = 0.55;
    let zoom = 1.0;
    let pan = { x: 0, y: 0 };
    let dragging = false, lastMx = 0, lastMy = 0;
    let hovered = null;

    function resize() {
        const r = canvas.getBoundingClientRect();
        canvas.width  = Math.max(1, Math.floor(r.width  * devicePixelRatio));
        canvas.height = Math.max(1, Math.floor(r.height * devicePixelRatio));
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    }
    window.addEventListener('resize', () => { resize(); render(); });
    resize();

    // ---- 3D projection ----
    // World coords are normalized to [-1, 1] on each axis (PCA-scaled).
    // Camera is fixed at +Z, looking at origin. We rotate the WORLD instead of
    // the camera — equivalent and simpler. Perspective via depth division.
    function project3D(x, y, z) {
        const cy = Math.cos(rotY), sy = Math.sin(rotY);
        const cx = Math.cos(rotX), sx = Math.sin(rotX);
        // rotate around Y (yaw)
        const x1 =  x * cy + z * sy;
        const z1 = -x * sy + z * cy;
        // rotate around X (pitch)
        const y2 = y * cx - z1 * sx;
        const z2 = y * sx + z1 * cx;
        const x2 = x1;

        const w = canvas.width  / devicePixelRatio;
        const h = canvas.height / devicePixelRatio;
        const camZ = 3.2;
        const depth = camZ - z2;
        const scale = (Math.min(w, h) * 0.42 * zoom) * (camZ / Math.max(depth, 0.05));
        return {
            sx: w / 2 + pan.x + x2 * scale,
            sy: h / 2 + pan.y + y2 * scale,
            depth,
            sizeMul: camZ / Math.max(depth, 0.05),
        };
    }

    function project2D(x, y) {
        const w = canvas.width  / devicePixelRatio;
        const h = canvas.height / devicePixelRatio;
        const s = Math.min(w, h) * 0.42 * zoom;
        return { sx: w / 2 + pan.x + x * s, sy: h / 2 + pan.y + y * s, depth: 1.0, sizeMul: 1.0 };
    }

    function projectAll() {
        for (const n of nodes) {
            const p = (mode === '3d') ? project3D(n.x3, n.y3, n.z3) : project2D(n.x, n.y);
            n._sx = p.sx; n._sy = p.sy; n._depth = p.depth; n._sizeMul = p.sizeMul;
            const baseR = 4 + Math.sqrt(n.deg) * 2.4 + (n.importance || 0.5) * 4;
            n._r = Math.max(2.5, baseR * (mode === '3d' ? p.sizeMul * 0.8 : 1.0));
        }
    }

    function fadeForDepth(d) {
        if (mode !== '3d') return 1.0;
        // depth ~ camZ - z; camZ=3.2, z in [-1,1] → depth in [2.2, 4.2].
        // Map to [1.0, 0.45].
        return Math.max(0.45, 1.15 - (d - 2.2) * 0.35);
    }

    function render() {
        const w = canvas.width  / devicePixelRatio;
        const h = canvas.height / devicePixelRatio;
        ctx.clearRect(0, 0, w, h);
        projectAll();

        // Edges first, painted with combined-depth fade so back edges recede.
        for (const e of edges) {
            const avgD = (e.source._depth + e.target._depth) / 2;
            const fade = fadeForDepth(avgD);
            ctx.strokeStyle = EDGE_COLOR[e.kind] || '#888';
            ctx.globalAlpha = (EDGE_ALPHA[e.kind] || 0.4) * fade;
            ctx.lineWidth = (EDGE_W[e.kind] || 1) * fade;
            ctx.beginPath();
            ctx.moveTo(e.source._sx, e.source._sy);
            ctx.lineTo(e.target._sx, e.target._sy);
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;

        // Nodes back-to-front so closer ones occlude.
        const back2front = [...nodes].sort((a, b) => b._depth - a._depth);
        for (const n of back2front) {
            const fade = fadeForDepth(n._depth);
            ctx.fillStyle = TYPE_COLOR[n.type] || '#888';
            ctx.globalAlpha = (n.superseded ? 0.32 : 0.95) * fade;
            ctx.beginPath();
            ctx.arc(n._sx, n._sy, n._r, 0, Math.PI * 2);
            ctx.fill();
            // Outline for separation from background.
            ctx.strokeStyle = '#0d0e12';
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.8 * fade;
            ctx.stroke();

            // Label hub nodes only.
            if ((n.deg >= 2 || n.type === 'state_object') && n.label) {
                const fontSize = Math.max(10, Math.min(14, n._r * 0.55 + 4));
                ctx.font = fontSize + 'px -apple-system, "SF Pro Text", "Segoe UI", sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.globalAlpha = 0.92 * fade;
                ctx.strokeStyle = '#0d0e12';
                ctx.lineWidth = 3.5;
                ctx.strokeText(n.label, n._sx, n._sy - n._r - 4);
                ctx.fillStyle = '#e6e7eb';
                ctx.fillText(n.label, n._sx, n._sy - n._r - 4);
            }
        }
        ctx.globalAlpha = 1.0;

        if (hovered) {
            ctx.strokeStyle = '#e6e7eb';
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            ctx.arc(hovered._sx, hovered._sy, hovered._r + 4, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    function pick(mx, my) {
        // Pick the front-most node within radius.
        const f2b = [...nodes].sort((a, b) => a._depth - b._depth);
        for (const n of f2b) {
            const dx = mx - n._sx, dy = my - n._sy;
            if (dx*dx + dy*dy <= n._r * n._r) return n;
        }
        return null;
    }

    // ---- 2D force-directed sim (only when 2D mode is selected) ----
    function simulate2D(iters) {
        const N = nodes.length;
        const k = Math.sqrt(2.0 / Math.max(N, 1)) * 0.9;
        const repK2 = k * k * 1.2;
        const attrK = 1.0 / k;
        const center = 0.04;
        let temp = 0.08;
        for (let it = 0; it < iters; it++) {
            for (let i = 0; i < N; i++) {
                const a = nodes[i];
                for (let j = i + 1; j < N; j++) {
                    const b = nodes[j];
                    let dx = a.x - b.x, dy = a.y - b.y;
                    let d2 = dx*dx + dy*dy;
                    if (d2 < 1e-6) { dx = (Math.random()-0.5)*0.01; dy = (Math.random()-0.5)*0.01; d2 = dx*dx + dy*dy; }
                    const f = repK2 / d2;
                    a.vx += dx * f; a.vy += dy * f;
                    b.vx -= dx * f; b.vy -= dy * f;
                }
            }
            for (const e of edges) {
                const dx = e.target.x - e.source.x, dy = e.target.y - e.source.y;
                const d = Math.sqrt(dx*dx + dy*dy) || 1e-6;
                const desired = e.kind === 'supersede' ? k * 0.5 : k * 1.0;
                const f = (d - desired) * attrK * e.weight;
                e.source.vx += (dx/d) * f; e.source.vy += (dy/d) * f;
                e.target.vx -= (dx/d) * f; e.target.vy -= (dy/d) * f;
            }
            for (const n of nodes) {
                n.vx -= n.x * center; n.vy -= n.y * center;
                const sp = Math.hypot(n.vx, n.vy);
                const lim = Math.min(sp, temp);
                if (sp > 1e-6) { n.x += n.vx/sp * lim; n.y += n.vy/sp * lim; }
                n.vx *= 0.6; n.vy *= 0.6;
            }
            temp *= 0.985;
        }
    }

    // ---- Interaction ----
    canvas.addEventListener('mousedown', (e) => {
        dragging = true; lastMx = e.clientX; lastMy = e.clientY;
        canvas.classList.add('dragging');
    });
    window.addEventListener('mouseup', () => { dragging = false; canvas.classList.remove('dragging'); });
    window.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        if (dragging) {
            const dx = e.clientX - lastMx, dy = e.clientY - lastMy;
            if (mode === '3d') {
                rotY += dx * 0.008;
                rotX += dy * 0.008;
                rotX = Math.max(-1.3, Math.min(1.3, rotX));
            } else {
                pan.x += dx; pan.y += dy;
            }
            lastMx = e.clientX; lastMy = e.clientY;
            render();
        } else {
            const inside = mx >= 0 && my >= 0 && mx < canvas.width / devicePixelRatio && my < canvas.height / devicePixelRatio;
            const n = inside ? pick(mx, my) : null;
            if (n !== hovered) { hovered = n; render(); }
            if (n) {
                tooltip.style.display = 'block';
                tooltip.style.left = (mx + 14) + 'px';
                tooltip.style.top  = (my + 14) + 'px';
                const label = n.label ? '<strong>' + escapeHtml(n.label) + '</strong><br/>' : '';
                tooltip.innerHTML = label
                    + '<span style="color:var(--fg-2)">' + n.type + ' · ' + n.deg + ' connections'
                    + (n.superseded ? ' · superseded' : '') + '</span><br/>'
                    + escapeHtml((n.content || '').slice(0, 240))
                    + ((n.content || '').length > 240 ? '…' : '');
            } else {
                tooltip.style.display = 'none';
            }
        }
    });
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        zoom *= (e.deltaY < 0 ? 1.15 : 0.87);
        zoom = Math.max(0.2, Math.min(5.0, zoom));
        render();
    }, { passive: false });
    canvas.addEventListener('click', () => {
        if (hovered) window.location = '/admin/memories/' + hovered.id;
    });

    mode3DBtn.addEventListener('click', () => {
        mode = '3d';
        mode3DBtn.classList.add('active'); mode2DBtn.classList.remove('active');
        pan.x = 0; pan.y = 0; zoom = 1.0;
        render();
    });
    mode2DBtn.addEventListener('click', () => {
        mode = '2d';
        mode2DBtn.classList.add('active'); mode3DBtn.classList.remove('active');
        pan.x = 0; pan.y = 0; zoom = 1.0;
        statusEl.textContent = 'running 2D force layout…';
        setTimeout(() => {
            simulate2D(280);
            statusEl.textContent = nodes.length + ' nodes · ' + edges.length + ' edges';
            statusEl.style.color = 'var(--good)';
            render();
        }, 30);
    });

    // ---- Auto-orbit on first load to advertise it's 3D ----
    let autoStart = performance.now();
    function autoOrbit() {
        const t = (performance.now() - autoStart) / 1000;
        if (t > 3.5 || dragging) return;
        rotY += 0.006;
        render();
        requestAnimationFrame(autoOrbit);
    }

    statusEl.textContent = nodes.length + ' nodes · ' + edges.length + ' edges';
    statusEl.style.color = 'var(--good)';
    render();
    requestAnimationFrame(autoOrbit);

    // ---- Live-refresh: poll /admin/api/map.json every 15s, only if the
    //      page is visible and the user isn't actively dragging. New nodes
    //      slot in at their server-computed UMAP positions; existing nodes
    //      keep whatever positions the SGD / user-drag gave them, so the
    //      scene doesn't jolt on every poll. ----
    setInterval(async () => {
        if (document.hidden || dragging) return;
        try {
            const r = await fetch('/admin/api/map.json', { credentials: 'same-origin' });
            if (!r.ok) return;
            const fresh = await r.json();
            const seen = new Set();
            for (const fn of fresh.nodes) {
                seen.add(fn.id);
                const existing = byId.get(fn.id);
                if (existing) {
                    // Patch latest content / superseded flag / labels, leave position.
                    existing.content = fn.content;
                    existing.label = fn.label;
                    existing.superseded = fn.superseded;
                } else {
                    const newNode = { ...fn, vx: 0, vy: 0, deg: 0 };
                    nodes.push(newNode);
                    byId.set(fn.id, newNode);
                }
            }
            // Remove deleted nodes.
            for (let i = nodes.length - 1; i >= 0; i--) {
                if (!seen.has(nodes[i].id)) {
                    byId.delete(nodes[i].id);
                    nodes.splice(i, 1);
                }
            }
            // Rebuild edges array — cheap, edges are small.
            edges.length = 0;
            for (const e of fresh.edges) {
                const a = byId.get(e.source), b = byId.get(e.target);
                if (a && b) edges.push({ source: a, target: b, kind: e.kind, weight: e.weight });
            }
            // Recompute degrees.
            for (const n of nodes) n.deg = 0;
            for (const e of edges) { e.source.deg++; e.target.deg++; }
            statusEl.textContent = nodes.length + ' nodes · ' + edges.length + ' edges · live';
            render();
        } catch (e) {
            // Network blip — silent, retry next tick.
        }
    }, 15000);

    function escapeHtml(s) {
        return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
})();
</script>"##;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn type_pill(t: &str) -> String {
    let cls = format!("pill t-{}", t);
    format!(r#"<span class="{cls}">{t}</span>"#)
}

pub fn short_id(id: Uuid) -> String {
    id.to_string()[..8].to_string()
}

/// A timestamp the browser can localise. The server only knows UTC, so it emits
/// the machine-readable instant in `datetime` and a UTC-labelled fallback as the
/// text — `LOCAL_TIME_JS` rewrites the text to the viewer's zone. Without the
/// label an unconverted UTC reading is silently wrong by the viewer's offset,
/// which is exactly how a 14:52 event came to read as 18:52.
pub fn format_when(t: DateTime<Utc>) -> String {
    format!(
        r#"<time datetime="{}" class="ts">{} UTC</time>"#,
        t.to_rfc3339(),
        t.format("%Y-%m-%d %H:%M")
    )
}

pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare wall-clock string is ambiguous: the viewer reads UTC as local and
    /// is wrong by their offset. The rendered timestamp must carry the instant
    /// machine-readably AND label the fallback text.
    #[test]
    fn format_when_is_machine_readable_and_labelled() {
        let t = DateTime::parse_from_rfc3339("2026-07-25T18:52:26Z")
            .unwrap()
            .with_timezone(&Utc);
        let out = format_when(t);

        assert!(out.contains(r#"datetime="2026-07-25T18:52:26+00:00""#));
        assert!(out.contains("2026-07-25 18:52 UTC"));
        assert!(out.contains(r#"class="ts""#));
    }

    #[test]
    fn page_ships_the_localiser() {
        let html = page("records", "alice", "<p>x</p>");
        assert!(html.contains("time.ts"));
        assert!(html.contains("toLocaleString"));
    }
}

// ---------------------------------------------------------------------------
// Playground — the dynamic-RAG loop, made visible.
// ---------------------------------------------------------------------------

/// Stand in for a host: send a turn, watch what was retrieved, see the exact
/// prompt that would go to a model, and see what got written back.
///
/// Chat-first on purpose — you judge a memory system by whether the reply reads
/// like it remembered, and only then by asking why. Settings are server-side
/// and per-operator, so they survive a different browser, a different origin,
/// or a rebuilt machine.
// ---------------------------------------------------------------------------
// Settings — the runtime provider control surface.
// ---------------------------------------------------------------------------

/// Everything the settings page renders: the stored overrides, the effective
/// config they resolve to over the environment, and what is actually live.
pub struct SettingsInfo {
    pub stored: crate::settings::SystemSettings,
    pub effective_provider: &'static str,
    pub effective_backend: String,
    pub effective_api_base: Option<String>,
    pub effective_extract_model: String,
    pub effective_distill_model: String,
    pub effective_extract_timeout_ms: u32,
    pub effective_distill_timeout_ms: u32,
    pub live_provider: &'static str,
    pub live_models: Option<(String, String)>,
    pub can_distill: bool,
    pub env_has_key: bool,
}

/// Static so the script needs no brace-escaping; every dynamic value arrives
/// through the JSON data island rendered next to it.
const SETTINGS_JS: &str = r##"<script>
const $ = id => document.getElementById(id);
const DATA = JSON.parse($('settings-data').textContent);

// Prefill: a field shows its stored override; empty means "inherit", and the
// placeholder shows what is inherited so blank is never a mystery.
$('st-provider').value = DATA.stored.provider || '';
$('st-backend').value = DATA.stored.remote_backend || '';
$('st-base').value = DATA.stored.api_base || '';
$('st-base').placeholder = (DATA.effective.api_base || 'backend default') + ' — inherited';
$('st-extract').value = DATA.stored.extract_model || '';
$('st-extract').placeholder = DATA.effective.extract_model + ' — inherited';
$('st-distill').value = DATA.stored.distill_model || '';
$('st-distill').placeholder = DATA.effective.distill_model + ' — inherited';
$('st-extract-t').value = DATA.stored.extract_timeout_ms ?? '';
$('st-extract-t').placeholder = DATA.effective.extract_timeout_ms + ' — inherited';
$('st-distill-t').value = DATA.stored.distill_timeout_ms ?? '';
$('st-distill-t').placeholder = DATA.effective.distill_timeout_ms + ' — inherited';

function currentBase() {
  return $('st-base').value.trim() || DATA.effective.api_base || '';
}
function collect() {
  const v = id => $(id).value.trim() || null;
  const n = id => { const x = parseInt($(id).value, 10); return Number.isFinite(x) ? x : null; };
  return {
    provider: v('st-provider'), remote_backend: v('st-backend'), api_base: v('st-base'),
    extract_model: v('st-extract'), distill_model: v('st-distill'),
    extract_timeout_ms: n('st-extract-t'), distill_timeout_ms: n('st-distill-t'),
  };
}
async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.text()).slice(0, 300));
  return res.json();
}

$('st-load-models').addEventListener('click', async e => {
  e.preventDefault();
  const base = currentBase();
  if (!base) { $('st-models-status').textContent = 'set a base URL first'; return; }
  $('st-models-status').textContent = 'asking the endpoint…';
  try {
    const { models } = await post('/admin/api/settings/models', { base });
    const dl = $('st-model-list');
    dl.replaceChildren(...models.map(m => Object.assign(document.createElement('option'), { value: m })));
    $('st-models-status').textContent = models.length + ' model(s) served — the fields now autocomplete from them';
  } catch (err) {
    $('st-models-status').textContent = 'failed: ' + err.message;
  }
});

$('st-test').addEventListener('click', async e => {
  e.preventDefault();
  $('st-test-out').textContent = 'running one real extraction…';
  try {
    const r = await post('/admin/api/settings/test', collect());
    $('st-test-out').textContent = r.ok
      ? 'ok — ' + r.model + ' answered in ' + r.latency_ms + ' ms\n\n'
        + JSON.stringify(r.extraction, null, 2)
      : 'FAILED after ' + r.latency_ms + ' ms — ' + r.error;
  } catch (err) {
    $('st-test-out').textContent = 'FAILED — ' + err.message;
  }
});

$('st-save').addEventListener('click', async e => {
  e.preventDefault();
  $('st-save-status').textContent = 'saving + applying…';
  try {
    const r = await post('/admin/api/settings', collect());
    $('st-save-status').textContent = r.warning
      ? 'saved, but: ' + r.warning
      : 'applied — pipeline now runs ' + r.applied_provider
        + (r.applied_models ? ' (' + r.applied_models[0] + ')' : '');
    if (!r.warning) setTimeout(() => location.reload(), 900);
  } catch (err) {
    $('st-save-status').textContent = 'failed: ' + err.message;
  }
});

// Load the model list on open when a base URL is known — the dropdown being
// pre-populated is the point of the page.
if (currentBase()) $('st-load-models').click();
</script>"##;

pub fn settings_view(user_id: &str, info: &SettingsInfo) -> String {
    let live_pill = if info.can_distill {
        format!(
            r#"<span class="pill" style="color:var(--good)">live: {}</span>"#,
            esc(info.live_provider)
        )
    } else {
        format!(
            r#"<span class="pill" style="color:var(--warn)">live: {} — no semantic facts</span>"#,
            esc(info.live_provider)
        )
    };
    let live_models = match &info.live_models {
        Some((e, d)) if e == d => format!("model <code>{}</code>", esc(e)),
        Some((e, d)) => format!(
            "extract <code>{}</code> · distill <code>{}</code>",
            esc(e),
            esc(d)
        ),
        None => "no model (provider runs without one)".to_string(),
    };
    let key_note = if info.env_has_key {
        "an API key is present in the server environment and is sent to the endpoint"
    } else {
        "no API key in the server environment — fine for local endpoints; hosted \
         backends need one set there (keys are never stored in the database)"
    };
    // `<` is escaped so no stored value can close the JSON script element and
    // start its own — the one break-out this embedding shape allows.
    let data = serde_json::json!({
        "stored": info.stored,
        "effective": {
            "provider": info.effective_provider,
            "backend": info.effective_backend,
            "api_base": info.effective_api_base,
            "extract_model": info.effective_extract_model,
            "distill_model": info.effective_distill_model,
            "extract_timeout_ms": info.effective_extract_timeout_ms,
            "distill_timeout_ms": info.effective_distill_timeout_ms,
        },
    })
    .to_string()
    .replace('<', "\\u003c");

    let content = format!(
        r#"<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
  <h1 style="margin-bottom:4px">Settings</h1>
  {live_pill}
</div>
<p class="muted" style="margin-top:0;font-size:13px">
  The extraction + distillation provider the whole pipeline runs on — every ingest and
  curation pass, server-wide. The environment seeds these values; anything saved here wins
  over it and applies immediately, no restart. Currently {live_models}.
</p>

<div class="card">
  <div class="row" style="gap:8px;flex-wrap:wrap">
    <label style="flex:0 0 160px">provider
      <select id="st-provider" style="width:100%">
        <option value="">inherit ({effective_provider})</option>
        <option value="heuristic">heuristic — in-process, no model</option>
        <option value="remote">remote — HTTP endpoint</option>
      </select>
    </label>
    <label style="flex:0 0 200px">backend
      <select id="st-backend" style="width:100%">
        <option value="">inherit ({effective_backend})</option>
        <option value="openai">openai-compatible</option>
        <option value="anthropic">anthropic</option>
        <option value="openrouter">openrouter</option>
      </select>
    </label>
    <label style="flex:2;min-width:260px">base URL
      <input id="st-base" style="width:100%;box-sizing:border-box" />
    </label>
  </div>
  <div class="row" style="gap:8px;flex-wrap:wrap;margin-top:10px;align-items:flex-end">
    <label style="flex:1;min-width:200px">extract model
      <input id="st-extract" list="st-model-list" style="width:100%;box-sizing:border-box" />
    </label>
    <label style="flex:1;min-width:200px">distill model
      <input id="st-distill" list="st-model-list" style="width:100%;box-sizing:border-box" />
    </label>
    <button id="st-load-models" title="ask the endpoint what it serves">Load models</button>
  </div>
  <datalist id="st-model-list"></datalist>
  <div class="muted" id="st-models-status" style="font-size:12px;margin-top:4px"></div>
  <div class="row" style="gap:8px;flex-wrap:wrap;margin-top:10px">
    <label style="flex:0 0 180px">extract timeout (ms)
      <input id="st-extract-t" style="width:100%;box-sizing:border-box" />
    </label>
    <label style="flex:0 0 180px">distill timeout (ms)
      <input id="st-distill-t" style="width:100%;box-sizing:border-box" />
    </label>
  </div>
  <div class="row" style="align-items:center;gap:10px;margin-top:14px">
    <button id="st-test">Test extraction</button>
    <button id="st-save">Save &amp; apply</button>
    <span id="st-save-status" class="muted" style="font-size:12px"></span>
  </div>
  <p class="muted" style="font-size:12px;margin-bottom:0">
    Model fields autocomplete from what the endpoint actually serves — load the list and
    pick from it. Leave a field blank to inherit the environment value shown in its
    placeholder. Note: {key_note}.
  </p>
</div>

<div class="card" style="margin-top:16px">
  <h2 style="margin-top:0">Test result</h2>
  <pre id="st-test-out" class="mono" style="white-space:pre-wrap;margin:0;min-height:40px">Run a test to see one real extraction with the draft settings — nothing is saved by testing.</pre>
</div>

<script id="settings-data" type="application/json">{data}</script>
{SETTINGS_JS}"#,
        effective_provider = esc(info.effective_provider),
        effective_backend = esc(&info.effective_backend),
        data = data,
    );

    page("settings", user_id, &content)
}

pub fn playground_view(
    user_id: &str,
    can_distill: bool,
    settings: &super::playground::Settings,
    system_model: Option<String>,
) -> String {
    let distill_note = if can_distill {
        r#"<span class="pill" style="color:var(--good)">distillation on</span>"#
    } else {
        r#"<span class="pill" style="color:var(--warn)">heuristic provider — no semantic facts yet</span>"#
    };
    let configured = settings.base_url.is_some() && settings.model.is_some();
    // Three states, in order of preference: a sandbox override (silent), no
    // override but a system provider to inherit (a note, not a warning), and
    // nothing anywhere (the original loud banner — a turn will get no reply).
    let warn_banner = if configured {
        String::new()
    } else if let Some(m) = &system_model {
        format!(
            r#"<p class="muted" style="margin:12px 0;font-size:13px">
  Turns use the system provider — <code>{}</code>, the model that distills your
  memories. Set a base URL and model in ⚙ settings to probe a different one.
</p>"#,
            esc(m)
        )
    } else {
        r#"<div class="error" style="margin:12px 0">
  <strong>No model configured.</strong> Turns will retrieve and write, but nothing will reply.
  Set a base URL and model below, or configure the system provider on the
  <a href="/admin/settings">Settings</a> page — the playground inherits it.
</div>"#
            .to_string()
    };
    let val = |o: &Option<String>| esc(o.as_deref().unwrap_or(""));
    let content = format!(
        r##"<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
  <h1 style="margin-bottom:4px">Playground</h1>
  {distill_note}
  <span class="spacer" style="flex:1"></span>
  <a href="#" id="pg-settings-toggle" class="muted" style="font-size:13px">⚙ settings</a>
</div>
<p class="muted" style="margin-top:0;font-size:13px">
  Retrieval and the write use the same seams ritsu does — what happens here is what a real host gets.
  Sandboxed by default: reads and writes stay in the <code>playground</code> scope. Tick
  <em>include real memories</em> to also probe the real store — writes never leave the sandbox either way.
</p>
{warn_banner}

<div class="card" id="pg-settings" style="display:{settings_display}">
  <div class="row" style="gap:8px;flex-wrap:wrap">
    <input id="pg-base" value="{base}" placeholder="http://127.0.0.1:1234/v1 — base URL" style="flex:2;min-width:240px" />
    <input id="pg-model" value="{model}" placeholder="model name (required)" style="flex:1;min-width:180px" />
    <input id="pg-limit" value="{limit}" placeholder="memories (12)" style="flex:0 0 120px" />
  </div>
  <input id="pg-key" type="password" placeholder="api key — optional, stays in this browser" style="width:100%;margin-top:8px;box-sizing:border-box" />
  <label class="muted" style="font-size:12px;display:block;margin-top:10px">
    System prompt — how retrieved memories get framed. The biggest lever on whether the model uses them.
  </label>
  <textarea id="pg-sys" rows="3" style="width:100%;box-sizing:border-box;margin-top:4px">{sys}</textarea>
  <label class="muted" style="font-size:12px;display:block;margin-top:12px">
    Seed the sandbox — one memory per line; an optional <code>YYYY-MM-DD |</code> prefix backdates it
    (that's how you test recency and newest-wins distillation).
  </label>
  <textarea id="pg-seed" rows="3" style="width:100%;box-sizing:border-box;margin-top:4px"
    placeholder="2025-11-02 | switched the home backup drive to the 4TB one&#10;prefers coffee brewed at 93C"></textarea>
  <div class="row" style="align-items:center;gap:10px;margin-top:6px">
    <button id="pg-seed-btn">Seed memories</button>
    <span id="pg-seed-status" class="muted" style="font-size:12px"></span>
  </div>
  <div class="row" style="align-items:center;gap:10px;margin-top:8px">
    <button id="pg-save">Save settings</button>
    <button id="pg-sys-reset" class="muted">reset prompt</button>
    <span id="pg-save-status" class="muted" style="font-size:12px"></span>
  </div>
  <p class="muted" style="font-size:12px;margin-bottom:0">
    Saved on the server for <code>{user_id}</code>. Any OpenAI-compatible endpoint —
    LM Studio, Ollama, LiteLLM, OpenRouter. The API key is the one exception: it stays
    in this browser rather than in the database.
  </p>
</div>

<div class="row" style="align-items:stretch;gap:16px;margin-top:16px">
  <div class="card" style="flex:1.15;min-width:340px;display:flex;flex-direction:column">
    <div class="muted" style="font-size:12px;margin-bottom:8px">
      <code id="pg-container"></code> · <a href="#" id="pg-new">new conversation</a>
    </div>
    <div id="pg-log" style="flex:1;min-height:46vh;max-height:60vh;overflow:auto;padding-right:4px"></div>
    <div style="margin-top:10px">
      <textarea id="pg-msg" rows="3" placeholder="Ask something it should remember. ⌘↵ to send."
        style="width:100%;box-sizing:border-box"></textarea>
      <div class="row" style="align-items:center;gap:10px;margin-top:6px">
        <button id="pg-send">Send</button>
        <label class="muted" style="font-size:12px;display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="pg-real" /> include real memories
        </label>
        <span id="pg-status" class="muted" style="font-size:12px"></span>
      </div>
    </div>
  </div>

  <div class="card" style="flex:1;min-width:320px;overflow:auto;max-height:78vh">
    <h2 style="margin-top:0">Diagnostics</h2>
    <div id="pg-trace"><p class="muted">Send a turn to see what it retrieved and why.</p></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const approxTokens = s => Math.round(s.length / 4);
const DEFAULT_SYS = 'You are a helpful assistant with persistent memory. Answer in a few sentences unless asked for detail — most of the wait a user experiences is output length.';

$('pg-settings-toggle').addEventListener('click', e => {{
  e.preventDefault();
  const el = $('pg-settings');
  el.style.display = el.style.display === 'none' ? '' : 'none';
}});
$('pg-sys-reset').addEventListener('click', e => {{ e.preventDefault(); $('pg-sys').value = DEFAULT_SYS; }});

// The API key is the only browser-held value; everything else is server-side.
$('pg-key').value = localStorage.getItem('fb_pg_key') || '';
$('pg-key').addEventListener('change', () => localStorage.setItem('fb_pg_key', $('pg-key').value));

$('pg-real').checked = localStorage.getItem('fb_pg_real') === '1';
$('pg-real').addEventListener('change', () => localStorage.setItem('fb_pg_real', $('pg-real').checked ? '1' : '0'));

$('pg-seed-btn').addEventListener('click', async e => {{
  e.preventDefault();
  const text = $('pg-seed').value;
  if (!text.trim()) {{ $('pg-seed-status').textContent = 'nothing to seed'; return; }}
  $('pg-seed-status').textContent = 'seeding…';
  try {{
    const res = await fetch('/admin/api/playground/seed', {{
      method: 'POST', headers: {{ 'content-type': 'application/json' }}, body: JSON.stringify({{ text }}),
    }});
    if (!res.ok) throw new Error((await res.text()).slice(0, 200));
    const d = await res.json();
    $('pg-seed-status').textContent = d.seeded + ' memories seeded — ask about them, or Distill now';
    $('pg-seed').value = '';
  }} catch (err) {{
    $('pg-seed-status').textContent = 'failed: ' + err.message;
  }}
}});

$('pg-save').addEventListener('click', async e => {{
  e.preventDefault();
  const limit = parseInt($('pg-limit').value, 10);
  const body = {{
    base_url: $('pg-base').value, model: $('pg-model').value,
    system_prompt: $('pg-sys').value,
    context_limit: Number.isFinite(limit) ? limit : null,
  }};
  $('pg-save-status').textContent = 'saving…';
  try {{
    const res = await fetch('/admin/api/playground/settings', {{
      method: 'POST', headers: {{ 'content-type': 'application/json' }}, body: JSON.stringify(body),
    }});
    if (!res.ok) throw new Error((await res.text()).slice(0, 200));
    const saved = await res.json();
    $('pg-save-status').textContent = (saved.base_url && saved.model)
      ? 'saved' : 'saved — but base URL AND model are both needed to get a reply';
    if (saved.base_url && saved.model) document.querySelector('.error')?.remove();
  }} catch (err) {{
    $('pg-save-status').textContent = 'failed: ' + err.message;
  }}
}});

const newContainer = () => 'playground:' + Math.random().toString(36).slice(2, 10);
let container = sessionStorage.getItem('fb_pg_container') || newContainer();
sessionStorage.setItem('fb_pg_container', container);
$('pg-container').textContent = container;
$('pg-new').addEventListener('click', e => {{
  e.preventDefault();
  container = newContainer();
  sessionStorage.setItem('fb_pg_container', container);
  $('pg-container').textContent = container;
  $('pg-log').innerHTML = '';
  $('pg-trace').innerHTML = '<p class="muted">New conversation. Nothing retrieved yet.</p>';
}});

function bubble(role, text) {{
  const mine = role === 'user';
  $('pg-log').insertAdjacentHTML('beforeend',
    `<div style="display:flex;justify-content:${{mine ? 'flex-end' : 'flex-start'}};margin-bottom:10px">
       <div style="max-width:82%;padding:9px 12px;border-radius:12px;
                   background:${{mine ? 'var(--accent,#2a4d69)' : 'var(--panel2,#1e1e24)'}};
                   white-space:pre-wrap;word-break:break-word"></div>
     </div>`);
  const el = $('pg-log').lastElementChild.firstElementChild;
  el.textContent = text;   // textContent, so streamed chunks can't inject HTML
  $('pg-log').scrollTop = $('pg-log').scrollHeight;
  return el;
}}

const stat = (l, v) => `<div class="stat" style="padding:8px"><div class="label" style="font-size:10px">${{l}}</div>
  <div class="value" style="font-size:17px">${{v}}</div></div>`;
const section = (t, b, open) => `<details ${{open ? 'open' : ''}} style="margin-top:12px">
  <summary style="cursor:pointer;font-weight:600">${{t}}</summary><div style="margin-top:8px">${{b}}</div></details>`;

function renderTrace(t, retrievalMs) {{
  const ctx = t.prompt.length > 2 ? t.prompt[1].content : '';
  const parts = ['<div class="stat-grid" id="pg-stats" style="grid-template-columns:repeat(auto-fit,minmax(84px,1fr))">'];
  parts.push(stat('retrieved', t.retrieved.length));
  parts.push(stat('context', '~' + approxTokens(ctx) + ' tok'));
  parts.push(stat('retrieval', (retrievalMs / 1000).toFixed(2) + 's'));
  parts.push('</div>');

  if (t.model) parts.push(`<p class="muted mono" style="font-size:11px;margin:4px 0">model: ${{esc(t.model)}}${{t.model_inherited ? ' · system default' : ''}}</p>`);
  if (t.degraded) parts.push(`<p class="pill" style="color:var(--warn)">degraded: ${{esc(t.warning || '')}}</p>`);
  parts.push('<div id="pg-llm-note"></div>');
  if (t.llm_error) llmNoteHtml = noReply(t.llm_error);

  let mem = '';
  if (!t.retrieved.length) {{
    mem = '<p class="muted">Nothing matched. On a near-empty store that is correct, not a bug.</p>';
  }} else {{
    mem = '<ol style="padding-left:18px;margin:0">';
    for (const r of t.retrieved) {{
      mem += `<li style="margin-bottom:10px">
        <div class="muted mono" style="font-size:11px">${{r.sandbox ? 'sandbox' : 'real'}} · ${{esc(r.type)}} · ${{esc(r.source)}}
          · <time class="ts" datetime="${{esc(r.event_time)}}">${{esc(r.event_time)}}</time></div>
        <div style="white-space:pre-wrap;font-size:13px">${{esc(r.content.slice(0, 320))}}${{r.content.length > 320 ? '…' : ''}}</div>
      </li>`;
    }}
    mem += '</ol>';
  }}
  parts.push(section(`Retrieved memories (${{t.retrieved.length}})`, mem, true));

  let pr = '';
  for (const m of t.prompt) {{
    pr += `<div style="margin-bottom:8px"><div class="muted mono" style="font-size:11px">${{esc(m.role)}}
      · ~${{approxTokens(m.content)}} tok</div>
      <pre style="white-space:pre-wrap;margin:2px 0;font-size:12px">${{esc(m.content)}}</pre></div>`;
  }}
  parts.push(section('Prompt sent — exactly what the model saw', pr, false));

  parts.push(section('Extraction — what the pipeline understood', '<div id="pg-extraction"><p class="muted">extracting…</p></div>', true));

  parts.push(section('Written to raw', `<p class="mono" id="pg-written" style="font-size:12px"></p>
    <p class="muted" style="font-size:12px">Run <a href="/admin/curate">Curate</a> to derive episodes from these turns.</p>`, false));

  parts.push(section('Distillation', `<button id="pg-distill">Distill now</button>
    <span class="muted" style="font-size:12px"> runs a real curation pass, then shows what this conversation produced</span>
    <div id="pg-distill-out" style="margin-top:8px"></div>`, false));

  $('pg-trace').innerHTML = parts.join('');
  if (llmNoteHtml) $('pg-llm-note').innerHTML = llmNoteHtml;
  if (lastExtraction) renderExtraction(lastExtraction);
  patchWritten(t.written);
  for (const el of document.querySelectorAll('#pg-trace time.ts')) {{
    const d = new Date(el.getAttribute('datetime'));
    if (!isNaN(d)) el.textContent = d.toLocaleString();
  }}
}}

let llmNoteHtml = '';
const noReply = m => `<div class="error" style="margin:10px 0"><strong>No reply.</strong>
  <div style="font-size:12px;margin-top:4px;white-space:pre-wrap">${{esc(m)}}</div></div>`;

// The extraction event can land before or after the trace rebuilds the panel;
// keep the latest and render whenever both exist.
let lastExtraction = null;
function renderExtraction(x) {{
  const el = $('pg-extraction');
  if (!el) return;
  if (x.error) {{ el.innerHTML = `<p class="muted">extraction failed: ${{esc(x.error)}}</p>`; return; }}
  const ent = (x.entities || []).map(e => `<code>${{esc(e)}}</code>`).join(' ');
  el.innerHTML = `<div class="mono" style="font-size:12px;line-height:1.9">
    topic: ${{esc(x.topic || '—')}} · intent: ${{esc(x.intent || '—')}} · operation: ${{esc(x.operation || '—')}}
    · mode: ${{esc(x.mode || '—')}} · confidence: ${{x.confidence != null ? Number(x.confidence).toFixed(2) : '—'}}<br/>
    entities: ${{ent || '—'}}</div>`;
}}

// Delegated: the button is recreated with every trace render.
$('pg-trace').addEventListener('click', async e => {{
  if (!e.target || e.target.id !== 'pg-distill') return;
  e.preventDefault();
  const out = $('pg-distill-out');
  e.target.disabled = true;
  out.innerHTML = '<p class="muted">running a curation pass…</p>';
  try {{
    const res = await fetch('/admin/api/playground/distill', {{
      method: 'POST', headers: {{ 'content-type': 'application/json' }},
      body: JSON.stringify({{ container_id: container }}),
    }});
    if (!res.ok) throw new Error((await res.text()).slice(0, 300));
    const d = await res.json();
    if (d.locked_out) {{ out.innerHTML = '<p class="muted">a curation pass is already running — try again in a moment</p>'; return; }}
    if (d.skipped_distill) {{ out.innerHTML = `<p class="muted">provider ${{esc(d.provider)}} cannot distill — configure a remote provider in Settings</p>`; return; }}
    let h = `<p class="muted mono" style="font-size:11px">promoted ${{d.promoted}} · refreshed ${{d.refreshed}} · distilled ${{d.distilled}}</p>`;
    if (!d.facts.length) {{
      h += '<p class="muted">No facts trace back to this conversation yet. A lone conversation stays a singleton until related evidence exists — revisit the topic in another conversation and distill again.</p>';
    }} else {{
      h += '<ul style="padding-left:18px;margin:0">' + d.facts.map(f =>
        `<li style="margin-bottom:6px;font-size:13px">${{esc(f.content)}}</li>`).join('') + '</ul>';
    }}
    out.innerHTML = h;
  }} catch (err) {{
    out.innerHTML = `<p class="muted">failed: ${{esc(err.message)}}</p>`;
  }} finally {{
    e.target.disabled = false;
  }}
}});

function patchWritten(ids) {{
  const el = $('pg-written');
  if (el) el.innerHTML = ids.map(id => `<a href="/admin/records/${{id.slice ? id : ''}}">${{String(id).slice(0,8)}}</a>`).join(' · ') || '—';
}}

/// After `done`: fold the model's numbers into the stat row.
function patchStats(stats, totalMs) {{
  const grid = $('pg-stats');
  if (!grid) return;
  let extra = stat('total', (totalMs / 1000).toFixed(1) + 's');
  if (stats) {{
    if (stats.prompt_tokens != null) extra += stat('prompt tok', stats.prompt_tokens);
    if (stats.completion_tokens != null) extra += stat('output tok', stats.completion_tokens);
    extra += stat('model', (stats.latency_ms / 1000).toFixed(1) + 's');
  }}
  grid.insertAdjacentHTML('beforeend', extra);
}}

let timer = null;
async function send() {{
  const msg = $('pg-msg').value.trim();
  if (!msg) return;
  $('pg-send').disabled = true;
  bubble('user', msg);
  $('pg-msg').value = '';
  llmNoteHtml = '';
  lastExtraction = null;

  const t0 = performance.now();
  let gotFirstDelta = false;
  timer = setInterval(() => {{
    if (!gotFirstDelta)
      $('pg-status').textContent = 'waiting for model… ' + ((performance.now() - t0) / 1000).toFixed(0) + 's';
  }}, 250);

  let assistantEl = null, assistantText = '';

  try {{
    const res = await fetch('/admin/api/playground/turn', {{
      method: 'POST', headers: {{ 'content-type': 'application/json' }},
      body: JSON.stringify({{ message: msg, container_id: container, api_key: $('pg-key').value || null, include_real: $('pg-real').checked }}),
    }});
    if (!res.ok) throw new Error((await res.text()).slice(0, 300));

    // Consume the SSE stream: trace (instant) → delta* → done.
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {{
      const {{ done, value }} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{ stream: true }});
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {{
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        let ev = 'message', data = '';
        for (const line of frame.split('\n')) {{
          if (line.startsWith('event:')) ev = line.slice(6).trim();
          else if (line.startsWith('data:')) data += line.slice(5).trim();
        }}
        if (!data) continue;
        if (ev === 'trace') {{
          renderTrace(JSON.parse(data), performance.now() - t0);
        }} else if (ev === 'extraction') {{
          lastExtraction = JSON.parse(data);
          renderExtraction(lastExtraction);
        }} else if (ev === 'delta') {{
          if (!assistantEl) {{
            gotFirstDelta = true;
            $('pg-status').textContent = '';
            assistantEl = bubble('assistant', '');
          }}
          assistantText += JSON.parse(data).t;
          assistantEl.textContent = assistantText;
          $('pg-log').scrollTop = $('pg-log').scrollHeight;
        }} else if (ev === 'done') {{
          const d = JSON.parse(data);
          patchStats(d.stats, performance.now() - t0);
          patchWritten(d.written);
          if (d.llm_error) $('pg-llm-note').innerHTML = noReply(d.llm_error);
        }} else if (ev === 'error') {{
          $('pg-status').textContent = 'failed: ' + data;
        }}
      }}
    }}
    clearInterval(timer);
    if (!assistantText) $('pg-status').textContent = 'written — see Diagnostics';
    else $('pg-status').textContent = '';
  }} catch (e) {{
    clearInterval(timer);
    $('pg-status').textContent = 'failed: ' + e.message;
  }} finally {{
    $('pg-send').disabled = false;
    $('pg-msg').focus();
  }}
}}
$('pg-send').addEventListener('click', send);
$('pg-msg').addEventListener('keydown', e => {{ if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send(); }});
</script>"##,
        settings_display = if configured { "none" } else { "" },
        base = val(&settings.base_url),
        model = val(&settings.model),
        sys = val(&settings.system_prompt),
        limit = settings
            .context_limit
            .map(|n| n.to_string())
            .unwrap_or_default(),
    );
    page("playground", user_id, &content)
}
