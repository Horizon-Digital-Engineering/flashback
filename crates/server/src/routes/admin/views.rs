//! HTML rendering helpers. Hand-rolled `format!` calls — no templating crate.

use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use serde_json::Value;
use uuid::Uuid;

use crate::models::{CoreMemoryRow, MemoryView};

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
</body>
</html>"#
    )
}

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
  <span class="spacer"></span>
  <span class="user">{user_id}</span>
  <a href="/admin/logout">logout</a>
</nav>"#,
        a = item("dashboard", "/admin", "Dashboard"),
        b = item("memories", "/admin/memories", "Memories"),
        c = item("state", "/admin/state", "State"),
        d = item("catalog", "/admin/catalog", "Catalog"),
        e = item("proposals", "/admin/proposals", "Proposals"),
        f = item("map", "/admin/map", "Map"),
        g = item("consolidate", "/admin/consolidate", "Consolidate"),
        h = item("tokens", "/admin/tokens", "Tokens"),
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
    pub memories_total: i64,
    pub memories_terminal: i64,
    pub state_objects: i64,
    pub tokens_active: i64,
    pub provider: String,
    pub embedder_model: String,
    pub embedder_dim: usize,
}

pub fn dashboard(user_id: &str, stats: DashboardStats, recent: &[MemoryView]) -> String {
    let mut recent_html =
        String::from(r#"<div class="card"><h2 style="margin-top:0">Recent memories</h2>"#);
    if recent.is_empty() {
        recent_html.push_str(
            r#"<p class="muted">No memories yet. Ingest some via POST /memory/ingest.</p>"#,
        );
    } else {
        recent_html.push_str(
            "<table><thead><tr><th>type</th><th>content</th><th>created</th></tr></thead><tbody>",
        );
        for m in recent {
            recent_html.push_str(&format!(
                r#"<tr><td>{ty}</td><td><a href="/admin/memories/{id}"><div class="content-preview">{content}</div></a></td><td class="mono muted">{when}</td></tr>"#,
                ty = type_pill(&m.type_),
                id = m.id,
                content = esc(&m.content),
                when = format_when(m.created_at),
            ));
        }
        recent_html.push_str("</tbody></table>");
    }
    recent_html.push_str("</div>");

    let content = format!(
        r#"<h1>Dashboard</h1>
<div class="stat-grid">
  <div class="stat"><div class="label">Memories (terminal)</div><div class="value">{term}</div></div>
  <div class="stat"><div class="label">Memories (all)</div><div class="value">{total}</div></div>
  <div class="stat"><div class="label">State objects</div><div class="value">{state}</div></div>
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
    <p><a href="/admin/memories">Browse memories →</a></p>
    <p><a href="/admin/map">Embedding map →</a></p>
    <p><a href="/admin/state">State objects →</a></p>
    <p><a href="/admin/tokens">Tokens →</a></p>
  </div>
</div>
{recent}"#,
        term = stats.memories_terminal,
        total = stats.memories_total,
        state = stats.state_objects,
        tok = stats.tokens_active,
        embed = esc(&stats.embedder_model),
        dim = stats.embedder_dim,
        prov = esc(&stats.provider),
        recent = recent_html,
    );

    page("dashboard", user_id, &content)
}

// ---------------------------------------------------------------------------
// Memories list
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct MemoriesFilter {
    pub r#type: Option<String>,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub include_superseded: bool,
}

pub fn memories_list(
    user_id: &str,
    filter: &MemoriesFilter,
    memories: &[MemoryView],
    total: i64,
) -> String {
    let q = build_query_string(filter);
    let mut filter_form = String::from(
        r#"<form method="get" action="/admin/memories" class="card" style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end">"#,
    );
    filter_form.push_str(&format!(
        r#"<div><label class="muted">Type</label><br />
            <select name="type" style="background:var(--bg-2);color:var(--fg-0);border:1px solid var(--border);border-radius:6px;padding:8px;min-width:140px">
              <option value="" {sel_any}>any</option>
              {opts}
            </select></div>"#,
        sel_any = if filter.r#type.is_none() { "selected" } else { "" },
        opts = ["working", "episodic", "semantic", "document", "procedural", "state_object"]
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
            <input type="text" name="session_id" value="{}" placeholder="any" style="min-width:160px" /></div>"#,
        esc(filter.session_id.as_deref().unwrap_or(""))
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
  <th>topic / content</th>
  <th>entities</th>
  <th>created</th>
  <th></th>
</tr></thead>
<tbody>"#,
    );
    for m in memories {
        let topic = m
            .state_key
            .as_deref()
            .map(|k| format!("<strong>{}</strong> &nbsp;", esc(k)))
            .unwrap_or_default();
        let entities = if m.entities.is_empty() {
            "<span class=\"muted\">—</span>".to_string()
        } else {
            format!(
                r#"<div class="tag-list">{}</div>"#,
                m.entities
                    .iter()
                    .take(5)
                    .map(|e| format!(r#"<span class="tag">{}</span>"#, esc(e)))
                    .collect::<Vec<_>>()
                    .join("")
            )
        };
        table.push_str(&format!(
            r#"<tr><td>{ty}</td><td><a href="/admin/memories/{id}">{topic}<div class="content-preview">{content}</div></a></td><td>{ents}</td><td class="mono muted">{when}</td><td class="right">
  <form method="post" action="/admin/memories/{id}/delete" class="inline" onsubmit="return confirm('Hard delete this memory? This cannot be undone.')">
    <button type="submit" class="btn danger">delete</button>
  </form>
</td></tr>"#,
            ty = type_pill(&m.type_),
            id = m.id,
            topic = topic,
            content = esc(&m.content),
            ents = entities,
            when = format_when(m.created_at),
        ));
    }
    if memories.is_empty() {
        table.push_str(
            r#"<tr><td colspan="5"><p class="muted" style="text-align:center;padding:16px">No memories match the current filter.</p></td></tr>"#,
        );
    }
    table.push_str("</tbody></table>");

    let content = format!(
        r#"<h1>Memories</h1>
<p class="muted">{total} total matching this filter. URL state: <code>?{q}</code></p>
{filter_form}
{table}"#,
        total = total,
        q = esc(&q),
        filter_form = filter_form,
        table = table,
    );
    page("memories", user_id, &content)
}

fn build_query_string(f: &MemoriesFilter) -> String {
    let mut parts = Vec::new();
    if let Some(t) = &f.r#type {
        parts.push(format!("type={}", t));
    }
    if let Some(p) = &f.project_id {
        parts.push(format!("project_id={}", p));
    }
    if let Some(s) = &f.session_id {
        parts.push(format!("session_id={}", s));
    }
    if f.include_superseded {
        parts.push("include_superseded=1".to_string());
    }
    parts.join("&")
}

// ---------------------------------------------------------------------------
// Memory detail
// ---------------------------------------------------------------------------

pub fn memory_detail(
    user_id: &str,
    m: &MemoryView,
    chain: &[MemoryView],
    extraction: Option<&Value>,
) -> String {
    let terminal_id = chain
        .iter()
        .find(|v| v.superseded_by.is_none())
        .map(|v| v.id);
    let chain_html = render_chain(chain, terminal_id, m.id);
    let extraction_html = match extraction {
        Some(v) => format!(
            r#"<pre class="json">{}</pre>"#,
            esc(&serde_json::to_string_pretty(v).unwrap_or_default())
        ),
        None => {
            r#"<p class="muted">No structured extraction recorded for this memory.</p>"#.to_string()
        }
    };
    let entities_html = if m.entities.is_empty() {
        "<span class=\"muted\">—</span>".to_string()
    } else {
        format!(
            r#"<div class="tag-list">{}</div>"#,
            m.entities
                .iter()
                .map(|e| format!(r#"<span class="tag">{}</span>"#, esc(e)))
                .collect::<Vec<_>>()
                .join("")
        )
    };

    let state_block = if let Some(data) = &m.state_data {
        format!(
            r#"<div class="card"><h2 style="margin-top:0">State data — {kind}/{key}</h2>
<pre class="json">{json}</pre></div>"#,
            kind = esc(m.state_kind.as_deref().unwrap_or("?")),
            key = esc(m.state_key.as_deref().unwrap_or("?")),
            json = esc(&serde_json::to_string_pretty(data).unwrap_or_default())
        )
    } else {
        String::new()
    };

    let content = format!(
        r#"<p><a href="/admin/memories">← back to memories</a></p>
<h1>{ty} <span class="mono muted" style="font-size:14px;font-weight:normal">{id}</span></h1>

<div class="row">
  <div class="card">
    <h2 style="margin-top:0">Content</h2>
    <pre class="json">{content}</pre>
    <p class="muted" style="margin-top:12px;font-size:12px">
      created {when} · importance {imp:.2} · decay <code>{decay}</code>{sup}
    </p>
    <div style="margin-top:16px">
      <span class="muted">entities:</span> {ents}
    </div>
    {project}
    {session}
  </div>
  <div class="card">
    <h2 style="margin-top:0">Structured extraction</h2>
    {extraction_html}
  </div>
</div>

{state_block}

<div class="card">
  <h2 style="margin-top:0">Supersede chain ({n} node{plural})</h2>
  {chain_html}
</div>

<form method="post" action="/admin/memories/{id}/delete" onsubmit="return confirm('Hard delete this memory? This cannot be undone.')">
  <button type="submit" class="btn danger">Delete this memory</button>
</form>"#,
        ty = type_pill(&m.type_),
        id = m.id,
        content = esc(&m.content),
        when = format_when(m.created_at),
        imp = m.importance,
        decay = esc(&m.decay_class),
        sup = if let Some(s) = m.superseded_by {
            format!(
                " · superseded by <a href=\"/admin/memories/{s}\">{}</a>",
                short_id(s)
            )
        } else {
            String::new()
        },
        ents = entities_html,
        project = m
            .project_id
            .as_deref()
            .map(|p| format!(
                r#"<p class="muted" style="margin-top:6px">project: <code>{}</code></p>"#,
                esc(p)
            ))
            .unwrap_or_default(),
        session = m
            .session_id
            .as_deref()
            .map(|s| format!(r#"<p class="muted">session: <code>{}</code></p>"#, esc(s)))
            .unwrap_or_default(),
        extraction_html = extraction_html,
        state_block = state_block,
        n = chain.len(),
        plural = if chain.len() == 1 { "" } else { "s" },
        chain_html = chain_html,
    );

    page("memories", user_id, &content)
}

fn render_chain(chain: &[MemoryView], terminal_id: Option<Uuid>, current_id: Uuid) -> String {
    if chain.is_empty() {
        return r#"<p class="muted">No supersede chain.</p>"#.to_string();
    }
    let mut out = String::from(r#"<div class="chain">"#);
    for (i, m) in chain.iter().enumerate() {
        let is_terminal = terminal_id == Some(m.id);
        let is_current = current_id == m.id;
        let cls = if is_terminal {
            "chain-node terminal"
        } else {
            "chain-node"
        };
        let marker = if is_current { " (this memory)" } else { "" };
        let terminal_label = if is_terminal {
            r#" <span class="pill t-semantic">current</span>"#
        } else {
            ""
        };
        out.push_str(&format!(
            r#"<div class="{cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
    <div style="flex:1">
      <strong><a href="/admin/memories/{id}">{short}</a></strong>{terminal_label} <span class="mono muted">{when}</span>{marker}
      <div class="content-preview" style="margin-top:6px;color:var(--fg-0)">{content}</div>
    </div>
  </div>
</div>"#,
            id = m.id,
            short = short_id(m.id),
            content = esc(&m.content),
            when = format_when(m.created_at),
        ));
        if i < chain.len() - 1 {
            out.push_str(r#"<div class="chain-arrow">↓ superseded by</div>"#);
        }
    }
    out.push_str("</div>");
    out
}

// ---------------------------------------------------------------------------
// State objects
// ---------------------------------------------------------------------------

pub fn state_list(user_id: &str, states: &[MemoryView]) -> String {
    let mut body = String::new();
    if states.is_empty() {
        body.push_str(r#"<p class="muted">No state objects yet. Create one via POST /state/todo_list (or any other kind).</p>"#);
    } else {
        for s in states {
            let rendered = esc(&s.content);
            let key = esc(s.state_key.as_deref().unwrap_or("?"));
            let kind = esc(s.state_kind.as_deref().unwrap_or("?"));
            body.push_str(&format!(
                r#"<div class="card">
  <h2 style="margin-top:0"><a href="/admin/memories/{id}"><code>{kind}/{key}</code></a></h2>
  <pre class="json">{rendered}</pre>
  <p class="muted" style="font-size:12px">updated {when}</p>
</div>"#,
                id = s.id,
                kind = kind,
                key = key,
                rendered = rendered,
                when = format_when(s.last_accessed_at),
            ));
        }
    }
    let content = format!(
        r#"<h1>State objects</h1>
<p class="muted">The "heap" — mutable named cells. See <a href="https://github.com/Horizon-Digital-Engineering/flashback/blob/main/docs/REFERENCES.md">docs/REFERENCES.md</a>.</p>
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
<p class="muted">{n} memories, {e} edges. <strong>3D scene</strong> rendered in plain canvas2d — hand-rolled perspective projection, depth-sorted, no Three.js / WebGL / framework. Drag to orbit the camera, scroll to zoom, click a node to open. Switch to 2D for the force-directed flat layout.</p>

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
<p class="map-help">Memories at their PCA-projected positions (3 principal components of the 384-dim embedding). Distance ≈ semantic similarity. Edge color = supersede / entity-overlap / same-session.</p>
{js}"#,
        n = node_count,
        e = edge_count,
        js = MAP_JS,
    );
    page("map", user_id, &content)
}

// ---------------------------------------------------------------------------
// Consolidation
// ---------------------------------------------------------------------------

pub fn consolidate_view(
    user_id: &str,
    runs: &[impl serde::Serialize],
    provider_can_distill: bool,
    provider_name: &str,
) -> String {
    let runs_json = serde_json::to_value(runs).unwrap_or(Value::Null);
    let rows_arr = runs_json.as_array().cloned().unwrap_or_default();
    let mut rows_html = String::new();
    for r in rows_arr.iter() {
        let kind = r.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
        let trigger = r.get("trigger").and_then(|v| v.as_str()).unwrap_or("?");
        let started = r.get("started_at").and_then(|v| v.as_str()).unwrap_or("?");
        let finished = r.get("finished_at").and_then(|v| v.as_str()).unwrap_or("—");
        let promoted = r
            .get("promoted_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let expired = r.get("expired_count").and_then(|v| v.as_i64()).unwrap_or(0);
        let distilled = r
            .get("distilled_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let user = r.get("user_id").and_then(|v| v.as_str()).unwrap_or("");
        let err = r.get("error").and_then(|v| v.as_str()).unwrap_or("");
        let status = if !err.is_empty() {
            format!(
                r#"<span class="pill" style="color:var(--bad)">error</span> <span class="muted">{}</span>"#,
                esc(err)
            )
        } else if finished == "—" {
            r#"<span class="pill" style="color:var(--warn)">running</span>"#.to_string()
        } else {
            format!(
                r#"<span class="pill" style="color:var(--good)">ok</span> +{promoted} promoted · {expired} expired · {distilled} distilled"#
            )
        };
        rows_html.push_str(&format!(
            r#"<tr><td><span class="pill">{kind}</span></td><td>{trigger}</td><td>{status}</td><td class="muted">{user}</td><td class="mono muted">{started}</td></tr>"#,
            kind = esc(kind),
            trigger = esc(trigger),
            status = status,
            user = esc(user),
            started = esc(&started.chars().take(19).collect::<String>()),
        ));
    }
    if rows_arr.is_empty() {
        rows_html.push_str(r#"<tr><td colspan="5" class="muted" style="text-align:center;padding:16px">No consolidation runs yet. Trigger one below or wait for the daily/weekly scheduler.</td></tr>"#);
    }

    let distill_note = if provider_can_distill {
        format!(
            r#"<p class="muted" style="font-size:13px">Weekly distillation uses provider <code>{}</code> — LLM-grade fact extraction is available.</p>"#,
            esc(provider_name)
        )
    } else {
        format!(
            r#"<div class="error" style="background:rgba(255,189,85,0.08);border-color:var(--warn);color:var(--warn)">
                Current provider <code>{}</code> does not implement fact distillation. The weekly job will skip with a warning until you set <code>PROVIDER=remote</code> (or <code>=embedded</code>) and supply credentials/model. The daily job (promote working→episodic) still works fine.
            </div>"#,
            esc(provider_name)
        )
    };

    let content = format!(
        r#"<h1>Consolidation</h1>
<p class="muted">Daily promotes <code>working</code> → <code>episodic</code> based on importance / access. Weekly clusters episodic memories ≥7d old by topic + entity overlap and asks the AI provider to distill them into <code>semantic</code> facts.</p>
{distill_note}
<div class="card" style="display:flex;gap:12px;align-items:center">
  <form method="post" action="/admin/api/consolidate/daily" class="inline">
    <button type="submit">Run daily now</button>
  </form>
  <form method="post" action="/admin/api/consolidate/weekly" class="inline">
    <button type="submit">Run weekly now</button>
  </form>
  <span class="muted" style="font-size:12px">manual runs scope to <code>{user_id}</code>; the scheduler iterates all users.</span>
</div>
<h2>Recent runs</h2>
<table>
<thead><tr><th>kind</th><th>trigger</th><th>status</th><th>user</th><th>started</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>"#,
        user_id = esc(user_id),
    );

    page("consolidate", user_id, &content)
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

pub fn format_when(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%d %H:%M").to_string()
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

// CoreMemoryRow display is referenced in dashboard helpers from handlers.rs;
// keep the import alive even if no current call site uses it.
#[allow(dead_code)]
fn _link_to_core(_c: &CoreMemoryRow) {}
