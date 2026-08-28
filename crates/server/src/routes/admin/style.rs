//! Embedded stylesheet for the admin UI. Served as static text from
//! `/admin/style.css`. Hand-written, ~250 lines — no Tailwind / CDN / build
//! step. Dark theme by default, system fonts for fast first paint.

pub const STYLE_CSS: &str = r#"
:root {
    --bg-0: #0d0e12;
    --bg-1: #14161c;
    --bg-2: #1c1f28;
    --bg-3: #262a36;
    --border: #2d3140;
    --fg-0: #e6e7eb;
    --fg-1: #b8bcc8;
    --fg-2: #7e8294;
    --accent: #6f9eff;
    --accent-hover: #88b3ff;
    --good: #5cd07a;
    --warn: #ffbd55;
    --bad:  #ff7a7a;
    --type-episodic:   #6f9eff;
    --type-semantic:   #5cd07a;
    --type-working:    #ffbd55;
    --type-document:   #b8bcc8;
    --type-procedural: #c594ff;
    --type-state:      #ff8fbf;
}

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    background: var(--bg-0);
    color: var(--fg-0);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", Roboto, Oxygen, sans-serif;
    font-size: 14px;
    line-height: 1.5;
}

code, pre, .mono {
    font-family: "SF Mono", "Menlo", "Monaco", "Consolas", "Liberation Mono", monospace;
    font-size: 13px;
}

a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hover); text-decoration: underline; }

.page {
    max-width: 1180px;
    margin: 0 auto;
    padding: 24px;
}

/* The build stamp shares the page's column instead of hugging the viewport
   edge, and reads as chrome, not content. */
footer.build {
    max-width: 1180px;
    margin: 32px auto 16px;
    padding: 8px 24px;
    color: var(--fg-2);
    font-size: 11px;
    opacity: 0.7;
}

.nav {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 16px 24px;
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
}
.nav .brand {
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-right: 24px;
}
.nav a {
    color: var(--fg-1);
    padding: 6px 12px;
    border-radius: 6px;
}
.nav a:hover {
    color: var(--fg-0);
    background: var(--bg-2);
    text-decoration: none;
}
.nav a.active {
    color: var(--fg-0);
    background: var(--bg-3);
}
.nav .spacer { flex: 1; }
.nav .user {
    color: var(--fg-2);
    font-size: 12px;
}

h1, h2, h3 { font-weight: 600; margin: 24px 0 12px; }
h1 { font-size: 24px; }
h2 { font-size: 18px; }
h3 { font-size: 15px; color: var(--fg-1); }

.muted { color: var(--fg-2); }
.right { text-align: right; }
.row { display: flex; gap: 16px; }
.row > * { flex: 1; }

.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 16px 0 32px;
}
.stat {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.stat .label {
    color: var(--fg-2);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stat .value {
    font-size: 24px;
    font-weight: 600;
    margin-top: 4px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
th, td {
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}
th {
    background: var(--bg-2);
    color: var(--fg-1);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-2); }

.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
    background: var(--bg-3);
    color: var(--fg-1);
    text-transform: lowercase;
}
.pill.t-episodic   { background: rgba(111,158,255,0.16); color: var(--type-episodic); }
.pill.t-semantic   { background: rgba(92,208,122,0.16);  color: var(--type-semantic); }
.pill.t-working    { background: rgba(255,189,85,0.16);  color: var(--type-working); }
.pill.t-document   { background: rgba(184,188,200,0.16); color: var(--type-document); }
.pill.t-procedural { background: rgba(197,148,255,0.16); color: var(--type-procedural); }
.pill.t-state_object { background: rgba(255,143,191,0.16); color: var(--type-state); }

.content-preview {
    color: var(--fg-1);
    white-space: pre-wrap;
    word-break: break-word;
    max-width: 540px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.tag-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}
.tag {
    background: var(--bg-3);
    color: var(--fg-1);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
    font-family: "SF Mono", "Menlo", monospace;
}

.card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}

pre.json {
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    color: var(--fg-1);
    white-space: pre-wrap;
    word-break: break-word;
}

.chain {
    display: flex;
    flex-direction: column;
    gap: 0;
}
.chain-node {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    margin: 4px 0;
}
.chain-node.terminal {
    border-color: var(--good);
    background: rgba(92,208,122,0.06);
}
.chain-arrow {
    text-align: center;
    color: var(--fg-2);
    font-size: 12px;
    padding: 2px 0;
}

form.inline {
    display: inline-block;
    margin: 0;
}

button, .btn {
    background: var(--bg-3);
    color: var(--fg-0);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    cursor: pointer;
    font-family: inherit;
}
button:hover, .btn:hover {
    background: var(--bg-2);
    border-color: var(--accent);
}
.btn.danger {
    color: var(--bad);
    border-color: rgba(255,122,122,0.3);
}
.btn.danger:hover {
    background: rgba(255,122,122,0.08);
}

input[type="text"], input[type="password"], textarea {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    color: var(--fg-0);
    font-family: inherit;
    font-size: 14px;
    width: 100%;
}
input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
    outline: none;
    border-color: var(--accent);
}

.login-page {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}
.login-box {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    width: 380px;
}
.login-box h1 { text-align: center; margin-bottom: 4px; }
.login-box .sub { text-align: center; color: var(--fg-2); margin-bottom: 24px; }
.login-box button { width: 100%; padding: 10px; margin-top: 16px; background: var(--accent); border-color: var(--accent); color: white; }
.login-box button:hover { background: var(--accent-hover); }

.error {
    background: rgba(255,122,122,0.08);
    border: 1px solid rgba(255,122,122,0.3);
    color: var(--bad);
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 16px;
}

/* ---- Map view ---- */
.map-wrap {
    background: radial-gradient(ellipse at center, #161823 0%, #0d0e12 80%);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0;
    overflow: hidden;
    height: 680px;
    position: relative;
}
#map-canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: grab;
    user-select: none;
}
#map-canvas.dragging { cursor: grabbing; }
.map-mode-toggle {
    position: absolute;
    bottom: 12px;
    left: 12px;
    background: rgba(20,22,28,0.92);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
    display: flex;
    gap: 0;
    font-size: 12px;
}
.map-mode-toggle button {
    background: transparent;
    border: 0;
    color: var(--fg-2);
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-family: inherit;
}
.map-mode-toggle button.active {
    background: var(--bg-3);
    color: var(--fg-0);
}
.map-legend {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(20,22,28,0.92);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 12px;
}
.map-legend .key { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
.map-legend .swatch { width: 10px; height: 10px; border-radius: 50%; }
.map-tooltip {
    position: absolute;
    background: rgba(13,14,18,0.96);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 12px;
    pointer-events: none;
    max-width: 360px;
    display: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
.map-help {
    color: var(--fg-2);
    font-size: 11px;
    margin: 8px 0 0;
}

/* ---- Dev-mode banner ---- */
.dev-banner {
    background: repeating-linear-gradient(
        45deg,
        rgba(255,189,85,0.10),
        rgba(255,189,85,0.10) 12px,
        rgba(255,189,85,0.18) 12px,
        rgba(255,189,85,0.18) 24px
    );
    border-bottom: 2px solid var(--warn);
    color: var(--warn);
    text-align: center;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.dev-banner code { color: var(--warn); }
"#;
