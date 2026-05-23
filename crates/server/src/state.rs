//! State objects — the reference half of memory.
//!
//! Each `state_kind` defines:
//! * a JSON shape for `state_data`
//! * a set of mutation ops
//! * a deterministic textual `render` used for embeddings + prompt injection
//!
//! Phase 1 ships `todo_list`. Adding new kinds = new module + dispatch arm.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::error::AppError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    TodoList,
}

impl StateKind {
    pub fn parse(s: &str) -> Result<Self, AppError> {
        match s {
            "todo_list" => Ok(Self::TodoList),
            other => Err(AppError::bad_request(format!(
                "unknown state_kind: {other}"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TodoList => "todo_list",
        }
    }

    /// Validate a fresh `state_data` blob for this kind. Returns the
    /// normalized value (auto-fills missing ids, etc.).
    pub fn validate_initial(&self, data: &Value) -> Result<Value, AppError> {
        match self {
            Self::TodoList => {
                let mut list: TodoList = serde_json::from_value(data.clone())
                    .map_err(|e| AppError::bad_request(format!("invalid todo_list: {e}")))?;
                list.assign_missing_ids();
                Ok(serde_json::to_value(list).expect("serializable"))
            }
        }
    }

    pub fn empty(&self) -> Value {
        match self {
            Self::TodoList => json!({ "items": [] }),
        }
    }

    pub fn render(&self, data: &Value, state_key: &str) -> String {
        match self {
            Self::TodoList => render_todo_list(data, state_key),
        }
    }

    pub fn apply(&self, data: &Value, op: &Value) -> Result<Value, AppError> {
        match self {
            Self::TodoList => apply_todo_op(data, op),
        }
    }
}

// ---------------------------------------------------------------------------
// todo_list
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TodoList {
    #[serde(default)]
    pub items: Vec<TodoItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    #[serde(default = "new_id")]
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub done: bool,
}

fn new_id() -> String {
    Uuid::new_v4().to_string()
}

impl TodoList {
    fn assign_missing_ids(&mut self) {
        for item in &mut self.items {
            if item.id.is_empty() {
                item.id = new_id();
            }
        }
    }
}

fn render_todo_list(data: &Value, state_key: &str) -> String {
    let list: TodoList = serde_json::from_value(data.clone()).unwrap_or_default();
    if list.items.is_empty() {
        return format!("Todo list \"{state_key}\" is empty.");
    }
    let mut out = format!("Todo list \"{state_key}\":\n");
    for item in &list.items {
        let mark = if item.done { "[x]" } else { "[ ]" };
        out.push_str(&format!("  {mark} {}\n", item.text));
    }
    out.trim_end().to_string()
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum TodoOp {
    Add { text: String },
    MarkDone { item_id: String },
    Unmark { item_id: String },
    Toggle { item_id: String },
    Remove { item_id: String },
    Update { item_id: String, text: String },
    Reorder { ids: Vec<String> },
    Replace { items: Vec<TodoItem> },
    Clear,
}

fn apply_todo_op(data: &Value, op_value: &Value) -> Result<Value, AppError> {
    let mut list: TodoList = serde_json::from_value(data.clone())
        .map_err(|e| AppError::bad_request(format!("corrupt todo_list state: {e}")))?;

    let op: TodoOp = serde_json::from_value(op_value.clone())
        .map_err(|e| AppError::bad_request(format!("invalid todo op: {e}")))?;

    match op {
        TodoOp::Add { text } => {
            let text = text.trim();
            if text.is_empty() {
                return Err(AppError::bad_request("add: text cannot be empty"));
            }
            list.items.push(TodoItem {
                id: new_id(),
                text: text.to_string(),
                done: false,
            });
        }
        TodoOp::MarkDone { item_id } => {
            let item = find_mut(&mut list.items, &item_id)?;
            item.done = true;
        }
        TodoOp::Unmark { item_id } => {
            let item = find_mut(&mut list.items, &item_id)?;
            item.done = false;
        }
        TodoOp::Toggle { item_id } => {
            let item = find_mut(&mut list.items, &item_id)?;
            item.done = !item.done;
        }
        TodoOp::Remove { item_id } => {
            let before = list.items.len();
            list.items.retain(|i| i.id != item_id);
            if list.items.len() == before {
                return Err(AppError::not_found(format!("item {item_id}")));
            }
        }
        TodoOp::Update { item_id, text } => {
            let text = text.trim();
            if text.is_empty() {
                return Err(AppError::bad_request("update: text cannot be empty"));
            }
            let item = find_mut(&mut list.items, &item_id)?;
            item.text = text.to_string();
        }
        TodoOp::Reorder { ids } => {
            if ids.len() != list.items.len() {
                return Err(AppError::bad_request(
                    "reorder: ids must cover every item exactly once",
                ));
            }
            let mut new_items: Vec<TodoItem> = Vec::with_capacity(list.items.len());
            for id in &ids {
                let pos = list
                    .items
                    .iter()
                    .position(|i| &i.id == id)
                    .ok_or_else(|| AppError::not_found(format!("item {id}")))?;
                new_items.push(list.items.remove(pos));
            }
            list.items = new_items;
        }
        TodoOp::Replace { mut items } => {
            for item in &mut items {
                if item.id.is_empty() {
                    item.id = new_id();
                }
            }
            list.items = items;
        }
        TodoOp::Clear => {
            list.items.clear();
        }
    }

    Ok(serde_json::to_value(list).expect("serializable"))
}

fn find_mut<'a>(items: &'a mut [TodoItem], id: &str) -> Result<&'a mut TodoItem, AppError> {
    items
        .iter_mut()
        .find(|i| i.id == id)
        .ok_or_else(|| AppError::not_found(format!("item {id}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list_with(items: &[(&str, &str, bool)]) -> Value {
        let v: Vec<Value> = items
            .iter()
            .map(|(id, text, done)| json!({ "id": id, "text": text, "done": done }))
            .collect();
        json!({ "items": v })
    }

    fn apply(data: &Value, op: Value) -> Result<Value, AppError> {
        StateKind::TodoList.apply(data, &op)
    }

    fn items_from(v: &Value) -> Vec<TodoItem> {
        let list: TodoList = serde_json::from_value(v.clone()).unwrap();
        list.items
    }

    // ---- StateKind --------------------------------------------------------

    #[test]
    fn state_kind_parse_known_and_unknown() {
        assert_eq!(StateKind::parse("todo_list").unwrap(), StateKind::TodoList);
        assert!(StateKind::parse("garbage").is_err());
    }

    #[test]
    fn state_kind_round_trips() {
        assert_eq!(StateKind::TodoList.as_str(), "todo_list");
    }

    #[test]
    fn state_kind_empty_is_empty_items() {
        let e = StateKind::TodoList.empty();
        assert_eq!(e, json!({ "items": [] }));
    }

    #[test]
    fn validate_initial_assigns_missing_ids() {
        let raw = json!({ "items": [{ "text": "first" }, { "text": "second" }] });
        let normalized = StateKind::TodoList.validate_initial(&raw).unwrap();
        let items = items_from(&normalized);
        assert_eq!(items.len(), 2);
        assert!(!items[0].id.is_empty());
        assert!(!items[1].id.is_empty());
        assert_ne!(items[0].id, items[1].id);
    }

    #[test]
    fn validate_initial_rejects_type_mismatch() {
        // `items` exists but is the wrong type (string instead of array).
        // Empty / missing items would deserialize fine via `#[serde(default)]`,
        // so we have to actually clash on a field type to trigger the error.
        assert!(StateKind::TodoList
            .validate_initial(&json!({ "items": "not an array" }))
            .is_err());
    }

    // ---- render -----------------------------------------------------------

    #[test]
    fn render_empty_list() {
        let out = StateKind::TodoList.render(&json!({ "items": [] }), "today");
        assert!(out.contains("today"));
        assert!(out.contains("empty"));
    }

    #[test]
    fn render_mixed_state() {
        let data = list_with(&[("a", "buy milk", false), ("b", "feed cat", true)]);
        let out = StateKind::TodoList.render(&data, "home");
        assert!(out.contains("home"));
        assert!(out.contains("[ ] buy milk"));
        assert!(out.contains("[x] feed cat"));
    }

    // ---- TodoOp::Add ------------------------------------------------------

    #[test]
    fn op_add_appends_item() {
        let before = list_with(&[]);
        let after = apply(&before, json!({ "op": "add", "text": "new" })).unwrap();
        let items = items_from(&after);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "new");
        assert!(!items[0].done);
        assert!(!items[0].id.is_empty());
    }

    #[test]
    fn op_add_trims_whitespace() {
        let after = apply(
            &list_with(&[]),
            json!({ "op": "add", "text": "  spaced  " }),
        )
        .unwrap();
        assert_eq!(items_from(&after)[0].text, "spaced");
    }

    #[test]
    fn op_add_rejects_empty_or_whitespace() {
        assert!(apply(&list_with(&[]), json!({ "op": "add", "text": "" })).is_err());
        assert!(apply(&list_with(&[]), json!({ "op": "add", "text": "    " })).is_err());
    }

    // ---- mark_done / unmark / toggle --------------------------------------

    #[test]
    fn op_mark_done_sets_done_true() {
        let before = list_with(&[("a", "task", false)]);
        let after = apply(&before, json!({ "op": "mark_done", "item_id": "a" })).unwrap();
        assert!(items_from(&after)[0].done);
    }

    #[test]
    fn op_unmark_sets_done_false() {
        let before = list_with(&[("a", "task", true)]);
        let after = apply(&before, json!({ "op": "unmark", "item_id": "a" })).unwrap();
        assert!(!items_from(&after)[0].done);
    }

    #[test]
    fn op_toggle_flips_done() {
        let before = list_with(&[("a", "task", false)]);
        let after = apply(&before, json!({ "op": "toggle", "item_id": "a" })).unwrap();
        assert!(items_from(&after)[0].done);
        let again = apply(&after, json!({ "op": "toggle", "item_id": "a" })).unwrap();
        assert!(!items_from(&again)[0].done);
    }

    #[test]
    fn ops_error_on_unknown_item_id() {
        let before = list_with(&[("a", "task", false)]);
        assert!(apply(&before, json!({ "op": "mark_done", "item_id": "ghost" })).is_err());
        assert!(apply(&before, json!({ "op": "toggle", "item_id": "ghost" })).is_err());
        assert!(apply(&before, json!({ "op": "remove", "item_id": "ghost" })).is_err());
        assert!(apply(
            &before,
            json!({ "op": "update", "item_id": "ghost", "text": "x" })
        )
        .is_err());
    }

    // ---- remove ----------------------------------------------------------

    #[test]
    fn op_remove_drops_by_id() {
        let before = list_with(&[("a", "first", false), ("b", "second", false)]);
        let after = apply(&before, json!({ "op": "remove", "item_id": "a" })).unwrap();
        let items = items_from(&after);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "b");
    }

    // ---- update ----------------------------------------------------------

    #[test]
    fn op_update_changes_text() {
        let before = list_with(&[("a", "old", false)]);
        let after = apply(
            &before,
            json!({ "op": "update", "item_id": "a", "text": "new text" }),
        )
        .unwrap();
        assert_eq!(items_from(&after)[0].text, "new text");
    }

    #[test]
    fn op_update_rejects_empty_text() {
        let before = list_with(&[("a", "old", false)]);
        assert!(apply(
            &before,
            json!({ "op": "update", "item_id": "a", "text": "" })
        )
        .is_err());
        assert!(apply(
            &before,
            json!({ "op": "update", "item_id": "a", "text": "   " })
        )
        .is_err());
    }

    // ---- reorder ---------------------------------------------------------

    #[test]
    fn op_reorder_rearranges_items() {
        let before = list_with(&[
            ("a", "one", false),
            ("b", "two", false),
            ("c", "three", false),
        ]);
        let after = apply(&before, json!({ "op": "reorder", "ids": ["c", "a", "b"] })).unwrap();
        let ids: Vec<String> = items_from(&after).into_iter().map(|i| i.id).collect();
        assert_eq!(ids, vec!["c", "a", "b"]);
    }

    #[test]
    fn op_reorder_rejects_partial_id_list() {
        let before = list_with(&[("a", "x", false), ("b", "y", false)]);
        assert!(apply(&before, json!({ "op": "reorder", "ids": ["a"] })).is_err());
        assert!(apply(&before, json!({ "op": "reorder", "ids": ["a", "b", "c"] })).is_err());
    }

    #[test]
    fn op_reorder_rejects_unknown_id() {
        let before = list_with(&[("a", "x", false), ("b", "y", false)]);
        assert!(apply(&before, json!({ "op": "reorder", "ids": ["a", "ghost"] })).is_err());
    }

    // ---- replace ---------------------------------------------------------

    #[test]
    fn op_replace_swaps_full_list_and_assigns_missing_ids() {
        let before = list_with(&[("a", "old", false)]);
        let after = apply(
            &before,
            json!({
                "op": "replace",
                "items": [{ "text": "new1" }, { "text": "new2" }]
            }),
        )
        .unwrap();
        let items = items_from(&after);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "new1");
        assert_eq!(items[1].text, "new2");
        assert!(!items[0].id.is_empty());
        assert!(!items[1].id.is_empty());
    }

    // ---- clear -----------------------------------------------------------

    #[test]
    fn op_clear_empties_list() {
        let before = list_with(&[("a", "x", false), ("b", "y", true)]);
        let after = apply(&before, json!({ "op": "clear" })).unwrap();
        assert!(items_from(&after).is_empty());
    }

    // ---- error shapes ----------------------------------------------------

    #[test]
    fn op_unknown_returns_error() {
        let before = list_with(&[]);
        assert!(apply(&before, json!({ "op": "shrug" })).is_err());
    }

    #[test]
    fn op_corrupt_input_returns_error() {
        // `items` exists but as the wrong type — that's what actually trips
        // deserialization. `{ "wrong": [] }` would NOT error because TodoList
        // has #[serde(default)] on the items field.
        let result = apply(
            &json!({ "items": "string-not-array" }),
            json!({ "op": "clear" }),
        );
        assert!(result.is_err());
    }
}
