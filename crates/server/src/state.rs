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
