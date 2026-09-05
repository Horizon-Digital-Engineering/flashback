# Records and References — The Heap Hypothesis

*A hypothesis we're testing: the missing piece in current AI memory systems isn't smarter retrieval — it's the distinction between immutable records and mutable references.*

---

## Status

**Hypothesis under test.** This document is not a settled design. It is the framing under which Flashback's `state_object` type was added. We are building it, watching it work in real conversations, and will revise (or retract) this document based on what we learn.

If you're reading this and the implementation no longer matches, the hypothesis lost. Check `git log`.

---

## The Thesis in One Sentence

> Every memory system today models append-only records (facts, episodes, embeddings). None of them model **references** — named cells of mutable state that the system maintains a current value for. Human memory has both. Computers have both. AI memory should have both.

---

## The Gap in the Original Spec

The Flashback architecture document defines five memory types: episodic, semantic, working, document, procedural. All five are **records**. Each is an immutable entry written once at some point in time, optionally superseded later by another record.

What is missing is the **reference**: a named mutable cell whose value evolves over time and which retrieval treats as a single object rather than a stream of timestamped opinions.

Two everyday examples that exposed the gap:

1. **A running todo list inside a conversation.** "Add buy milk." "Got the milk, cross it off." "What's on my list?" Without references, the system has stored three episodic snippets and retrieval returns all three. The LLM has to reconcile them in-context every time. The retrieval cost grows with conversation length.

2. **An evolving mental model of a project.** "We're using Postgres." "Actually we switched to Postgres + pgvector." "We added Redis too." Without references, the system stores three claims about "the stack" and retrieves all three. With references, the project's stack is a single variable that the system updates.

Both cases involve a thing the user expects the system to *maintain*, not just *log*.

---

## The Computer-Memory Metaphor, Extended

The original architecture leans on the computer-memory analogy: ROM, cache, RAM, disk. That analogy is fine for the *tier* hierarchy (latency, capacity, persistence). But it stops short of describing what *kinds* of things live at each tier.

Computers have both kinds of memory, distinguished by mutability:

```
RECORDS (immutable, append-only)        REFERENCES (mutable, named)
─────────────────────────────────       ────────────────────────────
log entries                             variables in scope
constants in the data segment           the heap
config files                            environment variables
the instruction stream                  the stack frame
```

The two categories serve different jobs:
- **Records** are how systems answer "what happened, in what order, and why."
- **References** are how systems answer "what is the current value of this thing."

You cannot replace one with the other. A log is not a database; a database is not a log. Most existing AI memory systems try to make a log do a database's job. Hence the failure mode in the todo-list example: a log of "add", "remove", "add" is technically sufficient to reconstruct the current list, but the reconstruction cost grows with history length and the reconstruction is fragile.

---

## The Human-Memory Mapping

The same distinction shows up in human cognition.

```
RECORDS                                 REFERENCES
─────────────────────────────────       ────────────────────────────
"I remember X happened last March"      "my current shopping list"
"I know that X is true"                 "my evolving theory of how Bob behaves"
"I learned how to do X"                 "what I am trying to accomplish today"
"I read that book"                      "the plan I'm executing right now"
```

When someone asks *"what's on your shopping list?"* you do not grep your episodic memory for the latest "I added X" event. You query the **current state of the list variable** you have been maintaining. That is a structurally different cognitive operation, and the original spec did not give the system a way to perform it.

Conversely, when someone asks *"how did the plan evolve?"* you walk the history — that is what episodic memory is for. The right model is both, side by side, addressed differently.

---

## Type × Tier Are Orthogonal Axes

Recognizing references as a memory category reveals that the original spec conflated two independent dimensions: *what kind of thing* a memory is, and *where in the hierarchy* it lives.

The spec called Tier 3 (RAM) "working memory" as if "working" were a type. It is not — "working" is a *location*. The same kind of thing can live at different tiers depending on access pattern:

|              | Tier 2 — Cache (always-on) | Tier 3 — RAM (session-scoped) | Tier 4 — Disk (long-term) |
|--------------|-----------------------------|-------------------------------|---------------------------|
| **Episodic** | rare                       | recent turns                 | session history          |
| **Semantic** | core facts                 | inferences from this session | established truths       |
| **State Object (ref)** | pinned plans ("main project plan") | active scratchpad ("today's todos") | archived plans ("Q3 plan v4") |
| **Procedural** | core procedures          | learning a new workflow      | mastered routines        |
| **Document** | rare                       | open file/page               | reference shelf          |

This refinement does not require a schema migration — the original `memories` table already has `decay_class` and `expires_at` columns that determine tier. It just clarifies that tier is a *property* of a memory, not a *type* of memory.

---

## How `state_object` Works in Flashback

A `state_object` is a record whose type is `state_object` and whose `payload` carries the reference convention:

```json
{
  "kind": "todo_list",          // 'todo_list' | 'plan' | 'decision_log' | …
  "key":  "groceries",          // canonical name within (user_id, kind)
  "data": { }                   // the actual structured value
}
```

Identity lives in the payload and nowhere else. It used to be copied onto columns by an insert trigger, which is normalising on the way in — and the trigger fired after the one computing the tamper hash, so every state object reported itself as tampered. An expression index on `(user_id, payload->>'kind', payload->>'key')` makes it queryable without the copy, and is faster than the columns were.

The triple `(user_id, kind, key)` identifies a logical variable. The terminal row — the newest for that identity — holds the current value. Older rows are the audit trail.

**Invariants:**
- `state_data` always contains the complete current value, not a delta. This means the terminal row is self-contained — no chain walk required to render the current state.
- `content` is a deterministic textual rendering of `state_data`, so embeddings + BM25 still work and the object can show up in standard semantic retrieval.
- Updates create new rows. The supersede edge is written to `derived_superseded`, not onto the raw row: the writer never claimed the older value was replaced, the server worked it out from identity and arrival order. That makes it a conclusion, and a rebuild is free to redo it.

**Mutation API:**

```
POST   /records/state/:kind/:key                 set (or revise) the value
GET    /records/state/:kind/:key                 current value
GET    /records/state/:kind/:key/history         every revision, oldest first
GET    /records/state/:kind                      everything of this kind
```

The PATCH body declares an `op` whose verbs depend on the `kind`. For `todo_list`:

```json
{ "op": "add",         "text": "buy milk" }
{ "op": "mark_done",   "item_id": "…" }
{ "op": "remove",      "item_id": "…" }
{ "op": "update",      "item_id": "…", "text": "…" }
{ "op": "reorder",     "ids": ["…", "…", …] }
{ "op": "clear" }
```

Each PATCH is an O(1) write that produces a new terminal node. The previous value remains queryable via `/history`.

**Prompt rendering:** when a state_object lands in `/records/context`, the renderer formats it as a structured block (e.g. `Project TODO: [✓] buy milk · [ ] fix bug`) rather than as embedded prose. This is what gives the LLM a clean current-state view instead of forcing it to reconstruct from history.

---

## Explicit vs Implicit Invocation

In Phase 1, state_objects are invoked **explicitly**: the integrator (the app calling Flashback) decides "this is a todo update" and posts to `/records/state/...`. We are not auto-classifying prose into state-update operations.

Auto-classification of stateful updates from raw conversation is tempting and dangerous. The worst failure mode of a memory system is *silently writing the wrong state* — an auto-classifier that misreads "I should probably buy milk tomorrow" as `{op: add, text: "buy milk"}` corrupts the user's working state in a way that is hard to detect and worse than no automation at all.

Phase 2 may add an opt-in auto-classifier behind a confidence threshold, with all auto-derived updates flagged and reviewable. That is research-grade and out of scope for now.

---

## How This Might Be Wrong (Things To Watch)

A hypothesis that cannot fail is not a hypothesis. Specific ways this design could turn out to be the wrong call:

1. **Most stateful objects in real conversations are too informal to address.** If users do not consistently say "my todo list" in a way that maps to a single `state_key`, the explicit-invocation API never gets used and the state_object type just sits there empty.

2. **Auto-classification turns out to be necessary, and the threshold problem dominates.** Explicit invocation only helps if the integrator hooks it up. If integrations cannot reliably detect when a user is doing a list operation, we're back to the prose-supersede chain.

3. **Token cost of structured rendering outpaces the cost of retrieving multiple episodic snippets.** If the rendered todo list is 400 tokens and three episodic snippets would have been 200, the cleaner abstraction loses on raw cost.

4. **The "current value" is rarely the right answer.** For some logical objects (a decision log, an evolving theory) the *evolution* is the value. Forcing them into a single-current-state model loses information.

5. **State_object becomes a junk drawer.** Without a clean line between "this should be a record" and "this should be a reference," developers reach for state_object as the default and we end up with mutable smear in places that should have been append-only history.

The success criterion is qualitative: do conversations with running stateful objects produce smaller, cleaner context windows under flashback than under prose-supersede chains, without losing audit-trail value? If yes, the hypothesis holds and this section becomes documentation. If no, we delete the type and this document becomes a postmortem.

---

## What Differs From Existing Systems

Comparing roughly to the public surfaces of mem0, Letta (MemGPT), and Zep:

- **mem0** stores facts with metadata and dedup via embedding similarity. A todo list becomes a sequence of facts with overlapping embeddings; the system might collapse some of them via similarity merging, but there is no concept of a named mutable cell. The "current value" of any logical object has to be inferred at retrieval time.
- **Letta** has a small, explicit core memory block that the agent can edit in place, plus an archival memory for retrieval. Core memory is a step in this direction but is a single block of prose, not a typed set of named structured objects. Updates are LLM-mediated rewrites, not deterministic ops.
- **Zep** offers temporal knowledge graphs with fact validity windows — strong on records, no native concept of structured references.

The thing Flashback is testing is whether **a typed, named, deterministically-updated reference** — a "variable in memory" — is sufficiently better than these alternatives in the running-state case to justify its existence.

---

## What Lives in Each Half

A rough rule of thumb for "is this a record or a reference":

```
write as a RECORD when:                 write as a REFERENCE when:
─────────────────────                   ──────────────────────────
something happened                      something is being maintained
a claim was made                        a value is being kept current
the timestamp matters                   the latest value matters
the history is part of the meaning      the history is an audit trail
```

If you find yourself wanting to "delete the old one" — that is a reference. References supersede; records accumulate. The supersede plumbing is shared between them; what changes is the default retrieval behavior and the rendering.

---

## Open Questions

These are not yet answered by the implementation and may motivate follow-up work:

- **How does a state_object get promoted between tiers?** A scratchpad in Tier 3 (RAM) that gets referenced for weeks should drift to Tier 2 (cache) on its own. A pinned plan in Tier 2 that hasn't been touched in months should drift to Tier 4. The tier should be a function of access pattern, not a fixed property. We do not yet have the promotion rules.
- **How are state_objects discovered by retrieval that doesn't know the `state_key`?** Right now the structured rendering goes into embeddings + BM25, so queries can find a todo list by its content. But should a query like "what am I working on" preferentially surface state_objects over episodic memories? Probably yes — references describe the present, records describe the past. The retrieval scoring needs a small bias term.
- **What happens when two integrators write to the same `state_key`?** Last-write-wins via the supersede chain. There is no merge. For Phase 1 this is fine because the integrator owns the key namespace; in Phase 2 this may need explicit conflict semantics.
- **Are state_objects portable across users?** The current schema scopes them to a user. A shared team todo list would need an additional scope (`team_id`?). Deferred until there's a user case.

---

## Tying Back

The original Flashback thesis was *dynamic RAG* — close the retrieval loop within a conversation so the index updates live. That thesis stands.

This document adds a second thesis: **records and references are different kinds of memory, and the system needs both as first-class citizens.** Dynamic RAG without references handles the "what was just said" half of memory. References handle the "what am I currently maintaining" half.

If both halves hold up under real use, Flashback is not just a better search index — it is the first AI memory system with a heap.
