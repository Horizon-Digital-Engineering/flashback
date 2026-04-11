# Flashback — The Dynamic RAG Thesis

*Why AI memory is broken, and what it would mean to fix it.*

---

## The Problem With How Memory Works Today

Every "memory solution" for LLMs is, under the hood, the same thing: a vector store with a thin wrapper. You take documents or conversation snippets, chunk them, embed them, and store the vectors. When the user says something, you embed the query and retrieve the top-k nearest chunks. Then you stuff those chunks into the context window alongside the user's message.

This is called RAG. It works. It's useful. And it's fundamentally the wrong model for conversational memory.

The core issue is that it's **static**. The knowledge base is loaded once. You retrieve from it, but you never update it *during* the conversation. The retrieval results for your first message and your tenth message are drawn from the same frozen index. The system has no idea that you spent the last nine messages establishing context, making decisions, and narrowing the problem space. It doesn't know what just happened.

That's not how memory works.

---

## Human Memory as the Model

When you're talking with someone and they mention a project name, your brain doesn't pause and run a database query. Something more interesting happens: the mention *activates* a web of associated memories, decisions, and context — nearly instantly. You remember not just the fact, but the story around it. The time you almost chose something different. The tradeoff you made. The conversation where you landed on this approach.

And critically — as the conversation continues, your brain is continuously updating. Three sentences in, you've already filed away that this person cares about X and doesn't know about Y. Your retrieval is *live*. It's shifting with each new sentence.

This maps to a four-tier model:

| Tier | Human Analogy | System Layer | Characteristics |
|------|---------------|--------------|-----------------|
| **ROM** | Subconscious / instinct | LLM training data | Baked-in. Immutable. No retrieval cost. |
| **Cache** | Trained instincts / core facts | Core memory (always loaded) | Permanent, zero-latency, always in context. |
| **RAM** | Working memory | Current conversation context | Fast decay. Tied to the active session. |
| **Disk** | Long-term episodic recall | Archival memory layer | Searchable on demand. Survives across sessions. |

Computer architects will recognize this: registers, L1 cache, RAM, disk, firmware. The same tradeoffs that hardware designers solved decades ago — latency vs. capacity, volatile vs. persistent — apply directly to how an AI system should manage what it knows.

Most memory systems today collapse this entire hierarchy into a single tier: the disk. There's no cache. There's no RAM. There's no concept of what's immediately relevant vs. what might be dug up later. Everything is equally distant.

---

## Why Static RAG Fails for Conversation

Consider a conversation that evolves like this:

1. "Help me debug this issue."
2. "I'm using the Acme framework — here's the error."
3. "Actually, this is the same pattern as last week's problem with the database layer."
4. "Right, and that was caused by the migration we ran in February."

At message 4, a static RAG system is still retrieving from an index that was built before the conversation started. It doesn't know that messages 2 and 3 exist as context. It can't connect message 4's reference to "the migration" with the established thread of the conversation — unless that context happened to be in a prior session that was already indexed.

A human with good memory would have been updating their mental model continuously. By message 3, they've already cross-referenced last week's problem. By message 4, they're reaching further back and surfacing the root cause.

**Dynamic RAG** is the difference. The retrieval index is live. Each exchange updates working memory. The next query runs against an index that includes what was just said.

The flow isn't:
> Query memory → Talk → Ingest after conversation ends

It's:
> User says something → Query memory → LLM responds → Ingest that exchange *immediately* → Next message retrieves against an index that now includes what was just established

The context window is alive. Not a snapshot.

---

## The Episodic Insight

There's a subtler problem with fact-based memory systems: they treat memory as a database of *current facts*.

Most memory systems delete old facts when new ones supersede them. Others store a small static notepad of current preferences. Most append-only approaches have no retrieval intelligence.

But human memory isn't a database of current facts. It's a **web of episodes**.

You don't just know that you use a particular tool. You remember the moment you chose it. The alternatives you considered. The argument you had with yourself about the tradeoffs. The version you tried first that didn't work. The pivot you made six months in.

Dead ends matter. Pivots matter. The decision trail is part of the memory.

This is episodic memory — not "what is true now," but "what happened, in what order, and why." The narrative is the knowledge.

Flashback preserves this with an **append-with-supersede** model. Old facts aren't deleted. They're marked as superseded. A query for what's current returns the latest. A query for what happened returns the trail. You can ask "what do I know about X?" and get the current picture. You can also ask "how did my understanding of X evolve?" and get the history.

---

## What Flashback Actually Is

Flashback is a self-contained microservice that gives any LLM dynamic, episodic memory.

It is not a monolith. It is not a plugin that only works with one frontend or one model. It exposes a REST API, a gRPC interface, and an MCP server — the transport layer is a choice. The intelligence lives in the server: extraction, scoring, consolidation, temporal decay, the episodic graph. The client just feeds conversations and asks questions.

The design goal is that swapping in Flashback should be low-friction. If you're already doing naive RAG with a vector store, Flashback replaces the store and adds the intelligence layer. The client interface is simple. The complexity is internal.

What lives inside:
- A **vector index** for semantic similarity search across episodes
- A **temporal graph** that connects memories to time, source, and successor facts
- A **tiered memory manager** that handles working memory (per-session, fast-decay) and archival memory (cross-session, permanent)
- An **ingestion pipeline** that extracts facts, relationships, and decisions from raw conversation turns in real time
- A **consolidation process** that promotes patterns from working memory into long-term storage
- A **decay model** that reduces retrieval weight for stale working memories while preserving them for historical queries

None of these pieces are novel individually. The insight is in combining them and wiring them together in a way that makes retrieval *dynamic* rather than static.

---

## The Name

Mid-conversation, something triggers a memory. The system surfaces it — not because you queried for it explicitly, but because the current context activated it. A flashback.

That's the experience the system is trying to produce. Not "here are the top 5 relevant documents." Not "here's what you told me to remember." But: the right memory, at the right moment, because the context demanded it.

Human conversations aren't transactions. They're journeys. Flashback is memory that travels with you.

---

## The Thesis, in One Paragraph

Current AI memory systems are search indexes pretending to be brains. They're useful but fundamentally incomplete — static, flat, and amnesiac within a conversation. The right model is a four-tier hierarchy (ROM, cache, RAM, disk) where each tier has distinct latency, persistence, and decay characteristics. Within-conversation memory should be dynamic: ingested in real time, immediately retrievable, and decaying naturally as the conversation moves on. Long-term memory should be episodic: preserving not just current facts but the decisions and pivots that produced them. Flashback is an attempt to build that system — not as a research project, but as a practical microservice that any LLM application can adopt.
