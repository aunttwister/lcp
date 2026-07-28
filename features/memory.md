# Feature: Memory Layer (LanceDB Plugin)

**Created:** 2026-07-28
**Status:** in_progress
**Phase:** post-Phase 7

## North Star

A unified memory endpoint inside 5mall so every LLM client in the workflow
(Hermes agents, laptop tools, VPS scripts, custom integrations) reads and writes
to **one** memory bank through **one** API. No separate Hindsight service to discover.
No fragmented memory per device.

The memory backend runs **embedded, in-process** — same Docker container, same port,
same bind mount, same auth model as chat completions. Zero new infrastructure.

---

## Architecture

```
POST /{profile}/memory/retain   → store fact + embedding in LanceDB
POST /{profile}/memory/recall   → ANN search by embedding (optionally filtered by tags)
POST /{profile}/memory/forget   → remove entry by id
GET  /{profile}/memory/count    → return total stored facts
```

```
┌──────────────────────────────────────────────────────────┐
│  5mall (:8734)                                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Provider Cost│  │ Tool Stripper│  │ Memory Plugin │  │
│  │ Plugins      │  │ (existing)   │  │ (new)         │  │
│  └──────────────┘  └──────────────┘  └───────┬───────┘  │
│                                               │          │
│  ┌────────────────────────────────────────────▼───────┐  │
│  │  LanceDB (embedded, in-process)                    │  │
│  │  /app/data/memory/                                 │  │
│  │  ├── data/fragment_*.lance  (columnar vectors)     │  │
│  │  └── _indices/ivf_pq.idx    (ANN index)            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Embedding model (bge-small-en-v1.5, 384-dim)      │  │
│  │  Loaded once at startup, ~400MB RSS, ~10ms/embed   │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Plugin Contract

```python
class MemoryBackend(Protocol):
    def retain(self, content: str, metadata: dict | None = None,
               tags: list[str] | None = None) -> str:
        """Store a fact. Returns memory_id."""

    def recall(self, query_text: str, top_k: int = 10,
               tag_filter: list[str] | None = None) -> list[dict]:
        """Semantic search. Returns [{id, content, metadata, tags, score}]."""

    def forget(self, memory_id: str) -> bool:
        """Remove a memory. Returns True if deleted."""

    def count(self) -> int:
        """Total stored facts."""
```

## Backend: LanceDB

### Why LanceDB over sqlite-vec

| Concern | sqlite-vec | LanceDB |
|---------|-----------|---------|
| **Query @ 1M** | 4-60ms (alpha ANN) | <1ms (IVF_PQ, production) |
| **Insert speed** | 42+ min DiskANN build | Append-only fragments, fast |
| **ANN maturity** | Alpha (v0.1.10), compile-flag | Production (3+ years) |
| **Scaling ceiling** | Unknown >1M | Designed for billions |
| **Storage format** | Row-oriented b-tree | Columnar Lance (purpose-built) |

### Disk layout

```
/app/data/memory/                    ← one directory = one table
├── _versions/
│   └── 1.manifest
├── data/
│   ├── fragment_0.lance             ← immutable columnar chunks
│   └── fragment_1.lance
└── _indices/
    └── ivf_pq.idx                   ← ANN index, built on demand
```

- Fragments are **immutable** — inserts create new fragments, never mutate existing ones
- No write locks — readers read existing fragments while writers append
- `docker cp -r` or `tar czf` for backup; bind-mount in compose

### Index strategy

| Scale | Index | Params |
|-------|-------|--------|
| <10K vectors | Flat (brute-force) | No index needed |
| 10K–100K | IVF_PQ | `num_partitions=16, num_sub_vectors=48` |
| 100K–1M | IVF_PQ | `num_partitions=128, num_sub_vectors=48` |
| 1M+ | IVF_PQ + GPU build | `num_partitions=256, accelerator="cuda"` |

## Config

```yaml
# gateway.yaml — new top-level section
plugins:
  memory:
    enabled: true
    backend: lancedb
    storage_path: /app/data/memory
    embedding:
      model: BAAI/bge-small-en-v1.5
      dim: 384
      device: cpu          # cpu | cuda
    ann:
      index: IVF_PQ
      num_partitions: 256
      num_sub_vectors: 48
```

## API detail

### POST /{profile}/memory/retain
```json
{
  "content": "node01 has an RTX 3090 and a Tesla P40 GPU",
  "tags": ["infrastructure", "hardware", "gpu"],
  "metadata": {"host": "node01", "ip": "10.0.0.1"}
}
→ {"memory_id": "abc123", "embedding_dim": 384}
```

### POST /{profile}/memory/recall
```json
{
  "query": "what GPUs are in node01",
  "top_k": 5,
  "tag_filter": ["infrastructure"]
}
→ {
  "results": [
    {
      "id": "abc123",
      "content": "node01 has an RTX 3090 and a Tesla P40 GPU",
      "score": 0.94,
      "tags": ["infrastructure", "hardware", "gpu"],
      "metadata": {"host": "node01"}
    }
  ],
  "query_ms": 3.2
}
```

### POST /{profile}/memory/forget
```json
{"memory_id": "abc123"}
→ {"deleted": true}
```

### GET /{profile}/memory/count
```json
→ {"count": 12453}
```

## Auth

Same as chat completions. If the profile requires `Authorization: Bearer <key>`,
memory endpoints require it too. The profile path determines which memory bank
is used — `/l2/memory/recall` queries the `l2` table, `/career/memory/recall`
queries `career`. This enables **per-profile memory isolation** via separate
LanceDB tables.

## Migration / Hindsight coexistence

- **Hindsight remains Hermes' memory** — it has entity graphs, cross-session
  synthesis, and `hindsight_reflect`. 5mall's memory layer does not replace it.
- **5mall memory is for everything else** — laptop scripts, VPS cron jobs,
  custom tools that hit the 5mall API for completions and now need memory too.
- **No migration needed.** They serve different clients and different use cases.
  If unification is desired later, Hindsight can be taught to read from 5mall's
  LanceDB via the plugin contract.

## Dependencies

```
# pyproject.toml additions
"lancedb>=0.15",
"sentence-transformers>=3.0",
```

Dockerfile: `pip install lancedb sentence-transformers` replaces ~400MB embedding
model pull at first run. Model cached in container image layer.

## Phases

### Phase 1: Core plugin (this PR)
- `src/plugins/` package with protocol definition
- `MemoryBackend` LanceDB implementation
- `memory_routes.py` HTTP handlers
- Wire into `server.py`
- Config in `gateway.yaml`
- Tests

### Phase 2: Index management
- Auto-detect when to build/re-train index (configurable threshold)
- `POST /memory/reindex` manual trigger
- Index stats endpoint

### Phase 3: Advanced features
- Hybrid search: semantic + BM25 full-text (Tantivy via LanceDB FTS)
- Time-decay scoring: recent memories boosted
- Memory consolidation: periodic dedup + summarization
- Tag auto-suggestion from content classification
