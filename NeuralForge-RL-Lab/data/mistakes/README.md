# Mistake memory

NeuralForge writes an append-only JSONL audit log to `data/mistakes/mistakes.jsonl` as the model makes mistakes and as a human reviews them.

The SQLite database is the queryable source of truth. The JSONL file is an additional durable audit trail. Original attempts are never overwritten. Review decisions are appended as separate events.

Only corrections whose latest review status is `verified` are used as examples in future task prompts. Review every correction carefully before verifying it.
