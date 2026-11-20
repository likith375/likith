# Dataset format

JSONL is the preferred interchange format: one JSON object per line.

## Fields

| Field | Required | Description |
|---|---:|---|
| prompt | yes | Instruction or task description |
| content | yes | Reference completion, code or example |
| category | no | Topic such as syntax, events, UI, services |
| split | no | train, validation or test |
| metadata | no | Any extra source or provenance fields |

The importer also accepts `instruction` / `title` as prompt aliases and `code` / `completion` / `output` / `text` as content aliases.

## Example

```json
{"prompt":"Create a part","category":"instances","content":"local part = Instance.new(\"Part\")","split":"train"}
```

CSV files should contain at least one prompt-like column and one content-like column.

## Recommended categories

- syntax and language basics
- services and instances
- events and connections
- player and character systems
- UI and input
- RemoteEvents and RemoteFunctions
- modules and architecture
- physics and movement
- data stores and persistence
- debugging and error handling
- performance and security

Do not upload secrets, API keys, personal data or private player information.
