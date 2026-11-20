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

CSV files should contain at least one prompt-like column and one content-like column. Text-based source and documentation files are imported as reference material. Jupyter notebooks are split into cells. PDF files import selectable text page by page; scanned-image PDFs need OCR and are not currently supported. DOCX files import paragraphs and tables. PDF/DOCX support uses `pypdf` and `python-docx`, installed by `requirements.txt`.

## Recommended categories

- programming languages and software engineering
- mathematics, physics, chemistry and biology
- writing, reading and languages
- engineering, electronics and design
- research methods and factual knowledge
- debugging, reasoning and problem solving
- any other user-defined subject

Do not upload secrets, API keys, personal data or private player information.
