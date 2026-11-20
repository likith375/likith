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

The importer accepts `instruction`, `question`, `input`, `user`, `title`, and `name` as prompt aliases; content aliases include `code`, `completion`, `output`, `response`, `answer`, `target`, `assistant`, and `text`. OpenAI-style `messages` and ShareGPT-style `conversations` rows are converted into prompt/completion examples.

## Example

```json
{"prompt":"Create a part","category":"instances","content":"local part = Instance.new(\"Part\")","split":"train"}
```

CSV files should contain at least one prompt-like column and one content-like column. Long prose documents are chunked on paragraph boundaries and all chunks from one document stay in the train split unless an explicit split is supplied. Text-based source and documentation files are imported as reference material. Jupyter notebooks are split into cells. PDF files import selectable text page by page; scanned-image PDFs need OCR and are not currently supported. DOCX files import paragraphs and tables. PDF/DOCX support uses `pypdf` and `python-docx`; XLSX uses `openpyxl` and PPTX uses `python-pptx`. These are installed by `requirements.txt`. XLSX rows with question/answer or prompt/completion columns become paired examples; other rows are preserved as labeled cell text. PPTX slide text and tables are imported slide by slide.

## Recommended categories

- programming languages and software engineering
- mathematics, physics, chemistry and biology
- writing, reading and languages
- engineering, electronics and design
- research methods and factual knowledge
- debugging, reasoning and problem solving
- any other user-defined subject

Do not upload secrets, API keys, personal data or private player information.
