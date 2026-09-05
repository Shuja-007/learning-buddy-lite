# Private input directory

Put the PDF files you want to index in this directory, then run:

```powershell
uv run python Database/embedding_pipeline.py
```

PDF files in this directory are deliberately ignored by Git. Do not add
private documents to a hackathon upload. Generated embeddings are also kept
local in `Database/chroma_db/`.
