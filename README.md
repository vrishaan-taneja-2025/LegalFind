# LegalFind / CaseHold IR

LegalFind is a FastAPI interface for searching CaseHOLD holdings and citing prompts using persisted inverted indexes. The two retrieval methods are TF-IDF and BM25. It uses PostgreSQL when a configured server is reachable; otherwise it automatically uses a local SQLite database.

## Windows setup

1. Install Python 3.10 or newer.
2. (Optional) To use PostgreSQL, run it at `127.0.0.1:5433` and put `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` in `casehold-postgres.env`. Without a reachable configured PostgreSQL server, LegalFind uses `data/casehold.sqlite3`.
3. Run `setup.bat`. It installs dependencies, downloads the official parquet if missing, generates the JSONL export, creates or reuses the selected database tables, builds missing/incomplete inverted indexes, builds the Windows launcher if needed, and starts the site.

The executable is a launcher. It uses the project's `.venv` and selected database, so keep it in the `dist` folder and keep the project folder and `.venv` in place. With SQLite, keep `data/casehold.sqlite3` as well. `setup.bat` builds the Windows executable on Windows; it cannot produce a Windows `.exe` when run on Linux or macOS.

To open the interface manually after setup, run:

```bat
.venv\Scripts\python.exe -m uvicorn WebAccess.main:app --host 127.0.0.1 --port 8000
```

Then visit `http://127.0.0.1:8000`.

## Database setup

When using PostgreSQL, the app expects it at `127.0.0.1:5433` and reads these variables from the root `casehold-postgres.env` file:

- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`

The dataset downloader uses the official [CaseHOLD Hugging Face dataset](https://huggingface.co/datasets/casehold/casehold), configuration `all`, split `train`. Run `python -m retrieval.download_dataset --force` to replace the local file manually. The resulting `data/casehold-train.parquet` is ignored by Git because it is downloaded data.

The setup script does not install or start PostgreSQL, or create a PostgreSQL database/user. It creates the project's tables and indexes in the selected database. Its initialization path preserves existing rows and refuses to overwrite mismatched or partial CaseHOLD tables. SQLite data and indexes are stored locally in `data/casehold.sqlite3`. For a deliberate PostgreSQL full reset, `python -m retrieval.rebuild_database` drops and recreates CaseHOLD tables; indexes can then be rebuilt with `python -m retrieval.rebuild_all_indexes`.

## Git hygiene

Local credentials, Python environments, generated datasets, caches, logs, and Windows build output are excluded by `.gitignore`. Do not commit real database credentials. The dataset is not included in this repository; follow its source license and distribution terms when obtaining it.
