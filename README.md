# idx-extract

STOXX Europe 600 index data extraction and membership computation pipeline.

## Overview

This project downloads quarterly STOXX 600 selection lists, extracts constituent data, computes index membership using the buffer rule, and stores the results as partitioned Parquet datasets in Cloudflare R2.

### Architecture

```
Download (stoxx.download)
    -> Parse CSV/PDF (stoxx.extract)
    -> Compute membership with buffer rule (stoxx.extract)
    -> Write Parquet dataset (stoxx.load)
    -> Upload to R2 (stoxx.storage)
```

### Data Pipeline

1. **Download** selection lists from stoxx.com (CSV from 2024+, PDF for historical)
2. **Extract** asset metadata and selection list entries
3. **Compute membership** using STOXX 600 buffer rule (top 550 auto-include, 551-750 buffer zone retains prior members, fill to 600)
4. **Write** partitioned Parquet datasets (assets, entries, membership by review_date)
5. **Sync** incrementally to Cloudflare R2 (only new periods)

## Quick Start

```bash
# Install dependencies
make install

# Run the sync pipeline (requires R2 credentials in environment)
uv run python -m stoxx.sync

# Or use the module entry point
uv run python -m stoxx
```

## Development

```bash
make install    # Setup environment
make test       # Run tests (90% coverage required)
make fmt        # Lint and format
make deptry     # Check dependencies
```

### Running a single test

```bash
uv run pytest tests/stoxx/test_sync.py -v
uv run pytest tests/stoxx/test_storage.py::TestListReviewDates -v
```

## Sync Schedule

The sync pipeline is orchestrated via [Prefect](https://www.prefect.io/). Deploy and run with `uv run python -m stoxx`.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `R2_ACCOUNT_ID` | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_BUCKET_NAME` | R2 bucket name |
