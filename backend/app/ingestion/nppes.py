"""CLI entry point for NPPES CSV ingestion.

Run from backend/:
    uv run python -m app.ingestion.nppes --file <csv-path> --chunk-size 500

Only prints counts and status — never a full path, database URL, or
credential. Does not download the national NPPES dataset, start Docker, or
run database migrations; run `alembic upgrade head` yourself first.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.db.nppes import IngestionStatus
from app.ingestion.nppes_service import DEFAULT_CHUNK_SIZE, IngestionResult, ingest_nppes_file


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a small NPPES CSV file into the provider schema."
    )
    parser.add_argument("--file", required=True, help="Path to an NPPES CSV file.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per database batch (default: {DEFAULT_CHUNK_SIZE}).",
    )
    return parser.parse_args(argv)


def _print_summary(result: IngestionResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"rows_read: {result.rows_read}")
    print(f"rows_inserted: {result.rows_inserted}")
    print(f"rows_updated: {result.rows_updated}")
    print(f"rows_rejected: {result.rows_rejected}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.file)

    if not path.is_file():
        # Basename only — never the full path the caller supplied.
        print(f"Error: file not found or not a regular file: {path.name}", file=sys.stderr)
        return 2

    result = asyncio.run(ingest_nppes_file(path, chunk_size=args.chunk_size))
    _print_summary(result)

    return 1 if result.status == IngestionStatus.FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
