"""CLI for doc-search management."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from repowise.docs import db
from repowise.docs.bootstrap import seed_libraries

logger = logging.getLogger(__name__)


async def list_libraries_cmd() -> None:
    """List all libraries in the database."""
    await db.init_pool()
    libraries = await db.list_libraries()
    await db.close_pool()

    if not libraries:
        print("No libraries found.")
        return

    print(f"\n{'Library ID':<40} {'Name':<25} {'Status':<10} {'Chunks':>8}")
    print("-" * 90)
    for lib in libraries:
        print(f"{lib.library_id:<40} {lib.name:<25} {lib.status.value:<10} {lib.chunk_count:>8}")
    print(f"\nTotal: {len(libraries)} libraries")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="doc-search CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # seed command
    seed_parser = subparsers.add_parser(
        "seed",
        help="Seed libraries from libraries.yaml",
    )
    seed_parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Path to libraries.yaml (default: auto-detect)",
    )

    # list command
    subparsers.add_parser(
        "list",
        help="List all libraries in the database",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "seed":
        summary = asyncio.run(seed_libraries(args.file))
        print(
            "\nSeeded {upserted} libraries successfully "
            "({adopted_from_qdrant} adopted, {jobs_queued} queued).".format(**summary)
        )
    elif args.command == "list":
        asyncio.run(list_libraries_cmd())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
