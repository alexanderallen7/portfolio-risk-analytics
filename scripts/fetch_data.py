"""One-time (or refreshable) data pull.

Usage, from the repo root:

    python scripts/fetch_data.py            # pull only if cache incomplete
    python scripts/fetch_data.py --refresh  # force re-pull everything

After a successful run, ``data/raw/`` contains everything the analysis
needs and ``run_analysis.py`` works fully offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_fetch import fetch_and_cache_all  # noqa: E402


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Fetch and cache all raw data.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a full re-pull even if the cache is complete.",
    )
    args = parser.parse_args()

    provenance = fetch_and_cache_all(force_refresh=args.refresh)
    print("\nData sources used:")
    for key, src in provenance.items():
        print(f"  {key}: {src}")
    print("\nDone. Run `python scripts/run_analysis.py` next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
