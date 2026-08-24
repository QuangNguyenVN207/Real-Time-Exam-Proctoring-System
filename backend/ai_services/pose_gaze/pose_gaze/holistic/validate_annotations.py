"""Validate human gaze/temporal annotation CSV against a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from .annotation_schema import validate_annotation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allowed-splits", default="train,test")
    args = parser.parse_args()
    errors = validate_annotation_rows(args.annotations, args.manifest, allowed_splits=set(args.allowed_splits.split(",")))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"OK: {args.annotations}")


if __name__ == "__main__":
    main()
