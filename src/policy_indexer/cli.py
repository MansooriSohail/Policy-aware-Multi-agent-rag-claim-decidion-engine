from __future__ import annotations

import argparse
import json
from pathlib import Path

from policy_indexer.pipeline import build_policy_index


def main() -> None:
    parser = argparse.ArgumentParser(description='Index the health insurance policy PDF for retrieval.')
    parser.add_argument('--pdf', type=Path, required=True, help='Path to the policy PDF file.')
    parser.add_argument('--output-dir', type=Path, default=Path('data/indexes'), help='Directory to save chunk metadata and indexes.')
    args = parser.parse_args()

    result = build_policy_index(args.pdf, output_dir=args.output_dir)
    print(f'Built {result["chunk_count"]} chunks from {args.pdf}')
    print(f'Output written to {args.output_dir}')


if __name__ == '__main__':
    main()
