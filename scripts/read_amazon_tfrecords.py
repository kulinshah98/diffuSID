#!/usr/bin/env python3
"""
Read GZIP-compressed TFRecord (.tfrecord.gz) files under a directory using TensorFlow.

The script will:
- find all files matching *.tfrecord*.gz under the target directory
- for each file, attempt to iterate records (optionally up to a limit)
- parse records as tf.train.Example when possible and print feature keys and a small preview

This requires TensorFlow to be installed (the workspace 'goatee' environment has TensorFlow).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable


def find_tfrecord_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.tfrecord*") if p.is_file()])


def preview_tfrecord_file(path: Path, sample: int = 2, max_count: int | None = 1000) -> None:
    import tensorflow as tf

    print(f"\nFILE: {path} — size={path.stat().st_size} bytes")
    ds = tf.data.TFRecordDataset(str(path), compression_type='GZIP')

    # show a small sample parsed as Example
    ds2 = tf.data.TFRecordDataset(str(path), compression_type='GZIP')
    for i, raw in enumerate(ds2.take(sample)):
        print(f"  Example #{i} — raw bytes: {len(raw.numpy())} bytes")
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        keys = list(ex.features.feature.keys())
        print(f"    parsed as tf.train.Example with keys: {keys}")
        for k in keys[:10]:
            f = ex.features.feature[k]
            if f.bytes_list.value:
                val = f.bytes_list.value[:1]
                # display as utf-8 when possible
                try:
                    pretty = [v.decode('utf-8', errors='replace') for v in val]
                except Exception:
                    pretty = [str(v) for v in val]
                print(f"      {k}: bytes_list len={len(f.bytes_list.value)} preview={pretty}")
            if f.int64_list.value:
                print(f"      {k}: int64_list len={len(f.int64_list.value)} preview={list(f.int64_list.value)[:5]}")
            if f.float_list.value:
                print(f"      {k}: float_list len={len(f.float_list.value)} preview={list(f.float_list.value)[:5]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./data/amazon_no_cap/beauty/training", help="Directory containing tfrecord.gz files")
    parser.add_argument("--sample", type=int, default=2, help="Number of examples to preview per file")
    parser.add_argument("--max-count", type=int, default=1000, help="Max records to iterate when counting (use 0 or -1 for unlimited)")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        print(f"Directory not found or not a directory: {root}")
        return 2

    files = find_tfrecord_files(root)
    if not files:
        print(f"No tfrecord files found under {root}")
        return 0

    max_count = None if args.max_count <= 0 else args.max_count

    for p in files:
        preview_tfrecord_file(p, sample=args.sample, max_count=max_count)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
