#!/usr/bin/env python3
"""
Create a downsampled copy of TFRecord training data where each Example's
`sequence_data` is reduced by 37.5% (i.e. we keep 62.5% of elements), but
we always keep at least 3 elements (and never more than the original length).

Output files are written to a sibling `training_37p5` directory (by default).

Usage:
  python mycode/scripts/create_training_37p5.py --src mycode/data/amazon_no_cap/beauty/training

Options include --dst, --seed, --limit-files, --max-examples-per-file, --dry-run
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import sys
from typing import Iterable

total_count = 0


def find_tfrecord_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.tfrecord*") if p.is_file()])


def compute_keep_count(n: int) -> int:
    # Keep 62.5% (0.625) rounded to nearest int, but at least 3, and no more than n
    if n <= 0:
        return 0
    k = int(round(0.375 * n))
    k = max(3, k)
    k = min(n, k)
    return k


def process_file(src: Path, dst: Path, rng: random.Random, sample_limit: int | None = None, dry_run: bool = False) -> tuple[int, int]:
    """Process a single TFRecord gz file: read examples, replace sequence_data, write to dst.

    Returns (written_examples, total_examples_seen)
    """
    import tensorflow as tf
    from google.protobuf import text_format

    dst.parent.mkdir(parents=True, exist_ok=True)

    # use GZIP compression for output
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = None
    written = 0
    seen = 0

    try:
        ds = tf.data.TFRecordDataset(str(src), compression_type='GZIP')
    except Exception as e:
        print(f"ERROR opening {src}: {e}")
        return 0, 0

    for i, raw in enumerate(ds):
        seen += 1
        if sample_limit is not None and seen > sample_limit:
            break

        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())

        fdict = ex.features.feature

        if 'sequence_data' in fdict:
            orig = list(fdict['sequence_data'].int64_list.value)
            n = len(orig)
            k = compute_keep_count(n)
            # sample indices without replacement and preserve original order
            idxs = sorted(rng.sample(range(n), k))
            new_vals = [orig[j] for j in idxs]
            # replace feature
            fdict['sequence_data'].int64_list.value[:] = new_vals
            # import pdb; pdb.set_trace()
            global total_count
            total_count += 1
        else:
            # no sequence_data key: leave example unchanged
            pass

        if dry_run:
            # don't write, just count
            written += 1
            continue

        if writer is None:
            writer = tf.io.TFRecordWriter(str(dst), options=options)

        writer.write(ex.SerializeToString())
        written += 1

    if writer is not None:
        writer.close()

    return written, seen


parser = argparse.ArgumentParser()
parser.add_argument("--src", default="./data/amazon_no_cap/beauty/training", help="Source training folder")
parser.add_argument("--dst", default="./data/amazon_no_cap/beauty/training_62p5", help="Destination folder")
parser.add_argument("--seed", type=int, default=1234, help="Random seed for sampling")
parser.add_argument("--limit-files", type=int, default=0, help="Process only the first N files (0 = all)")
parser.add_argument("--max-examples-per-file", type=int, default=0, help="Max examples to read per file (0 = all)")
parser.add_argument("--dry-run", action='store_true', help="Don't write output, just show counts")
args = parser.parse_args()

src = Path(args.src)
dst_base = Path(args.dst)

files = find_tfrecord_files(src)

limit_files = None if args.limit_files <= 0 else args.limit_files
max_examples = None if args.max_examples_per_file <= 0 else args.max_examples_per_file

rng = random.Random(args.seed)

total_written = 0
total_seen = 0
processed = 0

for idx, p in enumerate(files):
    if limit_files is not None and idx >= limit_files:
        break

    rel = p.relative_to(src)
    out_path = dst_base / rel
    out_path_parent = out_path.parent
    out_path_parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {p} -> {out_path}")
    print(total_count, "examples modified so far")
    written, seen = process_file(p, out_path, rng, sample_limit=max_examples, dry_run=args.dry_run)
    print(f"  seen={seen}, written={written}")
    total_written += written
    total_seen += seen
    processed += 1

print("Total count of modified examples:", total_count)
print(f"Finished. Files processed: {processed}, examples seen: {total_seen}, examples written: {total_written}")