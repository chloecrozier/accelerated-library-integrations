"""
One-time prep for the cuGraph lateral-movement demo.

Turns the raw Los Alamos National Lab "Comprehensive, Multi-Source
Cyber-Security Events" files (auth.txt.gz + redteam.txt.gz, CC0) into a
single, manageable CSV that examples/cyber_lateral_movement.py loads on the
GPU.

This script is stdlib-only on purpose: it does NOT need RAPIDS and can run on
any machine (including the laptop you downloaded the data on). Heavy GPU work
happens later, in the demo.

Raw formats (comma-separated, no header):

  auth.txt:    time, src_user@domain, dst_user@domain, src_computer,
               dst_computer, auth_type, logon_type, auth_orientation, success
  redteam.txt: time, user@domain, src_computer, dst_computer

A redteam row labels exactly one authentication event. We match on
(time, user, src_computer, dst_computer): the redteam "user" is the source
user of the auth event. Every matched auth row is written with is_redteam=1
and is always kept. Normal (un-flagged) rows are down-sampled so the output
stays small enough to push onto the GPU, while still giving the red-team hosts
realistic surrounding traffic to build a graph from.

Output columns (is_redteam is the 10th column):

  time, src_user, dst_user, src_computer, dst_computer,
  auth_type, logon_type, auth_orientation, success, is_redteam

Usage:

  python dataset/prepare_lanl.py \
      --auth    /path/to/auth.txt.gz \
      --redteam /path/to/redteam.txt.gz
"""

import argparse
import csv
import gzip
import sys
import zlib
from pathlib import Path


HEADER = [
    "time",
    "src_user",
    "dst_user",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "success",
    "is_redteam",
]


def open_text(path):
    """Open a plain or gzipped text file transparently."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def load_redteam(path):
    """Return (key_set, t_min, t_max).

    key is "time,user,src_computer,dst_computer" -- the same tuple we can
    reconstruct from an auth row, so membership testing is a single dict/set
    lookup per auth line.
    """
    keys = set()
    t_min = None
    t_max = None
    with open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            t, user, src, dst = parts
            keys.add(f"{t},{user},{src},{dst}")
            t_int = int(t)
            t_min = t_int if t_min is None else min(t_min, t_int)
            t_max = t_int if t_max is None else max(t_max, t_int)
    return keys, t_min, t_max


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", required=True,
                    help="path to raw auth.txt or auth.txt.gz")
    ap.add_argument("--redteam", required=True,
                    help="path to raw redteam.txt or redteam.txt.gz")
    ap.add_argument("--out", default=str(here / "lanl_auth_sample.csv"),
                    help="output CSV path (default: dataset/lanl_auth_sample.csv)")
    ap.add_argument("--max-normal", type=int, default=400_000,
                    help="cap on non-red-team rows to keep (default: 400000)")
    ap.add_argument("--stride", type=int, default=400,
                    help="keep 1 in every STRIDE in-window normal rows "
                         "(default: 400)")
    args = ap.parse_args()

    auth_path = Path(args.auth)
    redteam_path = Path(args.redteam)
    out_path = Path(args.out)
    for p in (auth_path, redteam_path):
        if not p.exists():
            sys.exit(f"Input not found: {p}")

    print(f"Reading red-team labels from {redteam_path} ...")
    red_keys, red_min, red_max = load_redteam(redteam_path)
    if not red_keys:
        sys.exit("No red-team rows parsed -- is the redteam file the right format?")
    print(f"  red-team events: {len(red_keys):,}")
    print(f"  red-team time window: t={red_min:,} .. t={red_max:,}")
    print()
    print(f"Streaming auth events from {auth_path} ...")
    print("  keeping every red-team match + 1 in every "
          f"{args.stride} in-window normal rows (cap {args.max_normal:,})")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    red_kept = 0
    normal_kept = 0
    normal_candidates = 0
    lines_read = 0
    truncated = False

    with open_text(auth_path) as fh, \
            open(out_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(HEADER)
        try:
            for line in fh:
                lines_read += 1
                if lines_read % 5_000_000 == 0:
                    print(f"  ...{lines_read:,} lines read "
                          f"(red-team kept {red_kept:,}, "
                          f"normal kept {normal_kept:,})")

                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 9:
                    continue

                t, src_user, dst_user, src, dst = parts[:5]
                key = f"{t},{src_user},{src},{dst}"

                if key in red_keys:
                    writer.writerow(parts + ["1"])
                    red_kept += 1
                    continue

                # Down-sample normal traffic, restricted to the red-team
                # time window so the kept context is temporally relevant.
                try:
                    t_int = int(t)
                except ValueError:
                    continue
                if not (red_min <= t_int <= red_max):
                    continue

                # Sample against a counter of eligible normal rows so the
                # kept-row rate stays at 1/stride regardless of how red-team
                # density or the time window interact with the modulus.
                normal_candidates += 1
                if normal_kept >= args.max_normal:
                    continue
                if normal_candidates % args.stride == 0:
                    writer.writerow(parts + ["0"])
                    normal_kept += 1
        except (EOFError, OSError, zlib.error, gzip.BadGzipFile) as exc:
            # A partial / truncated .gz (e.g. an interrupted download) raises
            # here. That's fine for a demo as long as some red-team rows were
            # captured before the stream ended.
            truncated = True
            print(f"\n  note: hit end of stream early ({type(exc).__name__}: "
                  f"{exc}); writing what was read so far.")

    print()
    print(f"Done. Read {lines_read:,} auth lines"
          f"{' (stream truncated)' if truncated else ''}.")
    print(f"  red-team rows kept: {red_kept:,} / {len(red_keys):,}")
    print(f"  normal rows kept:   {normal_kept:,}")
    print(f"  wrote: {out_path}")

    if red_kept == 0:
        sys.exit(
            "\nNo red-team rows were captured. The auth file's time range "
            "probably does not overlap the red-team window "
            f"(t={red_min:,}..{red_max:,}). Use a fuller auth.txt.gz."
        )


if __name__ == "__main__":
    main()
