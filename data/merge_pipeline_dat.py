"""
merge_pipeline_dat.py

Merges pipelineout_1_datasubset.dat and pipelineout_2_datasubset.dat into
pipelineout_datasubset_all.dat, renumbering rows continuously across both files.

Usage:
    python merge_pipeline_dat.py --target <target_name> --night <observation_night>

Example:
    python merge_pipeline_dat.py --target MyTarget --night 2024-07-15

Expected folder structure:
    <main_folder>/
        <target_name>/
            <observation_night>/
                pipelineout_1_datasubset.dat
                pipelineout_2_datasubset.dat
                pipelineout_datasubset_all.dat   ← output written here
"""

import argparse
import os
import sys

# ── Parameters ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Merge two pipeline .dat subsets.")
parser.add_argument("--target",    required=True, help="Target name (folder)")
parser.add_argument("--night",     required=True, help="Observation night date (folder)")
parser.add_argument("--main_dir",  default=".",   help="Path to main folder (default: current directory)")
args = parser.parse_args()

night_dir   = os.path.join(args.main_dir, args.target, args.night)
file1_path  = os.path.join(night_dir, "pipelineout_1_datasubset.dat")
file2_path  = os.path.join(night_dir, "pipelineout_2_datasubset.dat")
output_path = os.path.join(night_dir, "pipelineout_datasubset_all.dat")

# ── Helpers ────────────────────────────────────────────────────────────────────

def read_dat(path):
    """Return (header_line, data_rows) where data_rows is a list of raw lines."""
    with open(path, "r") as f:
        lines = f.readlines()
    header = None
    data   = []
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("#"):
            header = stripped
        elif stripped.strip():          # skip blank lines
            data.append(stripped)
    return header, data

def get_index(row):
    """Return the integer index (first tab-separated field) of a data row."""
    return int(row.split("\t")[0])

def reindex_row(row, new_index):
    """Replace the first field of a tab-separated row with new_index."""
    parts = row.split("\t")
    parts[0] = str(new_index)
    return "\t".join(parts)

# ── Read files ─────────────────────────────────────────────────────────────────

for p in (file1_path, file2_path):
    if not os.path.isfile(p):
        sys.exit(f"ERROR: file not found: {p}")

header1, rows1 = read_dat(file1_path)
header2, rows2 = read_dat(file2_path)

if header1 != header2:
    print(f"WARNING: headers differ!\n  File 1: {header1}\n  File 2: {header2}")

n1 = len(rows1)
n2 = len(rows2)

# ── Validate original numbering ────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Validation — original numbering")
print(f"{'='*55}")

errors = []

idx1 = [get_index(r) for r in rows1]
idx2 = [get_index(r) for r in rows2]

# File 1: must start at 1, be contiguous
if idx1[0] != 1:
    errors.append(f"File 1 does not start at 1 (starts at {idx1[0]})")
for i, (a, b) in enumerate(zip(idx1, idx1[1:]), start=1):
    if b != a + 1:
        errors.append(f"File 1: gap between rows {i} and {i+1} (indices {a} → {b})")

# File 2: must start at 1, be contiguous
if idx2[0] != 1:
    errors.append(f"File 2 does not start at 1 (starts at {idx2[0]})")
for i, (a, b) in enumerate(zip(idx2, idx2[1:]), start=1):
    if b != a + 1:
        errors.append(f"File 2: gap between rows {i} and {i+1} (indices {a} → {b})")

if errors:
    print("  ✗ Issues found in source files:")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"  ✓ File 1: indices {idx1[0]} → {idx1[-1]}  ({n1} rows)")
    print(f"  ✓ File 2: indices {idx2[0]} → {idx2[-1]}  ({n2} rows)")
    print(f"  ✓ Both files start at 1 and are contiguous.")

# ── Reindex and merge ──────────────────────────────────────────────────────────

merged_rows = rows1[:]                                    # File 1 keeps indices 1..n1
offset      = n1
for i, row in enumerate(rows2):
    merged_rows.append(reindex_row(row, offset + i + 1)) # File 2 gets n1+1..n1+n2

# ── Write output ───────────────────────────────────────────────────────────────

with open(output_path, "w") as f:
    f.write(header1 + "\n")
    for row in merged_rows:
        f.write(row + "\n")

# ── Validate merged file ───────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Validation — merged file")
print(f"{'='*55}")

_, merged_check = read_dat(output_path)
midx = [get_index(r) for r in merged_check]

merge_errors = []
expected_total = n1 + n2

if len(merged_check) != expected_total:
    merge_errors.append(f"Row count mismatch: expected {expected_total}, got {len(merged_check)}")
if midx[0] != 1:
    merge_errors.append(f"Merged file does not start at 1 (starts at {midx[0]})")
if midx[-1] != expected_total:
    merge_errors.append(f"Last index is {midx[-1]}, expected {expected_total}")
for i, (a, b) in enumerate(zip(midx, midx[1:]), start=1):
    if b != a + 1:
        merge_errors.append(f"Gap in merged file between rows {i} and {i+1} (indices {a} → {b})")

# Check the seam specifically
seam_last  = get_index(merged_rows[n1 - 1])
seam_first = get_index(merged_rows[n1])

if merge_errors:
    print("  ✗ Issues found in merged file:")
    for e in merge_errors:
        print(f"    - {e}")
else:
    print(f"  ✓ Total rows : {len(merged_check)}  (= {n1} + {n2})")
    print(f"  ✓ Index range: {midx[0]} → {midx[-1]}  (continuous)")
    print(f"  ✓ Seam check : last of file 1 = {seam_last}, "
          f"first of file 2 = {seam_first}  ✓")

print(f"\n  Output written to:\n  {output_path}\n")