"""
merge_all_nights_dat.py

Merges pipelineout_datasubset_all.dat from all available observation nights
for a given target into a single file: pipelineout_datasubset_all_nights.dat

The script auto-detects all night folders (format YY-MM-DD) under the target
folder and merges them in chronological order.

Usage:
    python merge_all_nights_dat.py --target <target_name>

Example:
    python merge_all_nights_dat.py --target 2001_EC

Expected folder structure:
    <main_folder>/
        <target_name>/
            26-03-01/
                pipelineout_datasubset_all.dat
            26-03-02/
                pipelineout_datasubset_all.dat
            ...
            pipelineout_datasubset_all_nights.dat  <- output written here
"""

import argparse
import os
import re
import sys

# -- Parameters ----------------------------------------------------------------

parser = argparse.ArgumentParser(description="Merge pipeline .dat files across all nights for a target.")
parser.add_argument("--target",   required=True, help="Target name (folder)")
parser.add_argument("--main_dir", default=".",   help="Path to main folder (default: current directory)")
args = parser.parse_args()

target_dir  = os.path.join(args.main_dir, args.target)
output_path = os.path.join(target_dir, "pipelineout_datasubset_all_nights.dat")

if not os.path.isdir(target_dir):
    sys.exit(f"ERROR: target directory not found: {target_dir}")

# -- Auto-detect night folders -------------------------------------------------

night_pattern = re.compile(r"^\d{2}_\d{2}_\d{2}$")
night_dirs = sorted([
    d for d in os.listdir(target_dir)
    if os.path.isdir(os.path.join(target_dir, d)) and night_pattern.match(d)
])

if not night_dirs:
    sys.exit(f"ERROR: no night folders (YY_MM_DD) found in:\n  {target_dir}")

# Check which nights have the merged file
input_files = []
missing     = []
for night in night_dirs:
    f = os.path.join(target_dir, night, "pipelineout_datasubset_all.dat")
    if os.path.isfile(f):
        input_files.append((night, f))
    else:
        missing.append(night)

print(f"\n{'='*55}")
print(f"  Found {len(input_files)} night(s) with merged data")
print(f"{'='*55}")
for night, f in input_files:
    print(f"  [{night}] {f}")
if missing:
    print(f"\n  WARNING: the following nights have no pipelineout_datasubset_all.dat")
    print(f"  and will be skipped (run merge_pipeline_dat.py for them first):")
    for night in missing:
        print(f"    - {night}")

if not input_files:
    sys.exit("ERROR: no input files found. Aborting.")

# -- Helpers -------------------------------------------------------------------

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
        elif stripped.strip():
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

# -- Read & validate all night files -------------------------------------------

print(f"\n{'='*55}")
print(f"  Validation - individual night files")
print(f"{'='*55}")

all_errors = []
ref_header = None
all_rows   = []  # list of (night, rows, row_count)

for night, path in input_files:
    header, rows = read_dat(path)
    label = f"Night {night}"

    if ref_header is None:
        ref_header = header
    elif header != ref_header:
        print(f"  WARNING: header of night {night} differs from {input_files[0][0]}!")
        print(f"    Expected : {ref_header}")
        print(f"    Got      : {header}")

    # Check contiguity within the night file
    indices = [get_index(r) for r in rows]
    errors  = []
    if indices[0] != 1:
        errors.append(f"{label} does not start at 1 (starts at {indices[0]})")
    for i, (a, b) in enumerate(zip(indices, indices[1:]), start=1):
        if b != a + 1:
            errors.append(f"{label}: gap between rows {i} and {i+1} (indices {a} -> {b})")
    all_errors.extend(errors)

    if not errors:
        print(f"  v {label}: {len(rows)} rows  (indices {indices[0]} -> {indices[-1]})")

    all_rows.append((night, rows, len(rows)))

if all_errors:
    print("  x Issues found:")
    for e in all_errors:
        print(f"    - {e}")
    sys.exit("Aborting merge due to validation errors in input files.")

# -- Reindex and merge ---------------------------------------------------------

merged_rows = []
offset      = 0

for night, rows, n in all_rows:
    for i, row in enumerate(rows):
        merged_rows.append(reindex_row(row, offset + i + 1))
    offset += n

# -- Write output --------------------------------------------------------------

with open(output_path, "w") as f:
    f.write(ref_header + "\n")
    for row in merged_rows:
        f.write(row + "\n")

# -- Validate merged file ------------------------------------------------------

print(f"\n{'='*55}")
print(f"  Validation - merged file")
print(f"{'='*55}")

_, merged_check = read_dat(output_path)
midx            = [get_index(r) for r in merged_check]
expected_total  = sum(n for _, _, n in all_rows)
counts_str      = " + ".join(f"{n} ({night})" for night, _, n in all_rows)
merge_errors    = []

if len(merged_check) != expected_total:
    merge_errors.append(f"Row count mismatch: expected {expected_total}, got {len(merged_check)}")
if midx[0] != 1:
    merge_errors.append(f"Merged file does not start at 1 (starts at {midx[0]})")
if midx[-1] != expected_total:
    merge_errors.append(f"Last index is {midx[-1]}, expected {expected_total}")
for i, (a, b) in enumerate(zip(midx, midx[1:]), start=1):
    if b != a + 1:
        merge_errors.append(f"Gap in merged file between rows {i} and {i+1} (indices {a} -> {b})")

# Check every seam between consecutive nights
seam_pos = 0
for idx, (night, rows, n) in enumerate(all_rows[:-1]):
    seam_pos  += n
    seam_last  = get_index(merged_rows[seam_pos - 1])
    seam_first = get_index(merged_rows[seam_pos])
    next_night = all_rows[idx + 1][0]
    if seam_first != seam_last + 1:
        merge_errors.append(
            f"Seam between {night} and {next_night}: "
            f"{seam_last} -> {seam_first} (not continuous)"
        )
    else:
        print(f"  v Seam {night} -> {next_night}  : ...{seam_last} | {seam_first}...")

if merge_errors:
    print("  x Issues found in merged file:")
    for e in merge_errors:
        print(f"    - {e}")
else:
    print(f"  v Total rows : {len(merged_check)}  (= {counts_str})")
    print(f"  v Index range: {midx[0]} -> {midx[-1]}  (continuous)")

print(f"\n  Output written to:\n  {output_path}\n")

# -- Export last 3 columns to .txt ---------------------------------------------

txt_path = os.path.join(target_dir, "data_all.txt")

with open(txt_path, "w") as f:
    for row in merged_rows:
        cols = row.split("\t")
        f.write("\t".join(cols[-3:]) + "\n")

print(f"  3-column txt written to:\n  {txt_path}\n")