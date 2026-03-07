"""
merge_pipeline_dat.py

Merges any number of pipelineout_N_datasubset.dat files into
pipelineout_datasubset_all.dat, renumbering rows continuously across all files.

The script auto-detects all pipelineout_N_datasubset.dat files present in the
night directory and merges them in numerical order.

Usage:
    python merge_pipeline_dat.py --target <target_name> --night <observation_night>

Example:
    python merge_pipeline_dat.py --target 2001_EC --night 26_03_01

Expected folder structure:
    <main_folder>/
        <target_name>/
            <observation_night>/
                pipelineout_1_datasubset.dat
                pipelineout_2_datasubset.dat
                pipelineout_3_datasubset.dat   <- any number of subsets
                ...
                pipelineout_datasubset_all.dat <- output written here
"""

import argparse
import glob
import os
import re
import sys

# -- Parameters ----------------------------------------------------------------

parser = argparse.ArgumentParser(description="Merge N pipeline .dat subsets.")
parser.add_argument("--target",   required=True, help="Target name (folder)")
parser.add_argument("--night",    required=True, help="Observation night date (folder)")
parser.add_argument("--main_dir", default=".",   help="Path to main folder (default: current directory)")
args = parser.parse_args()

night_dir   = os.path.join(args.main_dir, args.target, args.night)
output_path = os.path.join(night_dir, "pipelineout_datasubset_all.dat")

if not os.path.isdir(night_dir):
    sys.exit(f"ERROR: directory not found: {night_dir}")

# -- Auto-detect subset files --------------------------------------------------

pattern = os.path.join(night_dir, "pipelineout_*_datasubset.dat")
found   = glob.glob(pattern)

# Keep only files whose wildcard part is a plain integer (exclude 'all')
subset_files = {}
for path in found:
    fname = os.path.basename(path)
    m = re.match(r"pipelineout_(\d+)_datasubset\.dat$", fname)
    if m:
        subset_files[int(m.group(1))] = path

if not subset_files:
    sys.exit(f"ERROR: no pipelineout_N_datasubset.dat files found in:\n  {night_dir}")

ordered_keys  = sorted(subset_files)
ordered_paths = [subset_files[k] for k in ordered_keys]

print(f"\n{'='*55}")
print(f"  Found {len(ordered_paths)} subset file(s)")
print(f"{'='*55}")
for k, p in zip(ordered_keys, ordered_paths):
    print(f"  [{k}] {os.path.basename(p)}")

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

def validate_contiguous(indices, label):
    """Check that a list of indices starts at 1 and is fully contiguous."""
    errors = []
    if indices[0] != 1:
        errors.append(f"{label} does not start at 1 (starts at {indices[0]})")
    for i, (a, b) in enumerate(zip(indices, indices[1:]), start=1):
        if b != a + 1:
            errors.append(f"{label}: gap between rows {i} and {i+1} (indices {a} -> {b})")
    return errors

# -- Read & validate all source files ------------------------------------------

print(f"\n{'='*55}")
print(f"  Validation - original numbering")
print(f"{'='*55}")

all_errors  = []
ref_header  = None
all_rows    = []   # list of (file_number, rows_list, row_count)

for k, path in zip(ordered_keys, ordered_paths):
    header, rows = read_dat(path)
    label = f"File {k}"

    if ref_header is None:
        ref_header = header
    elif header != ref_header:
        print(f"  WARNING: header of file {k} differs from file {ordered_keys[0]}!")
        print(f"    Expected : {ref_header}")
        print(f"    Got      : {header}")

    indices = [get_index(r) for r in rows]
    errors  = validate_contiguous(indices, label)
    all_errors.extend(errors)

    if not errors:
        print(f"  v {label}: indices {indices[0]} -> {indices[-1]}  ({len(rows)} rows)")

    all_rows.append((k, rows, len(rows)))

if all_errors:
    print("  x Issues found:")
    for e in all_errors:
        print(f"    - {e}")
    sys.exit("Aborting merge due to validation errors in source files.")

# -- Reindex and merge ---------------------------------------------------------

merged_rows = []
offset      = 0

for k, rows, n in all_rows:
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

_, merged_check  = read_dat(output_path)
midx             = [get_index(r) for r in merged_check]
expected_total   = sum(n for _, _, n in all_rows)
counts_str       = " + ".join(str(n) for _, _, n in all_rows)
merge_errors     = []

if len(merged_check) != expected_total:
    merge_errors.append(f"Row count mismatch: expected {expected_total}, got {len(merged_check)}")
if midx[0] != 1:
    merge_errors.append(f"Merged file does not start at 1 (starts at {midx[0]})")
if midx[-1] != expected_total:
    merge_errors.append(f"Last index is {midx[-1]}, expected {expected_total}")
for i, (a, b) in enumerate(zip(midx, midx[1:]), start=1):
    if b != a + 1:
        merge_errors.append(f"Gap in merged file between rows {i} and {i+1} (indices {a} -> {b})")

# Check every seam between consecutive files
seam_pos = 0
for idx, (k, rows, n) in enumerate(all_rows[:-1]):
    seam_pos   += n
    seam_last   = get_index(merged_rows[seam_pos - 1])
    seam_first  = get_index(merged_rows[seam_pos])
    next_k      = all_rows[idx + 1][0]
    if seam_first != seam_last + 1:
        merge_errors.append(
            f"Seam between file {k} and file {next_k}: "
            f"{seam_last} -> {seam_first} (not continuous)"
        )
    else:
        print(f"  v Seam file {k}->{next_k}  : ...{seam_last} | {seam_first}...")

if merge_errors:
    print("  x Issues found in merged file:")
    for e in merge_errors:
        print(f"    - {e}")
else:
    print(f"  v Total rows : {len(merged_check)}  (= {counts_str})")
    print(f"  v Index range: {midx[0]} -> {midx[-1]}  (continuous)")


# -- Export last 3 columns to .txt ---------------------------------------------

txt_path = os.path.join(night_dir, "data.txt")

with open(txt_path, "w") as f:
    for row in merged_rows:
        cols = row.split("\t")
        f.write("\t".join(cols[-3:]) + "\n")

print(f"\n  3-column txt written to:\n  {txt_path}\n")