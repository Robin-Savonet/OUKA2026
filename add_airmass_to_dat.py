import os
import glob
import numpy as np
from astropy.io import fits

# -------- Locate the .dat file --------
dat_files = glob.glob("*.dat")

if not dat_files:
    dat_files = glob.glob("report/*.dat")

if not dat_files:
    raise FileNotFoundError("No .dat file found in current directory or report/")

dat_path = dat_files[0]
print(f"Using dat file: {dat_path}")

# -------- Read file --------
with open(dat_path, "r") as f:
    lines = f.readlines()

header = lines[0].strip()

data_lines = lines[1:]

airmass_values = []

# -------- Process rows --------
for line in data_lines:
    parts = line.strip().split()

    if len(parts) < 2:
        airmass_values.append(np.nan)
        continue

    fits_name = parts[1]
    fits_path = os.path.join(".", fits_name)

    if not os.path.exists(fits_path):
        print(f"WARNING: FITS file not found: {fits_name}")
        airmass_values.append(np.nan)
        continue

    try:
        with fits.open(fits_path) as hdul:
            header_fits = hdul[0].header
            airmass = header_fits.get("AIRMASS", np.nan)

            if airmass is np.nan:
                print(f"WARNING: AIRMASS not found in header: {fits_name}")

            airmass_values.append(airmass)

    except Exception as e:
        print(f"WARNING: Could not read {fits_name}: {e}")
        airmass_values.append(np.nan)

# -------- Write updated file --------
new_lines = []
new_header = header + "\tairmass\n"
new_lines.append(new_header)

for line, am in zip(data_lines, airmass_values):
    new_line = line.strip() + f"\t{am}\n"
    new_lines.append(new_line)

with open(dat_path, "w") as f:
    f.writelines(new_lines)

print("AIRMASS column added successfully.")