"""
AI gen'd file name change.
Hard copy here for record or which names changed to which
"""

import os
import re

# Set to False when you're ready to perform actual file renames
DRY_RUN = False

# Target directory path (Change this to your actual folder path)
TARGET_DIR = r"/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/fiber_multi_cell"

# Original text data mapping PS IDs to raw lines
RAW_MAPPING_TEXT = """
PS01684|HG008-SC24 P15+9 Fiber-seq - Tube 1
PS01682|HG008-3E4 P15+12 Fiber-seq - Tube 1
PS01680|HG008-SC14 P20+15 Fiber-seq - Tube 1
PS01524|GM27730 (iPSC) Fiber-seq (Coriell) - rep 1
PS01520|GM25455 (LCL) Fiber-seq (Coriell) - rep 1
PS01519|GM25456 (Fibroblast) Fiber-seq (Coriell) - rep 1
PS01518|GM28570 (LCL) Fiber-seq (Coriell) - rep 1
PS01517|GM28572 (Fibroblast) Fiber-seq (Coriell) - rep 1
PS01388|WTC-11 Fiber-seq (200U/1M cells)
PS01376|H9 Fiber-seq (200U/1M cells)
PS01375|H9 Fiber-seq (100U/1M cells)
PS01373|H1 Fiber-seq (200U/1M cells)
PS01372|H1 Fiber-seq (100U/1M cells)
PS01370|K562 Fiber-seq (200U/1M cells)
PS01369|K562 Fiber-seq (100U/1M cells)
PS01367|Hap1 Fiber-seq (200U/1M cells)
PS01366|Hap1 Fiber-seq (100U/1M cells)
PS01332|Hek293T Fiber-seq (200U/1M cells)
PS01329|Jurkat Fiber-seq (200U/1M cells)
PS01328|Jurkat Fiber-seq (100U/1M cells)
PS01326|THP-1 Fiber-seq (200U/1M cells)
PS01325|THP-1 Fiber-seq (100U/1M cells)
PS01319|Caco-2 Fiber-seq (100U/1M cells)
PS01317|Panc1 Fiber-seq (200U/1M cells)
PS01316|Panc1 Fiber-seq (100U/1M cells)
PS01314|HCT116 Fiber-seq (200U/1M cells)
PS01308|HepG2 Fiber-seq (200U/1M cells)
PS01307|HepG2 Fiber-seq (100U/1M cells)
PS01305|MCF-7 Fiber-seq (200U/1M cells)
PS01302|A549 Fiber-seq (200U/1M cells)
PS01280|HAP1 Fiber-seq
"""


def parse_mappings(raw_text):
    """Parses text map and builds safe, space-free prefixes for each PS ID."""
    mapping = {}
    for line in raw_text.strip().split("\n"):
        if not line or "|" not in line:
            continue
        ps_id, desc = line.split("|", 1)
        ps_id = ps_id.strip()

        # Check for enzyme concentration label (e.g., 100U or 200U)
        unit_match = re.search(r"(\d+U)", desc)
        unit_label = unit_match.group(1) if unit_match else ""

        # Extract cell type identifier
        if "HG008-" in desc:
            cell_type = re.search(r"(HG008-[^\s]+)", desc).group(1)
        elif "(" in desc:
            base_cell = desc.split("(")[0].strip()
            type_in_paren = re.search(r"\((.*?)\)", desc).group(1)
            cell_type = (
                f"{base_cell}_{type_in_paren}"
                if "Coriell" not in type_in_paren
                else base_cell
            )
        else:
            cell_type = desc.split()[0]

        # Clean spaces, hyphens, and punctuation to make filenames safe
        cell_type = re.sub(r"[-\s\W]+", "_", cell_type).strip("_")

        # Assemble new prefix: CellType_Units or CellType
        prefix = f"{cell_type}_{unit_label}" if unit_label else cell_type
        mapping[ps_id] = prefix

    return mapping


def rename_files(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    mapping = parse_mappings(RAW_MAPPING_TEXT)
    renamed_count = 0
    skipped_count = 0

    print(f"--- Running in {'DRY RUN' if DRY_RUN else 'LIVE'} Mode ---\n")

    for filename in sorted(os.listdir(directory)):
        match = re.match(r"^(PS\d+)", filename)
        if match:
            ps_id = match.group(1)
            if ps_id in mapping:
                prefix = mapping[ps_id]

                # Avoid double renaming
                if filename.startswith(f"{prefix}_"):
                    print(f"[SKIPPED - Already Renamed] {filename}")
                    skipped_count += 1
                    continue

                new_filename = f"{prefix}_{filename}"
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)

                if DRY_RUN:
                    print(
                        f"[WOULD RENAME] {filename}\n         ---> {new_filename}\n"
                    )
                else:
                    os.rename(old_path, new_path)
                    print(f"[RENAMED] {filename} -> {new_filename}")

                renamed_count += 1
            else:
                print(f"[WARNING] Unmapped PS ID in file: {filename}")

    print(
        f"Completed! Processed {renamed_count} files ({skipped_count} skipped)."
    )
    if DRY_RUN:
        print("Set `DRY_RUN = False` in the script to execute the actual changes.")


if __name__ == "__main__":
    rename_files(TARGET_DIR)
