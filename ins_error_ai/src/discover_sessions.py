"""
discover_sessions.py
====================
Walk the IO-VNBD synchronised dataset directory tree and discover all
session pairs (S-file = smartphone IMU, V-file = vehicle CAN-bus ground truth).

The dataset is organised as:
    Categorised IOVNB Dataset/
        S (Driver A)/
            S1/  S-S1.csv   V-S1.csv   V-S1.JPG
            S2/  S-S2.csv   V-S2.csv   ...
        M (Driver B)/
            S-M.csv   V-M.csv    (flat, no subfolder)
        Vta (Driver E)/
            Vta01a/  S-Vta1a.csv  V-Vta1a.csv  ...

The naming is not perfectly consistent, so we match by:
    - Find all files starting with "S-" and ending with ".csv"
    - Find all files starting with "V-" and ending with ".csv" in the same dir
    - Pair them up
"""

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def discover_all_sessions(dataset_root: str = None) -> list[dict]:
    """
    Returns a list of dicts, each containing:
        {
            "name":   "S1",           # human-readable session name
            "s_path": "/abs/path/to/S-S1.csv",
            "v_path": "/abs/path/to/V-S1.csv",
            "category": "S (Driver A)"
        }
    """
    dataset_root = dataset_root or config.DATASET_ROOT
    if not os.path.isdir(dataset_root):
        print(f"[WARNING] Dataset root not found: {dataset_root}")
        return []

    sessions = []

    for root, dirs, files in os.walk(dataset_root):
        s_files = [f for f in files if f.startswith("S-") and f.endswith(".csv")]
        v_files = [f for f in files if f.startswith("V-") and f.endswith(".csv")]

        if not s_files or not v_files:
            continue

        for s_name in s_files:
            # Extract session identifier: "S-S1.csv" -> "S1"
            s_id = s_name[2:-4]  # strip "S-" prefix and ".csv" suffix

            # Try to find matching V-file. Common patterns:
            #   S-S1.csv <-> V-S1.csv
            #   S-Vta1a.csv <-> V-Vta1a.csv
            #   S-M.csv <-> V-M.csv
            v_name = f"V-{s_id}.csv"
            if v_name not in v_files:
                # Try case-insensitive match
                v_match = [v for v in v_files if v.lower() == v_name.lower()]
                if v_match:
                    v_name = v_match[0]
                else:
                    # If only one V-file in this dir, assume it's the match
                    if len(v_files) == 1 and len(s_files) == 1:
                        v_name = v_files[0]
                    else:
                        print(f"[skip] No matching V-file for {s_name} in {root}")
                        continue

            # Determine category from the relative path
            rel = os.path.relpath(root, dataset_root)
            category = rel.split(os.sep)[0] if rel != "." else "uncategorised"

            sessions.append({
                "name": s_id,
                "s_path": os.path.join(root, s_name),
                "v_path": os.path.join(root, v_name),
                "category": category,
            })

    # Sort by name for reproducibility
    sessions.sort(key=lambda x: x["name"])
    return sessions


if __name__ == "__main__":
    sessions = discover_all_sessions()
    print(f"Discovered {len(sessions)} session pairs:\n")
    for s in sessions:
        print(f"  [{s['category']}] {s['name']}")
        print(f"    S: {s['s_path']}")
        print(f"    V: {s['v_path']}")
        print()
