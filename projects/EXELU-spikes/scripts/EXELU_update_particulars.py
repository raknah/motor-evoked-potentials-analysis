#!/usr/bin/env python3
"""
Write Animal_ID, genotype and comments into every session_particulars.txt.

Reads `mice-records` in the project root, joins it to the session directories on the
`filebase` column, and writes the three fields into each session's particulars file, so that
a recording on disk is self-describing and the analysis never has to reach back to a
spreadsheet.

Safe to run repeatedly: a key already present is replaced in place, never appended a second
time. A `.bak` copy of each file is made the first time only, so the pristine original is
never overwritten by a later run.

    python scripts/update_particulars.py --dry-run     # show what would change
    python scripts/update_particulars.py               # apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import EXELU_paths as paths

paths.add_script_paths()

import EXELU_records as records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args()

    data_root = arguments.data_root or paths.data_root()
    table = records.load_mice_records()

    print(f"mice-records : {paths.mice_records_path()}  ({len(table)} recordings)")
    print(f"data root    : {data_root}")
    print(f"fields       : {', '.join(records.MICE_RECORD_FIELDS)}")
    print()

    report = records.update_all_particulars(data_root, table, dry_run=arguments.dry_run)
    print(report.to_string(index=False))

    missing = report[report.status == "NOT IN mice-records"]
    if len(missing):
        print(f"\n{len(missing)} session(s) have no mice-records row — those stay unlabelled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
