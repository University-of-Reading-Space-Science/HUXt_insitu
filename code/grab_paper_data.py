"""Download the public upstream data used by the paper analysis.

The large, processed InSitu-HUXt and WSA-HUXt lead-time CSVs are distributed
with this repository.  This command fetches the remaining reproducible input:
hourly OMNI data from NASA CDAWeb.  The Richardson-Cane ICME catalogue used by
the paper is also distributed in ``data/`` so the exact reviewed event list is
preserved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import huxt_inputs as Hin  # noqa: E402


def grab_omni(start: dt.datetime, stop: dt.datetime, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"Already present: {output} (use --force to replace)")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hourly OMNI data for {start:%Y-%m-%d} to {stop:%Y-%m-%d}")
    omni = Hin.get_omni(start, stop)
    if omni.empty:
        raise RuntimeError("CDAWeb returned no OMNI records")
    omni.to_csv(output, index=False)
    # Re-read the artifact so malformed/truncated output fails immediately.
    check = pd.read_csv(output, parse_dates=['datetime'])
    if check.empty or not {'datetime', 'mjd', 'V', 'BX_GSE'}.issubset(check.columns):
        raise RuntimeError(f"OMNI cache failed validation: {output}")
    print(f"Wrote {len(check):,} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=dt.datetime.fromisoformat,
                        default=dt.datetime(2007, 12, 4))
    parser.add_argument('--stop', type=dt.datetime.fromisoformat,
                        default=dt.datetime(2023, 1, 28))
    parser.add_argument('--output', type=Path,
                        default=REPO_ROOT / 'data' / 'omni_1hour_2007_2023.csv')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    grab_omni(args.start, args.stop, args.output.resolve(), args.force)


if __name__ == '__main__':
    main()
