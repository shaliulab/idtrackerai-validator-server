"""
download_metadata_batch.py  —  download_flyhostel_metadata for many experiments at once.

Run by metadata_sync.predownload with the Python of the environment where googleapi
is installed (next to download_flyhostel_metadata), not the server's: it must not
import anything from the server.

download_flyhostel_metadata downloads every sheet for each experiment (~10 s). Here
each sheet is downloaded once and then filtered per experiment with googleapi's own
functions, so every CSV is what download_flyhostel_metadata would have written.

stdin: {"outdir": ..., "experiments": [{"experiment", "hostel", "date_time", "arena"}]}
("arena": rows of that arena only, as download_flyhostel_metadata --arena; null for all)
Writes <outdir>/<experiment with _ for />.csv (none when nothing matched, as the
original for old v1/v2 sheets) and prints one line per experiment:
@@PROGRESS {"done", "total", "experiment", "error"}
"""
import os
import sys
import json

import pandas as pd
from googleapi.download.googlelink import googlelink, SHEETS
from googleapi.download.flyhostel import (
    choose_flyhostel_metadata_sheet, save_flyhostel_metadata, save_flyhostel_metadata_v3,
)

MARKER = "@@PROGRESS "


def main():
    request = json.load(sys.stdin)
    outdir = request["outdir"]
    experiments = request["experiments"]
    junk = os.path.join(outdir, "_per_sheet.csv")     # save_flyhostel_metadata_v3 writes each sheet's match here
    sheets = {}

    def sheet(spreadsheet_id, title):
        """The raw sheet, downloaded once (None when empty or not readable, as in googlelink)."""
        key = (spreadsheet_id, title)
        if key not in sheets:
            sheets[key] = googlelink(lambda data: data, spreadsheetId=spreadsheet_id, cell_range=None, sheet_title=title)
        return sheets[key]

    for i, experiment in enumerate(experiments):
        output = os.path.join(outdir, experiment["experiment"].replace("/", "_") + ".csv")
        date = experiment["date_time"][:10]
        arena = experiment.get("arena")
        error = None
        try:
            spreadsheet_id, version = choose_flyhostel_metadata_sheet(date)
            if version in ["v1", "v2"]:
                data = sheet(spreadsheet_id, date)
                if data is not None:
                    save_flyhostel_metadata(data.copy(), flyhostel_number=experiment["hostel"], date=date, output=output,
                                            arena=arena)
            else:
                parts = []
                for title in SHEETS:
                    data = sheet(spreadsheet_id, title)
                    parts.append(None if data is None else save_flyhostel_metadata_v3(
                        data.copy(), flyhostel_number=experiment["hostel"], date=date, output=junk, arena=arena))
                pd.concat(parts, axis=0).to_csv(output)
        except Exception as exc:     # one bad experiment (or sheet row) must not stop the others
            error = f"{type(exc).__name__}: {exc}"
        print(MARKER + json.dumps({"done": i + 1, "total": len(experiments),
                                   "experiment": experiment["experiment"], "error": error}), flush=True)


if __name__ == "__main__":
    main()
