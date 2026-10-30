"""Inspect column structure of all raw Excel files."""
import os, openpyxl

RAW = os.path.join(os.path.dirname(__file__), "raw")

for fname in sorted(os.listdir(RAW)):
    if not fname.endswith(".xlsx"):
        continue
    path = os.path.join(RAW, fname)
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            rows = list(ws.iter_rows(max_row=4, values_only=True))
            print(f"\n{fname}  [sheet: {sheet}]")
            for i, row in enumerate(rows):
                print(f"  row{i}: {row}")
        wb.close()
    except Exception as e:
        print(f"\n{fname}  ERROR: {e}")
