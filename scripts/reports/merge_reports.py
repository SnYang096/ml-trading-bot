import os
from PyPDF2 import PdfMerger

REPORTS_DIR = "reports"
SRC1 = os.path.join(REPORTS_DIR, "june_oos_summary.pdf")
SRC2 = os.path.join(REPORTS_DIR, "oos_summary_5T_15T_60T_240T.pdf")
OUT = os.path.join(REPORTS_DIR, "june_oos_summary.pdf")


def main():
    merger = PdfMerger()
    added = False
    if os.path.exists(SRC1):
        merger.append(SRC1)
        added = True
    if os.path.exists(SRC2):
        merger.append(SRC2)
        added = True
    if not added:
        print("No source PDFs found to merge")
        return
    merger.write(OUT)
    merger.close()
    print("Merged into", OUT)


if __name__ == "__main__":
    main()
