import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import os

RESULTS_DIR = os.path.join("results", "june_2025_oos")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "wavelet_5T_june_results.json")) as f:
    r5 = json.load(f)
with open(os.path.join(RESULTS_DIR, "wavelet_15T_june_results.json")) as f:
    r15 = json.load(f)

# Expect charts already generated externally; if not, skip embedding
imgs = []
for fn in ["june_oos_5t.png", "june_oos_15t.png"]:
    p = os.path.join("reports", fn)
    if os.path.exists(p):
        imgs.append(p)

pdf_path = os.path.join(REPORTS_DIR, "june_oos_summary.pdf")
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.clf()
    txt = (
        "June OOS Summary (Wavelet Model)\n\n"
        + f"5T: trades={r5['total_trades']}, win_rate={r5['win_rate']:.2f}%, return={r5['total_return']:.2f}%, PF={r5['profit_factor']:.2f}, maxDD={r5['max_drawdown']:.2f}%\n"
        + f"15T: trades={r15['total_trades']}, win_rate={r15['win_rate']:.2f}%, return={r15['total_return']:.2f}%, PF={r15['profit_factor']:.2f}, maxDD={r15['max_drawdown']:.2f}%\n\n"
        + f"Highlights:\n- 5T higher return but larger drawdown\n- 15T lower drawdown, more stable signals\n"
    )
    fig.text(0.05, 0.9, "June OOS Performance Comparison", fontsize=18, weight="bold")
    fig.text(0.05, 0.8, txt, fontsize=12, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)
    for imgp in imgs:
        try:
            img = Image.open(imgp)
            fig2 = plt.figure(figsize=(11.69, 8.27))
            plt.axis("off")
            plt.imshow(img)
            pdf.savefig(fig2)
            plt.close(fig2)
        except Exception:
            pass

print("Saved", pdf_path)
