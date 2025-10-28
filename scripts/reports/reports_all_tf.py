import os, json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_DIR = os.path.join("results", "june_2025_oos")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

TIMEFRAMES = ["5T", "15T", "60T", "240T"]


def load_results():
    rows = []
    for tf in TIMEFRAMES:
        jp = os.path.join(RESULTS_DIR, f"wavelet_{tf}_june_results.json")
        if not os.path.exists(jp):
            continue
        with open(jp) as f:
            r = json.load(f)
        rows.append(
            {
                "timeframe": tf,
                "trades": r.get("total_trades", 0),
                "win_rate_%": round(r.get("win_rate", 0.0), 2),
                "return_%": round(r.get("total_return", 0.0), 2),
                "profit_factor": round(r.get("profit_factor", 0.0), 2),
                "max_drawdown_%": round(r.get("max_drawdown", 0.0), 2),
            }
        )
    return pd.DataFrame(rows)


def to_pdf(df: pd.DataFrame, path: str):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.axis("off")
        plt.title("June OOS Summary (5T/15T/60T/240T)", loc="left", fontsize=18, pad=20)
        table = plt.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    df = load_results()
    if df.empty:
        print("No results found.")
        return
    csvp = os.path.join(REPORTS_DIR, "oos_summary_5T_15T_60T_240T.csv")
    df.to_csv(csvp, index=False)
    pdfp = os.path.join(REPORTS_DIR, "oos_summary_5T_15T_60T_240T.pdf")
    to_pdf(df, pdfp)
    print("Saved", csvp, "and", pdfp)


if __name__ == "__main__":
    main()
