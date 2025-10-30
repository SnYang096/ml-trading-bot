import os, json, pickle
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME",
                                    "trained_model_btcusdt_20250501_20250531")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.environ.get("MODEL_DIR", "models"),
                 f"{DEFAULT_MODEL_NAME}.pkl"),
)
OUT_DIR = os.path.join("reports")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    m = pickle.load(open(MODEL_PATH, "rb"))
    pipeline = m["strategy"].pipeline
    # stage1 classification importance
    for tf, model in pipeline.stage1_models.items():
        booster = getattr(model.model, "booster_", None) or model.model
        try:
            imp = booster.feature_importance()
            names = booster.feature_name()
            df = pd.DataFrame({
                "feature": names,
                "importance": imp
            }).sort_values("importance", ascending=False)
        except Exception:
            # fallback using lightgbm sklearn API
            if hasattr(model.model, "feature_importances_"):
                # Feature names are unknown; output generic indices
                df = pd.DataFrame({
                    "feature": [
                        f"f{i}"
                        for i in range(len(model.model.feature_importances_))
                    ],
                    "importance":
                    model.model.feature_importances_,
                })
                df = df.sort_values("importance", ascending=False)
            else:
                continue
        csvp = os.path.join(OUT_DIR, f"feature_importance_stage1_{tf}.csv")
        df.to_csv(csvp, index=False)
        top = df.head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title(f"Stage1 Feature Importance Top-20 ({tf})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"feature_importance_stage1_{tf}_top20.png"))
        plt.close()
    # stage2 regression importance
    for tf, model in pipeline.stage2_models.items():
        booster = getattr(model.model, "booster_", None) or model.model
        try:
            imp = booster.feature_importance()
            names = booster.feature_name()
            df = pd.DataFrame({
                "feature": names,
                "importance": imp
            }).sort_values("importance", ascending=False)
        except Exception:
            if hasattr(model.model, "feature_importances_"):
                df = pd.DataFrame({
                    "feature": [
                        f"f{i}"
                        for i in range(len(model.model.feature_importances_))
                    ],
                    "importance":
                    model.model.feature_importances_,
                })
                df = df.sort_values("importance", ascending=False)
            else:
                continue
        csvp = os.path.join(OUT_DIR, f"feature_importance_stage2_{tf}.csv")
        df.to_csv(csvp, index=False)
        top = df.head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title(f"Stage2 Feature Importance Top-20 ({tf})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"feature_importance_stage2_{tf}_top20.png"))
        plt.close()
    print("Exported feature importance CSVs and plots to", OUT_DIR)


if __name__ == "__main__":
    main()
