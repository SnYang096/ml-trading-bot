def fake_label(df, **kwargs):
    """Fake label generator that returns a simple future return series."""
    import pandas as pd
    import numpy as np

    horizon = kwargs.get("horizon", 24)

    # Calculate simple future return
    if "close" in df.columns:
        future_return = df["close"].shift(-horizon) / df["close"] - 1
    else:
        # Fallback: return zeros
        future_return = pd.Series(0.0, index=df.index)

    return future_return


def fake_trainer(df, **kwargs):
    return [], 0.0, None, []
