def fake_label(df, **kwargs):
    return df.get("label", 0)


def fake_trainer(df, **kwargs):
    return [], 0.0, None, []
