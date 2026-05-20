from src.live_data_stream.feature_bus import effective_max_rows_for_warmup


def test_max_rows_bumped_for_warmup_days():
    assert effective_max_rows_for_warmup(3000, 180) == 180 * 24 * 60


def test_max_rows_unchanged_without_warmup():
    assert effective_max_rows_for_warmup(5000, 0) == 5000
