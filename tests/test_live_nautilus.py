import pytest

import live.nautilus_live as nautilus_live


def test_build_engine_requires_nautilus():
    if nautilus_live.BacktestEngine is not None:
        pytest.skip("Nautilus trader installed; skip requirement check.")
    parser = nautilus_live.build_arg_parser()
    args = parser.parse_args(
        [
            "--data-dir",
            "dummy",
            "--results-dir",
            "dummy",
            "--symbols",
            "BTCUSDT",
        ]
    )
    with pytest.raises(RuntimeError):
        nautilus_live.build_engine(args)

