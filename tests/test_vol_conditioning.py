import pandas as pd

from vol_conditioning import conditioned


def test_positions_halved_only_when_next_month_is_high_vol():
    sig = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "month": ["2020-01", "2020-02", "2020-03"],
            "lookback": 12,
            "signal": [1, -1, 1],
            "vol_exante": 0.1,
            "position": [1.0, -1.0, 1.0],
        }
    )
    hv = pd.Series([True, False, None], index=pd.period_range("2020-02", periods=3, freq="M"))
    out = conditioned(sig, hv, k=0.5)
    # Jan positions apply to Feb (high vol) -> halved; Feb -> Mar (calm); Mar -> Apr (unknown)
    assert out.position.tolist() == [0.5, -1.0, 1.0]
    assert sig.position.tolist() == [1.0, -1.0, 1.0]  # input untouched
