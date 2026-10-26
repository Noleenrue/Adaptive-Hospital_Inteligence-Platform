import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from explanation_utils import (  # noqa: E402
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    build_shap_figure,
    parse_explanation,
    readable_feature_name,
)


def _payload(n=12):
    return {
        "status": "success",
        "features": [
            {
                "feature": f"num__f{i}",
                "feature_value": float(i),
                "shap_value": (-1) ** i * i / 10,
            }
            for i in range(1, n + 1)
        ],
    }


def test_readable_names():
    assert (
        readable_feature_name("cat__primary_diagnosis_category_Circulatory")
        == "Primary diagnosis category: Circulatory"
    )
    assert readable_feature_name("num__length_of_stay") == "Length of stay"
    assert readable_feature_name("weird_thing") == "Weird thing"


def test_parse_keeps_top_ten_sorted_and_values_unchanged():
    df = parse_explanation(_payload())
    assert len(df) == 10
    assert df["abs_shap"].is_monotonic_decreasing
    assert df.loc[0, "feature"] == "num__f12"
    assert df.loc[0, "feature_value"] == 12.0


@pytest.mark.parametrize(
    "bad", [None, [], {}, {"features": []}, {"features": [{"feature": "a"}]}]
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_explanation(bad)


def test_figure_colors_by_sign():
    df = parse_explanation(_payload(4))
    fig = build_shap_figure(df)
    colors = list(fig.data[0].marker.color)
    assert set(colors) == {POSITIVE_COLOR, NEGATIVE_COLOR}
    assert fig.data[0].orientation == "h"
