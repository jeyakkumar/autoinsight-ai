import pandas as pd

from autoinsight_ai.orchestration.chat_demo import _live_clm, run_chat


def test_chat_does_not_invent_price_for_missing_vehicle() -> None:
    vehicles = pd.DataFrame(
        {
            "make": ["Ford"],
            "model": ["Escape"],
            "model_year": [2022],
            "mileage": [20_000],
            "fuel_type": ["gas"],
            "price": [24_800],
        }
    )
    from autoinsight_ai.ingestion.normalization import normalize_vehicles

    result = run_chat(
        "What is the 2020 Toyota RAV4 price?",
        normalize_vehicles(vehicles, {c: c for c in vehicles.columns}),
    )
    assert "couldn't find an exact" in result["answer"]
    assert result["analysis"] == {}


def test_chat_ranks_top_selling_as_listing_frequency() -> None:
    vehicles = pd.DataFrame(
        {
            "make": ["Ford", "Ford", "Toyota"],
            "model": ["Escape", "Escape", "RAV4"],
            "model_year": [2022, 2022, 2022],
            "mileage": [20_000, 30_000, 15_000],
            "fuel_type": ["gas", "gas", "hybrid"],
            "price": [24_800, 23_000, 28_000],
        }
    )
    from autoinsight_ai.ingestion.normalization import normalize_vehicles

    result = run_chat(
        "Show me the top selling cars",
        normalize_vehicles(vehicles, {c: c for c in vehicles.columns}),
    )
    assert "Ford Escape" in result["answer"]
    assert "listing frequency" in result["answer"]


def test_chat_corrects_small_make_typo() -> None:
    vehicles = pd.DataFrame(
        {
            "make": ["Toyota"],
            "model": ["RAV4"],
            "model_year": [2021],
            "mileage": [20_000],
            "fuel_type": ["gas"],
            "price": [28_000],
        }
    )
    from autoinsight_ai.ingestion.normalization import normalize_vehicles

    result = run_chat(
        "Is the 2021 Toyato RAV4 good?",
        normalize_vehicles(vehicles, {c: c for c in vehicles.columns}),
    )
    assert "2021 Toyota RAV4" in result["answer"]


def test_chat_compares_two_model_years() -> None:
    vehicles = pd.DataFrame(
        {
            "make": ["Honda", "Honda", "Honda"],
            "model": ["CR-V", "CR-V", "CR-V"],
            "model_year": [2024, 2024, 2025],
            "mileage": [10_000, 20_000, 8_000],
            "fuel_type": ["gas", "gas", "hybrid"],
            "price": [30_000, 29_000, 33_000],
        }
    )
    from autoinsight_ai.ingestion.normalization import normalize_vehicles

    result = run_chat(
        "Compare Honda CR-V 2024 vs 2025",
        normalize_vehicles(vehicles, {c: c for c in vehicles.columns}),
    )
    assert "2024" in result["answer"] and "2025" in result["answer"]
    assert len(result["analysis"]["rows"]) == 2


def test_live_clm_maps_descriptive_candidates(monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "ranked": [
                    {
                        "rank": 1,
                        "candidate": (
                            "price_percentile: Compare listing price with same make, model "
                            "and year peers."
                        ),
                        "prob": 0.9,
                    }
                ]
            }

    monkeypatch.setattr(
        "autoinsight_ai.orchestration.chat_demo.httpx.post", lambda *a, **k: Response()
    )
    result = _live_clm("Is it a good price?", ["price_percentile"], "http://localhost:8700")
    assert result["status"] == "live"
    assert result["selected"] == ["price_percentile"]
