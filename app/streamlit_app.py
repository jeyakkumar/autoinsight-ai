"""Streamlit vehicle value explorer for the deterministic analytics slice."""

from pathlib import Path

import pandas as pd
import streamlit as st

from autoinsight_ai.config import Settings
from autoinsight_ai.ingestion.normalization import normalize_vehicles
from autoinsight_ai.orchestration.chat_demo import run_chat

REAL_DATASET_PATH = "data/raw/vehicles_us_cleaned.csv"


def demo_listings() -> pd.DataFrame:
    """Small fallback fixture used only when the real listing file is absent."""
    return pd.DataFrame(
        {
            "make": [
                "Ford",
                "Ford",
                "Ford",
                "Ford",
                "Toyota",
                "Toyota",
                "Toyota",
                "Honda",
                "Honda",
                "Honda",
            ],
            "model": [
                "Escape",
                "Escape",
                "Escape",
                "Escape",
                "RAV4",
                "RAV4",
                "RAV4",
                "CR-V",
                "CR-V",
                "CR-V",
            ],
            "model_year": [2022, 2022, 2022, 2021, 2022, 2022, 2021, 2022, 2022, 2021],
            "mileage": [
                20_000,
                40_000,
                10_000,
                55_000,
                25_000,
                60_000,
                48_000,
                18_000,
                42_000,
                52_000,
            ],
            "fuel_type": [
                "gas",
                "gas",
                "hybrid",
                "gas",
                "hybrid",
                "gas",
                "gas",
                "hybrid",
                "gas",
                "gas",
            ],
            "fuel_economy": [29, 27, 38, 26, 37, 28, 27, 35, 28, 27],
            "price": [
                24_800,
                22_000,
                30_500,
                19_900,
                28_000,
                23_500,
                20_900,
                29_500,
                24_000,
                21_500,
            ],
        }
    )


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "year": "model_year",
        "odometer": "mileage",
        "odometer_miles": "mileage",
        "fuel": "fuel_type",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    required = {"make", "model", "model_year", "mileage", "fuel_type", "price"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    source_map = {column: column for column in required}
    normalized = normalize_vehicles(frame, source_map)
    if "fuel_economy" not in normalized:
        normalized["fuel_economy"] = pd.NA
    return normalized


def load_listings(uploaded) -> tuple[pd.DataFrame, str]:
    if uploaded is not None:
        return canonicalize(pd.read_csv(uploaded)), "uploaded CSV"
    path = Path(REAL_DATASET_PATH)
    if path.exists():
        return canonicalize(pd.read_csv(path)), str(path)
    return canonicalize(demo_listings()), "built-in fallback fixture"


settings = Settings()
st.set_page_config(page_title="AutoInsight AI", page_icon="🚗", layout="wide")
st.title("AutoInsight AI · Vehicle Chat")
st.caption("Ask about a specific make, model, and year to analyze listing evidence.")

with st.sidebar:
    st.header("Explore")
    uploaded = st.file_uploader("Optional CSV", type=["csv"])
    st.caption(
        "Real dataset columns may use year, odometer, and fuel; these are normalized automatically."
    )
    st.caption(f"Default dataset: `{REAL_DATASET_PATH}`")
    st.caption(f"Pipeline mode: {settings.pipeline_mode.value}")
    st.divider()
    st.subheader("Model and source demo")
    live_models = st.checkbox("Call live CLM/JEV", value=True)
    live_sources = st.checkbox("Query NHTSA .gov source", value=False)
    clm_url = st.text_input("CLM endpoint", value="http://localhost:8700")

try:
    vehicles, dataset_label = load_listings(uploaded)
    st.caption(f"Dataset in use: {dataset_label} · {len(vehicles):,} rows before filters")
    if vehicles.empty:
        st.warning("No listings match the current filters.")
        st.stop()
except (ValueError, KeyError) as exc:
    st.error(str(exc))
    st.stop()

st.header("Chat analysis")
st.caption("Ask a vehicle question and inspect the CLM → analytics → JEV → sources.json trace.")
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            trace = message["trace"]
            with st.expander("Decision trace", expanded=False):
                st.write("CLM action selection")
                st.json(
                    {
                        "status": trace["clm"].get("status"),
                        "selected_actions": trace["selected_actions"],
                    }
                )
                st.write("Deterministic evidence")
                st.json(trace["analysis"])
                st.write("JEV verification")
                st.json(trace["jev"])
                if trace.get("source"):
                    st.write("Source request from data/metadata/sources.json")
                    st.json(trace["source"])
question = st.chat_input("Ask: Is the 2022 Ford Escape a good value, and are there recalls?")
if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    trace = run_chat(
        question, vehicles, live_models=live_models, live_sources=live_sources, clm_url=clm_url
    )
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": trace["answer"], "trace": trace}
    )
    st.rerun()

with st.expander("Source catalog (data/metadata/sources.json)", expanded=False):
    st.json(
        {
            "nhtsa_recalls": "https://www.nhtsa.gov/nhtsa-datasets-and-apis",
            "api_pattern": "https://api.nhtsa.gov/recalls/recallsByVehicle?make={MAKE}&model={MODEL}&modelYear={MODEL_YR}",
        }
    )
