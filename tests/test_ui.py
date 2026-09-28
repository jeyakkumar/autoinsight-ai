"""Smoke test the chat-first Streamlit interface without a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_chat_interface_starts_with_real_listing_data():
    app = AppTest.from_file(Path(__file__).parents[1] / "app" / "streamlit_app.py").run()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert len(app.dataframe) == 0
