"""NHTSA vehicle lookups and the legacy local-loader hook."""

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pandas as pd


def load_nhtsa(path: Path) -> pd.DataFrame:
    raise NotImplementedError("Load local recall data; reserve complaints and safety for v2")


def lookup_vehicle(
    make: str,
    model: str,
    model_year: int,
    *,
    include_complaints: bool = False,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> dict:
    """Fetch vehicle-specific recalls (and optionally complaints) from NHTSA.

    An unavailable endpoint is represented as ``status=error``.  An empty result
    is represented as ``status=ok`` with a zero count; callers must not turn that
    into a claim that the vehicle is safe.
    """
    query = {"make": make.strip(), "model": model.strip(), "modelYear": int(model_year)}
    endpoints = [("nhtsa_recalls", "/recalls/recallsByVehicle")]
    if include_complaints:
        endpoints.append(("nhtsa_complaints", "/complaints/complaintsByVehicle"))
    own_client = client is None
    http = client or httpx.Client(base_url="https://api.nhtsa.gov", timeout=timeout)
    payload: dict = {
        "query": query,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "status": "ok",
        "sources": [],
        "recalls": [],
        "complaints": [],
    }
    try:
        for source_id, path in endpoints:
            url = f"https://api.nhtsa.gov{path}?{urlencode(query)}"
            response = http.get(path, params=query)
            response.raise_for_status()
            body = response.json()
            records = body.get("results", [])
            payload["sources"].append({"id": source_id, "url": url, "record_count": len(records)})
            payload["recalls" if source_id == "nhtsa_recalls" else "complaints"] = records[:25]
        payload["recall_count"] = len(payload["recalls"])
        payload["complaint_count"] = len(payload["complaints"])
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        payload.update({"status": "error", "error": str(exc)})
    finally:
        if own_client:
            http.close()
    return payload
