"""Experimental CLM ranking on Apple Silicon. Run --check before --evaluate.

Uses original Qwen3-8B weights on MPS and upstream CLM heads on CPU.
No vLLM, chat generation, quantization, or OpenRouter API key is involved.
"""

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter

ENCODER = "Qwen/Qwen3-8B"
HEAD_REPO = "Contrastive-LM/CLM-v0.1-8B"
CACHE_DIR = Path("/Volumes/mmext/autoinsight-models/clm")

STATE = (
    "The user wants used SUVs under $30,000 that offer good value and need investigation. "
    "Available data: price, mileage, make, model, model year, fuel economy and recalls."
)
CANDIDATES = [
    "Compare each listing's price against peers of the same make, model and year.",
    "Compare mileage against similar vehicles to identify unusually high wear.",
    "Review recall campaigns for each model and year before interpreting low prices.",
    "Count the most common exterior paint colors in the dataset.",
]


def last_token_embeddings(hidden, attention_mask):
    """Pool the last non-padding position for either left- or right-padded inputs."""
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(1).values
    if (last < 0).any():
        raise ValueError("Cannot embed an empty token sequence")
    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)


class MacEmbedder:
    """Small-batch last-token encoder implementing upstream Engine's embed interface.

    Tokenization uses plain text, without chat templates or added special tokens.
    Overlong inputs fail explicitly instead of silently dropping evidence.
    """

    def __init__(self, cache_dir: Path, max_tokens: int = 512, local_only: bool = False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            ENCODER, cache_dir=cache_dir, local_files_only=local_only, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                ENCODER,
                cache_dir=cache_dir,
                local_files_only=local_only,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .eval()
            .to("mps")
        )
        if self.model.config.hidden_size != 4096:
            raise ValueError("CLM requires the original 4096-dimensional Qwen3-8B encoder")
        self.cache = {}
        self.token_count = 0

    def embed(self, texts: list[str]):
        import numpy as np
        import torch

        spent = 0
        for text in dict.fromkeys(texts):
            if text in self.cache:
                continue
            tokens = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            length = tokens["input_ids"].shape[1]
            if not 0 < length <= self.max_tokens:
                raise ValueError(f"Input has {length} tokens; expected 1..{self.max_tokens}")
            tokens = {key: value.to("mps") for key, value in tokens.items()}
            with torch.inference_mode():
                output = self.model(**tokens, use_cache=False)
                vector = last_token_embeddings(output.last_hidden_state, tokens["attention_mask"])
            self.cache[text] = vector[0].cpu().numpy()
            spent += length
        self.token_count += spent
        return np.stack([self.cache[text] for text in texts]), spent


def preflight(cache_dir: Path) -> dict:
    import torch

    existing = cache_dir.resolve()
    while not existing.exists():
        existing = existing.parent
    return {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "free_disk_gib": round(shutil.disk_usage(existing).free / 1024**3, 1),
        "cache_dir": str(cache_dir.resolve()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "contrastive-lm")
        },
        "encoder": ENCODER,
        "head_repository": HEAD_REPO,
        "status": "preflight_only_no_model_loaded",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="Check runtime without downloading weights"
    )
    mode.add_argument(
        "--evaluate", action="store_true", help="Download/load weights and rank actions"
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--local-only", action="store_true", help="Use cached weights without download"
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 2048:
        parser.error("--max-tokens must be between 1 and 2048 for this initial experiment")

    try:
        check = preflight(args.cache_dir)
        if not args.evaluate:
            print(json.dumps(check, indent=2))
            return 0 if check["mps_available"] else 1
        if not check["mps_available"]:
            print("Apple GPU (MPS) is unavailable in this process.", file=sys.stderr)
            return 1

        from clm import Engine
        from huggingface_hub import hf_hub_download

        print("Loading Qwen3-8B and CLM heads; first use downloads about 16 GB.", file=sys.stderr)
        started = perf_counter()
        checkpoint = hf_hub_download(
            HEAD_REPO,
            "CLM_v0.1-8B.pt",
            cache_dir=args.cache_dir,
            local_files_only=args.local_only,
        )
        embedder = MacEmbedder(args.cache_dir, args.max_tokens, args.local_only)
        engine = Engine(embedder=embedder, checkpoint=checkpoint, device="cpu", action_cache=0)
        load_seconds = perf_counter() - started
        started = perf_counter()
        ranked = engine.rank(
            STATE, CANDIDATES, instructions="Which analysis best addresses the user's question?"
        )
        result = {
            "status": "local_evaluation_succeeded",
            "backend": "experimental_transformers_mps_bfloat16_cpu_heads",
            "encoder": ENCODER,
            "encoder_revision": getattr(embedder.model.config, "_commit_hash", None),
            "head_checkpoint": checkpoint,
            "load_seconds": round(load_seconds, 2),
            "evaluation_ms": round((perf_counter() - started) * 1000, 1),
            "encoder_tokens": embedder.token_count,
            "ranked": ranked,
            "reference_backend_parity": "not_verified",
        }
        print(json.dumps(result, indent=2))
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"CLM Mac demo failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
