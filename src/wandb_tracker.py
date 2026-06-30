"""
Weights & Biases tracking helpers
==================================
Loads credentials from .env and provides a unified init/log API
used by the prediction and tuning pages.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def is_available() -> bool:
    return _WANDB_AVAILABLE and bool(os.environ.get("WANDB_API_KEY"))


def _ascii_safe(value: Any) -> Any:
    """Strip emoji / non-ASCII so the W&B MySQL backend (utf8mb3) accepts it."""
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKD", value)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"\s+", " ", ascii_only).strip(" -_·")
        return cleaned or "run"
    if isinstance(value, dict):
        return {_ascii_safe(k): _ascii_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ascii_safe(v) for v in value]
    return value


def init_run(run_name: str, config: dict[str, Any], job_type: str = "train"):
    """Start a W&B run. Returns the run object or None if disabled/unavailable."""
    if not is_available():
        return None
    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False, verify=False)
        run = wandb.init(
            project=_ascii_safe(os.environ.get("WANDB_PROJECT", "ds4e-final-project")),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=_ascii_safe(run_name),
            job_type=_ascii_safe(job_type),
            config=_ascii_safe(config),
            reinit=True,
        )
        return run
    except Exception as exc:
        st.warning(f"⚠️ W&B init failed: {exc}. Continuing without tracking.")
        return None


def log_metrics(run, metrics: dict[str, float], step: int | None = None) -> None:
    if run is None:
        return
    try:
        if step is None:
            run.log(metrics)
        else:
            run.log(metrics, step=step)
    except Exception:
        pass


def finish_run(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


def status_badge() -> None:
    """Render a sidebar badge indicating W&B status."""
    if is_available():
        project = os.environ.get("WANDB_PROJECT", "ds4e-final-project")
        st.caption(f"📡 W&B tracking: **ON** · project `{project}`")
    else:
        st.caption("📡 W&B tracking: **OFF** (set WANDB_API_KEY in `.env`)")    