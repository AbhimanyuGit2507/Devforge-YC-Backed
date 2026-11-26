"""Monsoon Digital Twin inference server (offline, CPU).   python serve.py --port 8000

Loads weights/ once at import, then POST /simulate rolls the model forward day by day. CORS is open so the separate
dashboard (dashboard/) can call it from the browser.
"""
import argparse
import numpy as np
import uvicorn
import torch
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import model as twin

torch.set_num_threads(4)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL_NAME = "monsoon-twin-localcnn-v1"
MODEL = twin.load(Path(__file__).parent / "weights")
MODEL.rollout(np.full((3, 6, 32, 36), [[[290.0]], [[1.0]], [[0.0]], [[0.0]], [[101000.0]], [[30.0]]], "f4"), [0.0, 1.0], 2)  # warm-up


def _arr(x):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        import base64
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Return [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    if init_states.shape != (3, 6, 32, 36) or (perturbation is not None and perturbation.shape != (6, 32, 36)):
        raise ValueError("bad shape")
    return MODEL.rollout(np.nan_to_num(init_states), doy_sincos, steps, None if perturbation is None else np.nan_to_num(perturbation))


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter(); body = await request.json()
    init = _arr(body["init_states"]); pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
    out = simulate(init, body["doy_sincos"], int(body["steps"]), pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
