"""PEG analysis API: orchestrates the step modules for the React front end.

Run:  uvicorn main:app --reload --port 8000
Contract: see the comment block at the top of the front end's App.jsx.
"""
import io
import os
import tempfile
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import clustering
import descriptives
import dfa
import entropy
import loaders
import mixed_models
import network
import spectral

app = FastAPI(title="PEG analysis (STAR Protocols 2025) - Python port")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


class Session:
    def __init__(self, dataset):
        self.id = uuid.uuid4().hex[:12]
        self.dataset = dataset
        self.dir = tempfile.mkdtemp(prefix=f"peg_{self.id}_")
        self.spectro: Dict[str, tuple] = {}
        self.stats_df: Optional[pd.DataFrame] = None


SESSIONS: Dict[str, Session] = {}


class RunBody(BaseModel):
    action: str = "run"
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def get_session(sid) -> Session:
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session. Load a dataset first.")
    return SESSIONS[sid]


def validate_subject_count(fields, params):
    available = min(a.shape[1] for a in fields.values())
    try:
        requested = int(params.get("numSubjects", available))
    except (TypeError, ValueError) as exc:
        raise ValueError("numSubjects must be a whole number.") from exc
    if requested < 1 or requested > available:
        raise ValueError(f"numSubjects must be between 1 and {available}; received {requested}.")


def fields_for(s: Session, params) -> "OrderedDict[str, np.ndarray]":
    fields = s.dataset.select(params.get("task") or None)
    validate_subject_count(fields, params)
    return fields


def electrodes_for(s: Session, params):
    ds = s.dataset
    if ds.tasks and not params.get("task"):
        raise ValueError(f"This dataset has several tasks {ds.tasks}; select a task so analyses are run separately "
                         "as recommended in the paper.")
    fields = ds.select(params.get("task") or None)
    validate_subject_count(fields, params)
    return fields


def preview(ds, max_subjects=None):
    """Return {field: [[subj1 trace], [subj2 trace], ...]} for the front end viewer.

    arr is (time_points, subjects). We downsample time to ~2000 points, then
    transpose so each inner list is one subject's full trace.
    """
    out = {}
    for name, arr in ds.fields.items():
        stride = max(1, arr.shape[0] // 2000)
        sampled = arr[::stride, :]                 # (time', subjects)
        if max_subjects is not None:
            sampled = sampled[:, :max_subjects]
        out[name] = sampled.T                      # (subjects, time')
    return out


# ---------------------------------------------------------------- step handlers
def h_descriptives(s, action, p):
    return descriptives.run(fields_for(s, p), p, s.dir)


def h_entropy(s, action, p):
    return entropy.run(fields_for(s, p), p, s.dir)


def h_clustering(s, action, p):
    return clustering.run(fields_for(s, p), p)


def h_stft(s, action, p):
    if action == "compare":
        return spectral.compare_windows(fields_for(s, p), p)
    return spectral.run_stft(fields_for(s, p), p, s.spectro, s.dir)


def h_plot_spec(s, action, p):
    return spectral.plot_data(s.spectro, p.get("traces", []), float(p.get("fs", 20)), float(p.get("timeWindow", 20)))


def h_dfa(s, action, p):
    fields = fields_for(s, p)
    if action == "filter":
        return dfa.filter_response(p)
    if action == "single":
        return dfa.run_single(fields, p)
    if action == "loop":
        return dfa.run_loop(fields, p, s.dir)
    raise ValueError(f"Unknown DFA action '{action}'.")


def make_network_handler(kind):
    def handler(s, action, p):
        return network.run(kind, electrodes_for(s, p), p, s.dir)
    return handler


def h_stats(s, action, p):
    if s.stats_df is None:
        raise ValueError("Upload the stats dataset (Subject, Electrode, Task and the metric) first.")
    return mixed_models.run(s.stats_df, action, p)


HANDLERS = {"descriptives": h_descriptives, "entropy": h_entropy, "clustering": h_clustering,
            "stft": h_stft, "plot_spec": h_plot_spec, "dfa": h_dfa, "stats": h_stats,
            **{k: make_network_handler(k) for k in network.STEP_INFO}}


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/sessions")
async def create_session(request: Request):
    """Create a session from uploaded files.

    Accept any multipart file field name to remain compatible with frontend
    versions that send `files`, `file`, or `files[]`. This also converts the
    otherwise opaque FastAPI 422 missing-field error into a readable message.
    """
    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=("Could not read the upload form. Please restart the backend and "
                    "make sure the request is sent as multipart/form-data. " + str(exc)),
        ) from exc

    uploads = []
    for _field_name, value in form.multi_items():
        # Starlette's request.form() returns UploadFile objects. Attribute
        # checks intentionally accept both Starlette and FastAPI UploadFile types.
        if hasattr(value, "filename") and callable(getattr(value, "read", None)):
            uploads.append(value)

    if not uploads:
        raise HTTPException(
            status_code=400,
            detail=("No file was received. The upload form must include UPEs.mat, "
                    "an Excel workbook (.xlsx/.xls), or a CSV file. If you still "
                    "see this message, update both frontend/App.jsx and backend/main.py "
                    "from the same project bundle, then restart the backend."),
        )

    try:
        payloads = []
        for uploaded in uploads:
            payloads.append((uploaded.filename or "uploaded_file", await uploaded.read()))
        ds = loaders.load_files(payloads)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    s = Session(ds)
    SESSIONS[s.id] = s
    return jsonable({"session_id": s.id, "fields": list(ds.fields), "tasks": ds.tasks,
                     "n_subjects": ds.n_subjects, "n_points": ds.n_points,
                     "preview": preview(ds), "warnings": ds.warnings})


@app.post("/sessions/{sid}/stats_dataset")
async def stats_dataset(sid: str, file: UploadFile = File(...)):
    s = get_session(sid)
    raw = await file.read()
    try:
        df = (pd.read_csv(io.BytesIO(raw)) if file.filename.lower().endswith(".csv")
              else pd.read_excel(io.BytesIO(raw)))
        mixed_models.prepare(df)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc))
    s.stats_df = df
    return {"columns": list(df.columns), "n_rows": int(len(df))}


@app.post("/sessions/{sid}/steps/{step_id}")
def run_step(sid: str, step_id: str, body: RunBody):
    s = get_session(sid)
    if step_id not in HANDLERS:
        raise HTTPException(404, f"Unknown step '{step_id}'.")
    try:
        result = HANDLERS[step_id](s, body.action, body.params)
    except (ValueError, KeyError, NotImplementedError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")
    files = result.pop("files", [])
    result["downloads"] = [{"label": f["label"], "url": f"/sessions/{sid}/files/{f['name']}"} for f in files]
    return jsonable(result)


@app.get("/sessions/{sid}/files/{name}")
def download(sid: str, name: str):
    s = get_session(sid)
    path = os.path.join(s.dir, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found.")
    return FileResponse(path, filename=os.path.basename(name))
