"""Load the UPEs dataset (rows = time points, columns = subjects) from Excel or MATLAB files."""
import io
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.io as sio

EXPORT_HINT = (
    "This file appears to contain MATLAB table objects. Install the project's updated "
    "backend dependencies with `pip install -r requirements.txt`; the table-aware "
    "`mat-io` reader is used to decode MATLAB tables directly. If decoding still fails, "
    "send the complete upload error and the MAT-file version; no user-side conversion "
    "should be required for supported MAT-file formats."
)


def _unwrap_singleton(value):
    """Unwrap one-element MATLAB arrays commonly used around structs/cells."""
    while isinstance(value, np.ndarray) and value.size == 1:
        if value.dtype == object or value.dtype.names:
            value = value.reshape(-1)[0]
        else:
            break
    return value


def _named_items(value):
    """Return named fields from dicts, SciPy mat_structs, and structured arrays."""
    value = _unwrap_singleton(value)
    if isinstance(value, dict):
        return [(str(k), v) for k, v in value.items() if not str(k).startswith("__")]
    if hasattr(value, "_fieldnames"):
        return [(str(name), getattr(value, name)) for name in value._fieldnames]
    if isinstance(value, np.void) and value.dtype.names:
        return [(str(name), value[name]) for name in value.dtype.names]
    if isinstance(value, np.ndarray) and value.dtype.names:
        return [(str(name), value[name]) for name in value.dtype.names]
    return []


def _as_numeric_matrix(value, field_name):
    """Convert a MATLAB table/DataFrame or numeric array to time x subject values."""
    value = _unwrap_singleton(value)
    if isinstance(value, pd.DataFrame):
        try:
            arr = value.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MATLAB table '{field_name}' contains non-numeric columns. "
                "UPE traces must be numeric values with rows=time points and columns=subjects."
            ) from exc
    elif isinstance(value, pd.Series):
        arr = pd.to_numeric(value, errors="raise").to_numpy(dtype=float).reshape(-1, 1)
    else:
        arr = np.asarray(value)
        if arr.dtype.kind not in "fiu b":
            try:
                arr = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return None
        else:
            arr = arr.astype(float, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _extract_fields(mat):
    """Extract UPEs' per-electrode fields or plain numeric top-level MAT variables."""
    root = mat.get("UPEs") if isinstance(mat, dict) else None
    items = _named_items(root) if root is not None else []
    if not items and isinstance(mat, dict):
        items = [(str(k), v) for k, v in mat.items() if not str(k).startswith("__")]

    out = OrderedDict()
    for name, value in items:
        # A UPEs struct is expected to contain one table per electrode/task field.
        # Do not treat the containing struct itself as a numeric data field.
        if name == "UPEs" and _named_items(value):
            continue
        arr = _as_numeric_matrix(value, name)
        if arr is not None:
            out[str(name)] = _orient(arr)
    return out


def _from_matio(data):
    """Use mat-io to decode MATLAB table objects across MAT v4/v6/v7/v7.3."""
    from matio import load_from_mat

    # mat-io expects a path, while FastAPI has already read the uploaded bytes.
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as temp:
        temp.write(data)
        temp_path = temp.name
    try:
        decoded = load_from_mat(temp_path, raw_data=False, add_table_attrs=False)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return _extract_fields(decoded)



@dataclass
class Dataset:
    fields: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict)
    warnings: list = field(default_factory=list)

    @property
    def tasks(self):
        """Task suffixes when every field is named <Electrode>_<Task>, else []."""
        names = list(self.fields)
        if not names or not all("_" in n for n in names):
            return []
        return list(OrderedDict.fromkeys(n.rsplit("_", 1)[1] for n in names))

    @property
    def n_subjects(self):
        return min(a.shape[1] for a in self.fields.values())

    @property
    def n_points(self):
        return max(a.shape[0] for a in self.fields.values())

    def select(self, task=None):
        """Fields to analyse. With a task, only that task's fields, named by electrode."""
        if not task:
            return OrderedDict(self.fields)
        out = OrderedDict((n.rsplit("_", 1)[0], a) for n, a in self.fields.items()
                          if "_" in n and n.rsplit("_", 1)[1] == task)
        if not out:
            raise ValueError(f"No fields found for task '{task}'. Tasks: {self.tasks}")
        return out


def _orient(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Each field must be a 2-D array (time points x subjects).")
    return arr.T if arr.shape[0] < arr.shape[1] else arr


def _from_xlsx(data):
    out = OrderedDict()
    for sheet, df in pd.read_excel(io.BytesIO(data), sheet_name=None, header=0).items():
        out[str(sheet)] = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return out


def _from_mat(data):
    """Load plain numeric MAT variables and MATLAB structs containing tables.

    SciPy handles ordinary numeric arrays and basic structs but leaves MATLAB
    ``table`` class objects opaque. The table-aware ``mat-io`` package is tried
    first; SciPy remains a fallback for simple MAT files if that reader is absent
    or cannot decode a file.
    """
    matio_error = None
    try:
        decoded = _from_matio(data)
        if decoded:
            return decoded
    except ImportError as exc:
        matio_error = exc
    except Exception as exc:  # malformed or unsupported by table-aware reader
        matio_error = exc

    try:
        mat = sio.loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        detail = f" Table-aware reader detail: {matio_error}" if matio_error else ""
        raise ValueError(
            "This is likely a MATLAB v7.3 MAT file. Install/update backend dependencies "
            "with `pip install -r requirements.txt` so `mat-io` can read its MATLAB table "
            f"objects directly.{detail}"
        ) from exc

    mat = {k: v for k, v in mat.items() if not k.startswith("__")}
    out = _extract_fields(mat)
    if out:
        return out

    # SciPy signals classdef tables as opaque objects; don't silently skip them.
    opaque_found = any(
        type(value).__name__ in {"MatlabOpaque", "MatlabObject"}
        for value in mat.values()
    )
    if opaque_found:
        detail = f" Table-aware reader detail: {matio_error}" if matio_error else ""
        raise ValueError(EXPORT_HINT + detail)

    detail = f" Table-aware reader detail: {matio_error}" if matio_error else ""
    raise ValueError(
        "No numeric arrays or decodable MATLAB tables were found in this .mat file. "
        "Expected a variable named UPEs whose fields are numeric matrices or tables, "
        "with rows=time points and columns=subjects. " + detail
    )


def load_files(files):
    """files: list of (filename, bytes). Later files add fields to earlier ones."""
    ds = Dataset()
    for name, data in files:
        lower = name.lower()
        if lower.endswith((".xlsx", ".xls")):
            parsed = _from_xlsx(data)
        elif lower.endswith(".mat"):
            parsed = _from_mat(data)
        elif lower.endswith(".csv"):
            frame = pd.read_csv(io.BytesIO(data), header=0)
            parsed = OrderedDict([(os.path.splitext(os.path.basename(name))[0] or "Trace",
                                   frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))])
        else:
            raise ValueError(f"{name}: unsupported file type (use .xlsx, .mat, or .csv)")
        for key, arr in parsed.items():
            ds.fields[key] = _orient(arr)
    if not ds.fields:
        raise ValueError("No data found in the uploaded files.")
    for key, arr in ds.fields.items():
        if np.isnan(arr).any():
            ds.warnings.append(f"{key}: contains missing values (NaN); results for those traces may be NaN.")
    if len({a.shape[0] for a in ds.fields.values()}) > 1:
        ds.warnings.append("Fields have different numbers of time points.")
    return ds
