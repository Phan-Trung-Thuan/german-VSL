#!/usr/bin/env python3
"""
kaggle_setup.py — Run this cell first in a Kaggle notebook to install all deps.

Usage (in a Kaggle code cell):
    !git clone https://github.com/Phan-Trung-Thuan/german-VSL /kaggle/working/german-VSL
    %cd /kaggle/working/german-VSL
    !python kaggle_setup.py

Kaggle GPU environment already provides (DO NOT reinstall these):
    torch, torchaudio, numpy, scipy, scikit-learn, pandas,
    numba (0.66+), cuml-cu12, cudf-cu12, matplotlib, Pillow

What we install here:
    1. Audio I/O : soundfile, librosa
    2. Web server: fastapi, uvicorn, python-multipart, python-dotenv
    3. Sign lang : pose-format, spoken-to-signed
    4. NeMo ASR  : installed from GitHub main to get numba 0.66 compatibility fix
                   (PyPI nemo_toolkit requires numba<0.62, which breaks Kaggle's Rapids libs)
"""
import subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARN] command exited with code {result.returncode}")

print("=" * 60)
print("Installing pipeline dependencies for Kaggle GPU environment")
print("=" * 60)

# ── 1. Audio I/O and web server ──────────────────────────────────────
# Kaggle does NOT pre-install these.
run("pip install -q soundfile librosa fastapi uvicorn[standard] python-multipart python-dotenv")

# ── 2. Sign language packages ─────────────────────────────────────────
# Kaggle does NOT pre-install these.
run("pip install -q pose-format spoken-to-signed")

# ── 3. NeMo ASR ───────────────────────────────────────────────────────
# Problem: PyPI nemo_toolkit[asr] pins numba<0.62, but Kaggle ships numba 0.66+.
#   - Downgrading numba would break Kaggle's pre-installed Rapids (cuml, cudf).
#   - The NPDatetime AttributeError was fixed in NeMo's GitHub main branch.
# Solution: install NeMo directly from GitHub (no numba version conflict).
run("pip install -q 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git'")

# ── 4. Verify all critical imports ────────────────────────────────────
print("\n" + "=" * 60)
print("Verifying imports...")
print("=" * 60)

checks = [
    ("torch",           "torch"),
    ("torchaudio",      "torchaudio"),
    ("numba",           "numba"),          # should remain at Kaggle's 0.66+
    ("nemo (ASR)",      "nemo.collections.asr"),
    ("soundfile",       "soundfile"),
    ("librosa",         "librosa"),
    ("pose_format",     "pose_format"),
    ("spoken_to_signed","spoken_to_signed"),
    ("fastapi",         "fastapi"),
]

import importlib
ok, fail = [], []
for label, mod in checks:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, "__version__", "?")
        ok.append((label, version))
    except Exception as e:
        # Catch ImportError, AttributeError, and any other runtime failures
        fail.append((label, f"{type(e).__name__}: {e}"))

for label, ver in ok:
    print(f"  [OK]      {label:<20} {ver}")
for label, err in fail:
    print(f"  [MISSING] {label}: {err}")

if fail:
    print("\nSome dependencies failed to install. Check errors above.")
    sys.exit(1)
else:
    print("\nAll dependencies ready. Run: python kaggle_run.py --help")
