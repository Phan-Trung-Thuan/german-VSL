#!/usr/bin/env python3
"""
kaggle_setup.py — Run this cell first in a Kaggle notebook to install all deps.

Usage (in a Kaggle code cell):
    !git clone https://github.com/Phan-Trung-Thuan/german-VSL /kaggle/working/german-VSL
    %cd /kaggle/working/german-VSL
    !python kaggle_setup.py
"""
import subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARN] command exited with code {result.returncode}")

# Kaggle already has: torch, torchaudio, numpy, scipy, scikit-learn
# We only need to install the missing pieces

print("=" * 60)
print("Installing pipeline dependencies for Kaggle GPU environment")
print("=" * 60)

# 1. Core audio/web deps not pre-installed on Kaggle
run("pip install -q soundfile librosa fastapi uvicorn[standard] python-multipart python-dotenv")

# 2. Sign language deps
run("pip install -q pose-format spoken-to-signed")

# 3. NeMo ASR — pin numpy to avoid downgrade conflicts
run('pip install -q "nemo_toolkit[asr]" "numpy>=2.0,<3"')

# 4. Verify all critical imports
print("\n" + "=" * 60)
print("Verifying imports...")
print("=" * 60)
checks = [
    ("torch",                   "torch"),
    ("torchaudio",              "torchaudio"),
    ("nemo (ASR)",              "nemo.collections.asr"),
    ("soundfile",               "soundfile"),
    ("librosa",                 "librosa"),
    ("pose_format",             "pose_format"),
    ("spoken_to_signed",        "spoken_to_signed"),
    ("fastapi",                 "fastapi"),
]

import importlib
ok, fail = [], []
for label, mod in checks:
    try:
        importlib.import_module(mod)
        ok.append(label)
    except ImportError as e:
        fail.append((label, str(e)))

for x in ok:
    print(f"  [OK]      {x}")
for label, err in fail:
    print(f"  [MISSING] {label}: {err}")

if fail:
    print("\nSome dependencies failed to install. Check errors above.")
    sys.exit(1)
else:
    print("\nAll dependencies ready. Run: python kaggle_run.py --help")
