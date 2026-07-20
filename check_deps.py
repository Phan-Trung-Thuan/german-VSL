"""
Quick dependency check — run with: .venv\Scripts\python check_deps.py
"""
import importlib, sys

checks = [
    ("fastapi",                    "fastapi"),
    ("uvicorn",                    "uvicorn"),
    ("numpy",                      "numpy"),
    ("soundfile",                  "soundfile"),
    ("librosa",                    "librosa"),
    ("torch",                      "torch"),
    ("torchaudio",                 "torchaudio"),
    ("nemo (ASR)",                 "nemo.collections.asr"),
    ("pose_format",                "pose_format"),
    ("spoken_to_signed",           "spoken_to_signed"),
    ("pose_format.pose_visualizer","pose_format.pose_visualizer"),
]

ok, fail = [], []
for label, mod in checks:
    try:
        importlib.import_module(mod)
        ok.append(label)
    except ImportError as e:
        fail.append((label, str(e)))

print("\n[OK]:")
for x in ok:
    print(f"   {x}")

if fail:
    print("\n[MISSING]:")
    for label, err in fail:
        print(f"   {label}: {err}")
    sys.exit(1)
else:
    print("\nAll dependencies present -- pipeline ready!")
