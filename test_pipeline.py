"""
test_pipeline.py — tests pipeline components WITHOUT the ASR model
(which requires NeMo/canary conda env).

Tests:
  1. Backend can be imported (no errors in main.py)
  2. spoken_to_signed import (system install or local repo)
  3. Text -> Gloss works
  4. Lexicon exists and readable
  5. Full gloss->pose->GIF end-to-end
  6. FastAPI /api/health endpoint responds
"""
import sys
import os
import time
import tempfile
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
AA_ROOT = PIPELINE_ROOT.parent
GLOSS_REPO = AA_ROOT / "gloss_to_gif" / "gloss_to_gif"
DUMMY_LEXICON = GLOSS_REPO / "assets" / "dummy_lexicon"

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def test(name, fn):
    try:
        msg = fn()
        results.append((PASS, name, msg or "ok"))
        print(f"  {PASS} {name}: {msg or 'ok'}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL} {name}: {e}")

def get_text_to_gloss():
    """Use system pip install of spoken-to-signed."""
    from spoken_to_signed.text_to_gloss.simple import text_to_gloss
    return text_to_gloss, "system-pip"

# ── Test 1: Backend imports ──────────────────────────────────────────
print("\n[1] Backend import test")
sys.path.insert(0, str(PIPELINE_ROOT))
def t_backend_import():
    import unittest.mock as mock, importlib.util
    with mock.patch.dict('sys.modules', {
        'nemo': mock.MagicMock(),
        'nemo.collections': mock.MagicMock(),
        'nemo.collections.asr': mock.MagicMock(),
    }):
        spec = importlib.util.spec_from_file_location(
            "pipeline_backend_test", PIPELINE_ROOT / "backend" / "main.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert hasattr(m, 'app'), "FastAPI app missing"
        assert hasattr(m, '_load_asr_model'), "_load_asr_model missing"
        assert hasattr(m, '_gloss_to_gif'), "_gloss_to_gif missing"
    return "all functions present"
test("backend_import", t_backend_import)

# ── Test 2: spoken_to_signed import ─────────────────────────────────
print("\n[2] spoken_to_signed import test")
def t_s2s_import():
    fn, src = get_text_to_gloss()
    return f"imported from {src}"
test("spoken_to_signed_import", t_s2s_import)

# ── Test 3: Text -> Gloss ───────────────────────────────────────────
print("\n[3] Text-to-gloss test")
def t_text_to_gloss():
    text_to_gloss, src = get_text_to_gloss()
    sentences = text_to_gloss(text="Ich esse Pizza.", language="de")
    tokens = [g for sent in sentences for _, g in sent]
    assert len(tokens) > 0, "No gloss tokens generated"
    return f"tokens={tokens} (via {src})"
test("text_to_gloss", t_text_to_gloss)

# ── Test 4: Lexicon exists ───────────────────────────────────────────
print("\n[4] Lexicon check")
def t_lexicon():
    index = DUMMY_LEXICON / "index.csv"
    assert DUMMY_LEXICON.exists(), f"Missing: {DUMMY_LEXICON}"
    assert index.exists(), f"Missing index.csv in {DUMMY_LEXICON}"
    with open(index) as f:
        rows = f.readlines()
    return f"found {len(rows)-1} entries in dummy_lexicon"
test("lexicon_exists", t_lexicon)

# ── Test 5: Full gloss->pose->GIF (subprocess) ──────────────────────
print("\n[5] Full gloss->GIF test (dummy lexicon, subprocess)")
def t_full_gloss_gif():
    import subprocess, json

    # Run in a subprocess to avoid pose_format Cython "cannot load module twice" error
    script = f"""
import sys, tempfile, json
from pathlib import Path
DUMMY_LEXICON = r'{DUMMY_LEXICON}'
from spoken_to_signed.text_to_gloss.simple import text_to_gloss
from spoken_to_signed.gloss_to_pose import CSVPoseLookup, gloss_to_pose, concatenate_poses
from pose_format.pose_visualizer import PoseVisualizer

text = "Kleine Kinder essen Pizza"
sentences = text_to_gloss(text=text, language="de")
tokens = [g for sent in sentences for _, g in sent]

lookup = CSVPoseLookup(DUMMY_LEXICON)
poses = [gloss_to_pose(gloss, lookup, "de", "sgg") for gloss in sentences]
combined = concatenate_poses(poses, trim=False)

scale = combined.header.dimensions.width / 200
combined.header.dimensions.width  = int(combined.header.dimensions.width / scale)
combined.header.dimensions.height = int(combined.header.dimensions.height / scale)
combined.body.data = combined.body.data / scale

with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
    gif_path = f.name

vis = PoseVisualizer(combined)
vis.save_gif(gif_path, vis.draw())
size_kb = Path(gif_path).stat().st_size // 1024
Path(gif_path).unlink()
print(json.dumps({{"tokens": tokens, "size_kb": size_kb}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:] if result.stderr else "subprocess failed")
    data = json.loads(result.stdout.strip().splitlines()[-1])
    return f"tokens={data['tokens']}, GIF={data['size_kb']}KB"
test("full_gloss_gif", t_full_gloss_gif)


# ── Test 6: FastAPI health endpoint (subprocess) ─────────────────────
print("\n[6] FastAPI health endpoint test")
def t_health_endpoint():
    import subprocess, urllib.request, json

    server_script = (
        "import sys, os; "
        f"sys.path.insert(0, r'{PIPELINE_ROOT}'); "
        "from unittest.mock import MagicMock; "
        "import sys as _sys; "
        "_sys.modules['nemo']=MagicMock(); "
        "_sys.modules['nemo.collections']=MagicMock(); "
        "_sys.modules['nemo.collections.asr']=MagicMock(); "
        "os.environ['PORT']='8766'; "
        "from backend.main import app; import uvicorn; "
        "uvicorn.run(app, host='127.0.0.1', port=8766, log_level='error')"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", server_script],
        cwd=str(PIPELINE_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(4)
        last_err = None
        for _ in range(6):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8766/api/health", timeout=3) as r:
                    data = json.loads(r.read())
                assert data["status"] == "ok"
                return f"status={data['status']}, lexicon_ok={data['lexicon_ok']}, asr_loaded={data['asr_model_loaded']}"
            except Exception as e:
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"Server not responding: {last_err}")
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: proc.kill()
test("health_endpoint", t_health_endpoint)

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "="*55)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
print("="*55 + "\n")
if failed:
    sys.exit(1)
