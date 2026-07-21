# German Speech → Sign Language Pipeline

Combines 3 sub-repos into a single end-to-end pipeline:

```
German Speech → [Canary ASR] → German Text → [spoken_to_signed] → DGS Gloss → [pose_format] → Animated GIF
```

## Sub-repos used

| Sub-repo | Role |
|----------|------|
| `canary_web_asr` | Step 1: German speech → German text (NVIDIA Canary-1B ASR) |
| `gloss_to_gif` | Steps 2–3: German text → Sign Language gloss → Animated GIF |
| `baseline` | Reference: Marian NMT phoenix-transformer (gloss ↔ text research baseline) |

## Quick start

### 1. Install dependencies

```powershell
# Activate your canary conda environment
conda activate canary

# Install pipeline requirements
pip install -r requirements.txt

# Install spoken-to-signed (from gloss_to_gif sub-repo)
pip install spoken-to-signed
# OR: pip install pose-format
```

### 2. Run the web app

```powershell
conda activate canary
python backend/main.py
```

Open `http://127.0.0.1:8000` in your browser.

**First run** will download the Canary-1B model (~4GB). After that it's instant.

### 3. Use the CLI & Scripts

All executable scripts are located in the `scripts/` folder:

```powershell
# Run Speech-to-Sign CLI pipeline
python scripts/pipeline_cli.py --audio path/to/speech.wav --gif output.gif

# Batch-download SignDict videos & build local vocabulary mapping
python scripts/prepare_local_library.py

# Extract Whole-Body 2D skeleton MP4s & keypoint JSON files
python scripts/extract_skeletons.py

# Fine-tune Text-to-Gloss Seq2Seq baseline model
python scripts/train_t2g.py --base_path /path/to/annotations
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `nvidia/canary-1b` | Canary model ID (`nvidia/canary-1b-v2` for better FLEURS accuracy) |
| `LEXICON_DIR` | `../gloss_to_gif/gloss_to_gif/assets/dummy_lexicon` | Path to sign language lexicon |
| `HOST` | `127.0.0.1` | Server host |
| `PORT` | `8000` | Server port |

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Backend health + model status |
| `POST` | `/api/transcribe` | Audio file → German text (ASR only) |
| `POST` | `/api/gloss` | `{"text":"..."}` → gloss tokens |
| `POST` | `/api/gif` | `{"text":"..."}` → gloss + GIF (base64) |
| `POST` | `/api/pipeline` | Audio file → transcript + gloss + GIF (full pipeline) |

## Notes

- **Lexicon**: The `dummy_lexicon` only contains 4 German words (`kleine`, `kinder`, `essen`, `pizza`). For production use, download the SignSuisse lexicon:
  ```python
  python -m spoken_to_signed.download_lexicon --name signsuisse --directory /path/to/lexicon
  ```
- **Sign languages**: `sgg` (Swiss German SL), `gsg` (DGS), `bfi` (British SL), `ase` (ASL)
- **Microphone**: Only works on `localhost` or HTTPS
