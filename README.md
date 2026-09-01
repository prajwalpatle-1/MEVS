# MEVS — Multimodal Educational Video Summarizer

MEVS is a modular pipeline for producing structured educational notes from
lecture-style YouTube videos. It combines audio transcription, scene detection,
keyframe extraction, OCR on slides, and LLM-based summarization to produce a
Markdown report with timestamps and links to representative images.

This repository contains:

- A FastAPI backend (`mevs/api/app.py`) that orchestrates the pipeline and
	exposes a `/summarize` endpoint.
- Modular pipeline components under `mevs/modules/`:
	- `ingestion.py` — video ingestion and audio processing (subtitle fetch,
		`yt-dlp` audio download, Whisper transcription, chunking into timestamped
		segments).
	- `vision.py` — visual processing (video download, PySceneDetect-based
		scene detection, keyframe extraction, deduplication via perceptual hashing
		and SSIM, OCR via EasyOCR or Tesseract).
	- `fusion.py` — multimodal fusion and summarization (aligns OCR text with
		transcript chunks, calls a summarization model, assembles Markdown output).
- A minimal Chrome extension in `mevs/frontend/` (Manifest V3) that grabs the
	active tab's YouTube URL and POSTs it to the backend.

The `/summarize` endpoint supports two modes. With a YouTube link, it fetches
captions (or uses Whisper if captions are unavailable), downloads the video,
extracts representative frames, runs OCR on those frames, aligns slide text
with transcript timestamps, and returns a combined Markdown summary. A pasted
transcript can also be submitted without a link; this skips video and OCR and
summarizes the supplied text.

Outputs and artifacts
- Summaries and keyframes are saved under `mevs_outputs/` in the current
	working directory when the `/summarize` endpoint is used.

Requirements and external tools
- Python 3.9+ (3.10/3.11 recommended)
- ffmpeg accessible on `PATH` (required by `yt-dlp` and audio post-processing)
- System OCR (Tesseract) only required if you prefer `pytesseract` fallback.

Quick start (Windows example)

1) Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

2) Run tests and linter (optional):

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m flake8 mevs tests
```

3) Start the FastAPI backend:

```powershell
uvicorn mevs.api.app:app --reload --port 8000
```

4) Load the Chrome extension for interactive usage:

- Open `chrome://extensions/` → Developer mode → Load unpacked → select
	`mevs/frontend/`.
- Click the extension popup to send the current tab's URL to
	`http://localhost:8000/summarize` and view the returned Markdown.

Using the API directly

POST JSON to `/summarize` with schema:

```json
{ "url": "https://www.youtube.com/watch?v=<id>", "whisper_model": "small" }
```

The endpoint returns `{ "markdown": "..." }` containing the assembled
Markdown summary and local relative links to saved keyframe images.

Configuration & notes
- Whisper model: set `whisper_model` to a supported model name (e.g. "small",
	"medium") to trade off speed vs accuracy. Large models require more RAM.
- LLM summarization: `mevs/modules/fusion.py` currently uses a
	HuggingFace Transformers summarization pipeline (default
	`sshleifer/distilbart-cnn-12-6`) for offline summarization. You can swap
	this to an OpenAI/Gemini/other API by updating `fusion.summarize_chunks` and
	providing API keys as environment variables. For production deployments,
	using a hosted LLM via LangChain or direct API calls with proper rate
	limiting and caching is recommended.
- Performance: processing long videos is CPU/GPU and I/O intensive. For large
	scale use, consider batching, GPU-enabled Whisper, or offloading scene
	detection to a worker queue.

Developer tips
- To change chunk length, update the `chunk_duration` parameter in
	`mevs/modules/ingestion.py::chunk_segments`.
- To prefer built-in YouTube subtitles and skip Whisper transcription, pass
	`prefer_subtitles=True` to `get_transcript_chunks` (default behavior tries
	subtitles first and falls back to Whisper).
- Logging: modules use Python `logging`. Configure the root logger in your
	deployment wrapper to capture info/debug messages.

Next steps and TODOs
- Add more robust error reporting and retry logic for network operations.
- Add persistent caching for transcripts, OCR results, and summaries to avoid
	repeated work on the same video.
- Harden security for the API (authentication, rate-limiting) before public
	exposure.

Contact
If you'd like, I can:
- Finish remaining flake8 style fixes, or run `black`/`isort` to automatically
	apply formatting.
- Replace the HF summarization with a LangChain/OpenAI implementation and
	add environment-variable configuration guidance.

Enjoy!
