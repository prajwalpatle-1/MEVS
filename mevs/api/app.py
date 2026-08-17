"""FastAPI application for the MEVS pipeline."""
from __future__ import annotations

import os
import logging
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from mevs.modules import ingestion, vision, fusion

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MEVS - Multimodal Educational Video Summarizer",
)

@app.get("/")
def read_root():
    return {"status": "API is running successfully!"}

# Allow CORS for development (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # For development only: allow all origins. Disable credentials so
    # the wildcard origin is allowed and browsers receive the header.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)





class SummarizeRequest(BaseModel):
    url: str
    whisper_model: str = "small"


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    url = req.url
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")

    out_root = os.path.abspath(os.path.join(os.getcwd(), "mevs_outputs"))
    os.makedirs(out_root, exist_ok=True)

    # Transcript chunks
    try:
        chunks = ingestion.get_transcript_chunks(url, whisper_model=req.whisper_model)
    except ingestion.FFMPEGNotFoundError as exc:
        # Clearer 503 for missing system dependency
        raise HTTPException(
            status_code=503,
            detail=(
                "ffmpeg/ffprobe not found on the server. "
                "Install ffmpeg or set the FFMPEG_PATH environment variable."
            ),
        )
    except Exception:
        logger.exception("Failed while obtaining transcript for %s", url)
        raise HTTPException(status_code=500, detail="Failed to obtain transcript")

    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to obtain transcript")

    # Video download + vision processing
    video_path = vision.download_video(url)
    if not video_path:
        raise HTTPException(status_code=500, detail="Failed to download video")
    scenes = vision.detect_scenes(video_path)
    keyframes = vision.extract_keyframes(video_path, scenes, out_dir=os.path.join(out_root, "images"))
    keyframes = vision.filter_similar_frames(keyframes)
    ocr_results = vision.run_ocr_on_frames(keyframes)

    # Fusion + summarization
    aligned = fusion.align_ocr_with_chunks(ocr_results, chunks)
    summaries = fusion.summarize_chunks(aligned)

    # assemble markdown and return
    md = fusion.assemble_markdown(summaries, keyframes, out_dir=out_root, title=url)
    return {"markdown": md}


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/preflight-debug")
async def preflight_debug():
    """Endpoint to test preflight/CORS from browsers. Curl OPTIONS should also
    trigger middleware to attach CORS headers."""
    return {"ok": True}
