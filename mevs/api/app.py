"""FastAPI application for the MEVS pipeline."""
from __future__ import annotations

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from mevs.modules import ingestion, vision, fusion

logger = logging.getLogger(__name__)

app = FastAPI(title="MEVS - Multimodal Educational Video Summarizer")


@app.get("/")
def read_root():
    """Confirm that the API process is accepting requests."""
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
    """Input accepted by the summarization endpoint."""

    url: str = ""
    transcript: str = Field(
        default="", description="Optional pasted transcript"
    )
    whisper_model: str = "small"


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    url = req.url.strip()
    pasted_transcript = req.transcript.strip()
    if not url and not pasted_transcript:
        raise HTTPException(
            status_code=400,
            detail="Provide a video link or transcript",
        )

    out_root = os.path.abspath(os.path.join(os.getcwd(), "mevs_outputs"))
    os.makedirs(out_root, exist_ok=True)

    # Step 1: obtain the transcript, or use pasted transcript text.
    if pasted_transcript:
        chunks = ingestion.chunk_plain_text(pasted_transcript)
    else:
        try:
            chunks = ingestion.get_transcript_chunks(
                url,
                prefer_subtitles=True,
                whisper_model=req.whisper_model,
                allow_whisper_fallback=False,
            )
        except Exception:
            logger.exception(
                "Failed while obtaining API transcript for %s", url
            )
            raise HTTPException(
                status_code=502,
                detail="Could not fetch the YouTube transcript API response",
            )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                "No transcript is available for this video. "
                "Choose a video with captions enabled or paste its transcript."
            ),
        )

    # Step 2: when a URL exists, extract frames and run OCR over the video.
    keyframes = []
    if url:
        try:
            ingestion.ensure_ffmpeg_available()
            video_path = vision.download_video(url)
        except ingestion.FFMPEGNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ffmpeg/ffprobe is required for video and OCR processing."
                ),
            )
        if not video_path:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Could not download the YouTube video for OCR. "
                    "Check the link, YouTube access, or browser cookies."
                ),
            )

        scenes = vision.detect_scenes(video_path)
        keyframes = vision.extract_keyframes(
            video_path, scenes, out_dir=os.path.join(out_root, "images")
        )
        keyframes = vision.filter_similar_frames(keyframes)
        ocr_results = vision.run_ocr_on_frames(keyframes)
    else:
        ocr_results = []

    # Step 3: combine transcript text and OCR slide text before summarization.
    aligned = fusion.align_ocr_with_chunks(ocr_results, chunks)
    summaries = fusion.summarize_chunks(aligned)

    # Save and return the Markdown summary.
    md = fusion.assemble_markdown(
        summaries,
        keyframes,
        out_dir=out_root,
        title="Multimodal Video Summary",
    )
    return {
        "markdown": md,
        "source_url": url or None,
        "chunks": len(chunks),
        "mode": "transcript-and-ocr",
    }


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/preflight-debug")
async def preflight_debug():
    """Endpoint used to verify browser preflight and CORS responses."""
    return {"ok": True}
