"""FastAPI application for the MEVS pipeline."""
from __future__ import annotations

import os
import logging
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from mevs.modules import document_export, ingestion, vision, fusion

logger = logging.getLogger(__name__)


def _safe_summary_from_transcript(chunks):
    """Generate transcript-only summary without requiring OCR/video download."""
    summaries = fusion.summarize_chunks(chunks)
    if not summaries:
        return "# Video Summary\n\n- No usable transcript text was found.\n"
    return summaries[0]["markdown"]

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


class ExportRequest(BaseModel):
    """Markdown content to export as a downloadable document."""

    markdown: str = Field(min_length=1, max_length=5000000)


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
    ocr_results = []
    video_download_failed = False
    if url:
        video_temp_dir = tempfile.TemporaryDirectory(prefix="mevs_video_")
        try:
            try:
                ingestion.ensure_ffmpeg_available()
                video_path = vision.download_video(
                    url,
                    os.path.join(video_temp_dir.name, "video.mp4"),
                )
                if not video_path:
                    raise RuntimeError("Video download returned no file")

                try:
                    candidate_count = int(
                        os.environ.get("MEVS_FRAME_CANDIDATES", "32")
                    )
                except ValueError:
                    logger.warning("Invalid MEVS_FRAME_CANDIDATES; using 32")
                    candidate_count = 32
                candidate_count = min(max(candidate_count, 8), 128)
                try:
                    max_keyframes = int(
                        os.environ.get("MEVS_MAX_KEYFRAMES", "8")
                    )
                except ValueError:
                    logger.warning("Invalid MEVS_MAX_KEYFRAMES; using 8")
                    max_keyframes = 8
                max_keyframes = min(max(max_keyframes, 1), 32)
                scenes = vision.sample_video_ranges(
                    video_path, scene_count=candidate_count
                )
                logger.info(
                    "Sampled %d candidate intervals; ranking up to %d frames",
                    len(scenes),
                    max_keyframes,
                )
                with tempfile.TemporaryDirectory(
                    prefix="mevs_frames_"
                ) as temp_dir:
                    candidate_frames = vision.extract_keyframes(
                        video_path,
                        scenes,
                        out_dir=os.path.join(temp_dir, "images"),
                    )
                    candidate_ocr = vision.run_ocr_on_frames(candidate_frames)
                    ocr_by_path = {
                        os.path.normcase(os.path.abspath(item["image_path"])): item
                        for item in candidate_ocr
                        if item.get("image_path")
                    }
                    for frame in candidate_frames:
                        ocr_item = ocr_by_path.get(
                            os.path.normcase(
                                os.path.abspath(frame["image_path"])
                            )
                        )
                        frame["ocr_text"] = (
                            ocr_item.get("text", "") if ocr_item else ""
                        )

                    keyframes = vision.select_important_frames(
                        candidate_frames,
                        chunks,
                        max_frames=max_keyframes,
                    )
                    keyframes = vision.filter_similar_frames(
                        keyframes,
                        hash_threshold=4,
                        ssim_threshold=0.96,
                    )
                    selected_images = {}
                    for frame in keyframes:
                        image_path = frame["image_path"]
                        normalized_path = os.path.normcase(
                            os.path.abspath(image_path)
                        )
                        with open(image_path, "rb") as image_file:
                            image_bytes = image_file.read()
                        selected_images[normalized_path] = image_bytes
                        frame["image_bytes"] = image_bytes
                        frame.pop("image_path", None)
                        frame.pop("path", None)
                        frame.pop("ocr_text", None)

                    ocr_results = []
                    for item in candidate_ocr:
                        image_path = item.get("image_path")
                        if not image_path:
                            continue
                        normalized_path = os.path.normcase(
                            os.path.abspath(image_path)
                        )
                        image_bytes = selected_images.get(normalized_path)
                        if image_bytes is None:
                            continue
                        item["image_bytes"] = image_bytes
                        ocr_results.append(item)
            except ingestion.FFMPEGNotFoundError:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "ffmpeg/ffprobe is required for video and OCR processing."
                    ),
                )
            except Exception:
                logger.exception(
                    "Video/OCR processing failed for %s; falling back to transcript summary",
                    url,
                )
                video_download_failed = True
        finally:
            video_temp_dir.cleanup()

    # Step 3: summarise using transcript only if OCR/video processing fails.
    has_ocr_text = any(str(item.get("text", "")).strip() for item in ocr_results)
    if video_download_failed or not has_ocr_text:
        summaries = fusion.summarize_chunks(chunks)
        overview = fusion.summarize_overview(summaries)
        md = fusion.assemble_markdown(
            summaries,
            keyframes,
            out_dir=out_root,
            title="Multimodal Video Summary",
            overview=overview,
            save_markdown=True,
        )
        return {
            "markdown": md,
            "source_url": url or None,
            "chunks": len(chunks),
            "mode": "transcript-only-fallback",
        }

    aligned = fusion.align_ocr_with_chunks(ocr_results, chunks)
    summaries = fusion.summarize_chunks(aligned)
    overview = fusion.summarize_overview(summaries)

    # Save and return the Markdown summary.
    md = fusion.assemble_markdown(
        summaries,
        keyframes,
        out_dir=out_root,
        title="Multimodal Video Summary",
        overview=overview,
        save_markdown=True,
    )
    return {
        "markdown": md,
        "source_url": url or None,
        "chunks": len(chunks),
        "mode": "transcript-and-ocr",
    }


@app.get("/images/{filename}")
def get_keyframe(filename: str):
    """Serve generated keyframes to the browser extension."""
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=404, detail="Image not found")

    image_root = os.path.realpath(
        os.path.join(os.getcwd(), "mevs_outputs", "images")
    )
    image_path = os.path.realpath(os.path.join(image_root, filename))
    if os.path.commonpath((image_root, image_path)) != image_root:
        raise HTTPException(status_code=404, detail="Image not found")
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    extension = os.path.splitext(filename)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if extension not in media_types:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        image_path,
        media_type=media_types[extension],
        headers={"Cache-Control": "no-store"},
    )


@app.post("/export/{export_format}")
async def export_summary(export_format: str, req: ExportRequest):
    markdown = req.markdown.strip()
    if not markdown:
        raise HTTPException(status_code=400, detail="Summary cannot be empty")
    if export_format not in {"docx", "pdf"}:
        raise HTTPException(
            status_code=400,
            detail="Export format must be 'docx' or 'pdf'",
        )

    out_root = os.path.abspath(os.path.join(os.getcwd(), "mevs_outputs"))
    try:
        if export_format == "docx":
            content = document_export.create_docx(markdown, out_root)
            media_type = (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
        else:
            content = document_export.create_pdf(markdown, out_root)
            media_type = "application/pdf"
    except Exception:
        logger.exception("Could not export summary as %s", export_format)
        raise HTTPException(
            status_code=500,
            detail="Could not create the requested summary file",
        )

    filename = f"MEVS-summary.{export_format}"
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/preflight-debug")
async def preflight_debug():
    """Endpoint used to verify browser preflight and CORS responses."""
    return {"ok": True}
