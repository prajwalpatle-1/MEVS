"""MEVS Module 1: video ingestion and audio processing.

Responsibilities:
- Fetch YouTube subtitles/captions when available.
- Fall back to audio download + Whisper transcription.
- Keep transcript timestamps.
- Clean transcript text.
- Chunk transcript into larger timestamped sections for BART.
- Delete temporary audio after transcription.

This module is intentionally independent from the visual pipeline. The visual
module can use the timestamped transcript chunks to guide important-frame
selection.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import imagehash
    from PIL import Image
except Exception:  # pragma: no cover
    imagehash = None
    Image = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover
    ssim = None

try:
    import easyocr
except Exception as exc:  # pragma: no cover
    easyocr = None
    _EASYOCR_IMPORT_ERROR = str(exc)
else:
    _EASYOCR_IMPORT_ERROR = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from scenedetect import detect, ContentDetector
except Exception:  # pragma: no cover
    detect = None
    ContentDetector = None

logger = logging.getLogger(__name__)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:  # pragma: no cover
    YouTubeTranscriptApi = None

try:
    import yt_dlp
except Exception:  # pragma: no cover
    yt_dlp = None

try:
    import whisper
except Exception:  # pragma: no cover
    whisper = None


class FFMPEGNotFoundError(Exception):
    """Raised when ffmpeg/ffprobe cannot be found."""


class VideoDownloadError(Exception):
    """Raised when audio/video download fails."""


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------


def _ffmpeg_in_path() -> Optional[str]:
    """Return a directory containing both ffmpeg and ffprobe, if available."""
    executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    probe = "ffprobe.exe" if os.name == "nt" else "ffprobe"

    for key in ("FFMPEG_PATH", "FFMPEG_LOCATION"):
        value = os.environ.get(key)
        if not value:
            continue

        path = value
        if os.path.isfile(path):
            path = os.path.dirname(path)

        if (
            os.path.isfile(os.path.join(path, executable))
            and os.path.isfile(os.path.join(path, probe))
        ):
            return os.path.abspath(path)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if ffmpeg and ffprobe:
        return os.path.abspath(os.path.dirname(ffmpeg))

    return None


def _download_and_extract_ffmpeg(dest_dir: str) -> Optional[str]:
    """Download and extract a lightweight FFmpeg build."""
    url = os.environ.get(
        "MEVS_FFMPEG_URL",
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    )

    tmp_zip = os.path.join(dest_dir, "ffmpeg_tmp.zip")

    try:
        os.makedirs(dest_dir, exist_ok=True)

        logger.info("Downloading ffmpeg build from %s", url)

        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()

        with open(tmp_zip, "wb") as output:
            output.write(data)

        logger.info("Extracting ffmpeg to %s", dest_dir)

        with zipfile.ZipFile(tmp_zip, "r") as archive:
            archive.extractall(dest_dir)

        try:
            os.remove(tmp_zip)
        except OSError:
            pass

        for root, _, files in os.walk(dest_dir):
            if "ffmpeg.exe" in files and "ffprobe.exe" in files:
                return os.path.abspath(root)

        return None

    except Exception:
        logger.exception("Automatic ffmpeg download/extract failed")
        return None

    finally:
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass


def ensure_ffmpeg_available() -> str:
    """Ensure ffmpeg and ffprobe are available and return their directory."""
    existing = _ffmpeg_in_path()
    if existing:
        return existing

    local_dir = os.path.join(
        os.path.abspath(os.getcwd()),
        ".mevs_ffmpeg",
    )

    # Reuse a previously auto-installed local copy.
    if os.path.isdir(local_dir):
        for root, _, files in os.walk(local_dir):
            if "ffmpeg.exe" in files and "ffprobe.exe" in files:
                os.environ["PATH"] = (
                    root
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )
                os.environ["FFMPEG_PATH"] = root
                return root

    auto_install = os.environ.get(
        "MEVS_AUTO_INSTALL_FFMPEG",
        "",
    ).lower()

    if auto_install not in {"1", "true", "yes", "on"}:
        raise FFMPEGNotFoundError(
            "ffmpeg/ffprobe not found. Install FFmpeg and add it to PATH "
            "or set FFMPEG_PATH/FFMPEG_LOCATION."
        )

    bin_path = _download_and_extract_ffmpeg(
        local_dir
    )

    if not bin_path:
        raise FFMPEGNotFoundError(
            "Automatic FFmpeg installation failed."
        )

    os.environ["PATH"] = (
        bin_path
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    os.environ["FFMPEG_PATH"] = bin_path

    logger.info(
        "ffmpeg available at %s",
        bin_path,
    )

    return bin_path


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------


def _extract_video_id(
    youtube_url: str,
) -> Optional[str]:
    """Extract an 11-character YouTube video ID."""
    if not youtube_url:
        return None

    patterns = (
        r"v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
        r"shorts/([0-9A-Za-z_-]{11})",
    )

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    match = re.search(
        r"([0-9A-Za-z_-]{11})$",
        youtube_url.rstrip("/"),
    )

    return match.group(1) if match else None


def _segment_to_dict(segment) -> Optional[Dict]:
    """Normalize a transcript object/dict into start/end/text."""
    try:
        if isinstance(segment, dict):
            text = str(segment.get("text", ""))
            start = float(segment.get("start", 0.0))
            duration = float(segment.get("duration", 0.0) or 0.0)
        else:
            text = str(getattr(segment, "text", ""))
            start = float(getattr(segment, "start", 0.0))
            duration = float(
                getattr(segment, "duration", 0.0) or 0.0
            )

        text = _clean_text(text)

        if not text:
            return None

        return {
            "start": start,
            "end": max(start, start + duration),
            "text": text,
        }
    except Exception:
        logger.debug(
            "Could not normalize transcript segment",
            exc_info=True,
        )
        return None


def fetch_subtitles(
    youtube_url: str,
) -> Optional[List[Dict]]:
    """Fetch YouTube captions when available."""
    if YouTubeTranscriptApi is None:
        logger.warning(
            "youtube-transcript-api is not installed"
        )
        return None

    video_id = _extract_video_id(youtube_url)

    if not video_id:
        logger.error(
            "Could not parse YouTube video ID from %s",
            youtube_url,
        )
        return None

    try:
        # youtube-transcript-api 1.x
        api = YouTubeTranscriptApi()

        if hasattr(api, "list"):
            tracks = list(api.list(video_id))

            preferred = next(
                (
                    track
                    for track in tracks
                    if getattr(track, "language_code", "") == "en"
                ),
                None,
            )

            if preferred is None:
                preferred = next(
                    (
                        track
                        for track in tracks
                        if str(
                            getattr(track, "language_code", "")
                        ).startswith("en-")
                    ),
                    None,
                )

            if preferred is None and tracks:
                preferred = tracks[0]

            if preferred is None:
                return None

            transcript = preferred.fetch()

            if hasattr(transcript, "to_raw_data"):
                transcript = transcript.to_raw_data()

        else:
            # Older youtube-transcript-api versions.
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id
            )

        normalized: List[Dict] = []

        for segment in transcript:
            item = _segment_to_dict(segment)
            if item:
                normalized.append(item)

        if not normalized:
            return None

        logger.info(
            "Fetched %d YouTube subtitle segments",
            len(normalized),
        )

        return normalized

    except Exception as exc:
        error_name = type(exc).__name__

        if error_name in {
            "TranscriptsDisabled",
            "NoTranscriptFound",
            "NoTranscriptAvailable",
        }:
            logger.info(
                "No YouTube transcript available for %s",
                video_id,
            )
            return None

        logger.warning(
            "YouTube subtitle fetch failed: %s",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Audio + Whisper
# ---------------------------------------------------------------------------


def download_audio(
    youtube_url: str,
    out_path: Optional[str] = None,
) -> Optional[str]:
    """Download audio to a temporary path unless out_path is supplied."""
    if yt_dlp is None:
        raise VideoDownloadError(
            "yt-dlp is not installed."
        )

    temporary = out_path is None

    if out_path is None:
        temp_dir = tempfile.mkdtemp(
            prefix="mevs_audio_"
        )
        output_stem = os.path.join(
            temp_dir,
            "audio",
        )
        out_path = output_stem + ".mp3"
    else:
        output_stem, _ = os.path.splitext(
            out_path
        )
        parent = os.path.dirname(
            os.path.abspath(out_path)
        )
        os.makedirs(parent, exist_ok=True)

    output_template = output_stem + ".%(ext)s"

    ffmpeg_dir = ensure_ffmpeg_available()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "ffmpeg_location": ffmpeg_dir,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
    }

    node_path = shutil.which("node")
    if node_path:
        ydl_opts["js_runtimes"] = {
            "node": {
                "path": node_path
            }
        }

    player_client = os.environ.get(
        "MEVS_YTDLP_PLAYER_CLIENT"
    )

    browser = os.environ.get(
        "MEVS_YTDLP_BROWSER"
    )

    cookie_file = os.environ.get(
        "MEVS_YTDLP_COOKIE_FILE"
    )

    if player_client:
        ydl_opts["extractor_args"] = {
            "youtube": {
                "player_client": [player_client]
            }
        }

    if browser:
        ydl_opts["cookiesfrombrowser"] = (
            browser,
        )
    elif cookie_file:
        ydl_opts["cookiefile"] = cookie_file

    try:
        logger.info(
            "Downloading audio%s",
            " (temporary)" if temporary else "",
        )

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        if (
            os.path.exists(out_path)
            and os.path.getsize(out_path) > 0
        ):
            return out_path

        raise VideoDownloadError(
            "yt-dlp completed but the audio file was not created."
        )

    except FFMPEGNotFoundError:
        raise
    except Exception as exc:
        error_text = str(exc).lower()

        if (
            isinstance(exc, FileNotFoundError)
            or (
                "not found" in error_text
                and (
                    "ffmpeg" in error_text
                    or "ffprobe" in error_text
                )
            )
        ):
            raise FFMPEGNotFoundError(
                "ffmpeg/ffprobe is required for audio extraction."
            ) from exc

        raise VideoDownloadError(
            f"Failed to download audio: {exc}"
        ) from exc


def cleanup_temp_audio(
    audio_path: Optional[str],
) -> None:
    """Remove only MEVS temporary audio directories."""
    if not audio_path:
        return

    try:
        path = os.path.abspath(audio_path)
        parent = os.path.dirname(path)
        parent_name = os.path.basename(parent)

        if not parent_name.startswith(
            "mevs_audio_"
        ):
            return

        if os.path.exists(path):
            os.remove(path)

        if os.path.isdir(parent):
            shutil.rmtree(
                parent,
                ignore_errors=True,
            )

        logger.info(
            "Removed temporary audio directory %s",
            parent,
        )

    except Exception:
        logger.debug(
            "Could not clean temporary audio %s",
            audio_path,
            exc_info=True,
        )


def sample_time_ranges(
    video_path: str,
    interval_seconds: float = 30.0,
) -> List[tuple[float, float]]:
    """Return evenly spaced time ranges without using scene detection."""
    if cv2 is None:
        return [(0.0, 0.0)]

    capture = cv2.VideoCapture(video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames else 100.0
        if duration <= 0:
            duration = 100.0

        if duration <= interval_seconds:
            return [(0.0, float(duration))]

        ranges: List[tuple[float, float]] = []
        start = 0.0
        while start < duration:
            end = min(start + interval_seconds, duration)
            ranges.append((float(start), float(end)))
            start = end
        return ranges
    finally:
        capture.release()


def sample_video_ranges(
    video_path: str,
    scene_count: int = 8,
) -> List[tuple[float, float]]:
    """Return evenly spaced time ranges for fixed-interval sampling."""
    if scene_count <= 1:
        return [(0.0, float(sample_time_ranges(video_path)[-1][1]))]

    if cv2 is None:
        return [(0.0, 0.0)]

    capture = cv2.VideoCapture(video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames else 100.0
        if duration <= 0:
            duration = 100.0

        ranges: List[tuple[float, float]] = []
        step = duration / scene_count
        for index in range(scene_count):
            start = index * step
            end = (index + 1) * step if index < scene_count - 1 else duration
            ranges.append((float(start), float(end)))
        return ranges
    finally:
        capture.release()


def detect_scenes(video_path: str, threshold: int = 30):
    """Detect scenes using the modern scenedetect API if available."""
    if detect is None or ContentDetector is None:
        return []
    detector = ContentDetector(threshold=threshold)
    scenes = detect(video_path, detector)
    return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]


def download_video(
    youtube_url: str,
    out_path: Optional[str] = None,
) -> Optional[str]:
    """Download a YouTube video, retrying with another player client on failure."""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed")

    target = out_path or os.path.join(tempfile.mkdtemp(prefix="mevs_video_"), "video.mp4")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": target,
        "quiet": True,
        "noplaylist": True,
    }

    js_runtimes = {}
    for runtime in ("deno", "node"):
        runtime_path = shutil.which(runtime)
        if runtime_path:
            js_runtimes[runtime] = {"path": runtime_path}
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes

    player_client = os.environ.get("MEVS_YTDLP_PLAYER_CLIENT")
    browser = os.environ.get("MEVS_YTDLP_BROWSER")
    cookie_file = os.environ.get("MEVS_YTDLP_COOKIE_FILE")
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)
    elif cookie_file:
        ydl_opts["cookiefile"] = cookie_file

    extractor_attempts = (
        [{"youtube": {"player_client": [player_client]}}]
        if player_client
        else [None, {"youtube": {"player_client": ["android"]}}]
    )
    last_error = None
    for extractor_args in extractor_attempts:
        try:
            options = dict(ydl_opts)
            if extractor_args:
                options["extractor_args"] = extractor_args
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([youtube_url])
            if os.path.exists(target) and os.path.getsize(target) > 0:
                return target
            last_error = RuntimeError("yt-dlp produced no video file")
        except Exception as exc:
            last_error = exc
        if extractor_args != extractor_attempts[-1]:
            logger.warning(
                "YouTube video attempt failed (%s); retrying with an "
                "alternate player client",
                last_error,
            )

    raise last_error or RuntimeError("YouTube video download failed")


def extract_keyframes(video_path: str, ranges, out_dir: str):
    """Save one midpoint frame for every sampled video time range."""
    if cv2 is None:
        logger.error("OpenCV is not installed; cannot extract video frames")
        return []

    os.makedirs(out_dir, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if hasattr(capture, "isOpened") and not capture.isOpened():
        capture.release()
        logger.error("Could not open video for keyframe extraction: %s", video_path)
        return []

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    keyframes = []
    try:
        for index, (start, end) in enumerate(ranges):
            start = float(start)
            end = float(end)
            timestamp = max(0.0, (start + end) / 2.0)
            frame_number = int(timestamp * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success or frame is None:
                logger.warning(
                    "Could not read frame at %.2f seconds from %s",
                    timestamp,
                    video_path,
                )
                continue

            image_path = os.path.abspath(
                os.path.join(out_dir, f"kf_{index:04d}.jpg")
            )
            if not cv2.imwrite(image_path, frame):
                logger.warning("Could not save keyframe: %s", image_path)
                continue

            keyframes.append(
                {
                    "image_path": image_path,
                    "path": image_path,
                    "start": start,
                    "end": end,
                    "timestamp": timestamp,
                }
            )
    finally:
        capture.release()

    logger.info("Extracted %d keyframes from %s", len(keyframes), video_path)
    return keyframes


def _are_similar_frames(frame_a, frame_b, hash_threshold=5, ssim_threshold=0.92):
    path_a = frame_a.get("image_path") or frame_a.get("path")
    path_b = frame_b.get("image_path") or frame_b.get("path")
    if not path_a or not path_b:
        return False
    if os.path.normcase(os.path.abspath(path_a)) == os.path.normcase(
        os.path.abspath(path_b)
    ):
        return True

    text_a = {
        word.lower()
        for word in str(frame_a.get("ocr_text", "")).split()
        if len(word) > 2
    }
    text_b = {
        word.lower()
        for word in str(frame_b.get("ocr_text", "")).split()
        if len(word) > 2
    }
    if len(text_a) >= 6 and len(text_b) >= 6:
        overlap = len(text_a & text_b) / len(text_a | text_b)
        if overlap >= 0.9:
            return True

    if imagehash is not None and Image is not None:
        try:
            hash_a = imagehash.phash(Image.open(path_a))
            hash_b = imagehash.phash(Image.open(path_b))
            if abs(hash_a - hash_b) <= hash_threshold:
                return True
        except Exception:
            logger.debug("Could not compare perceptual hashes", exc_info=True)

    if cv2 is None or ssim is None:
        return False
    try:
        image_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        if image_a is None or image_b is None:
            return False
        height = min(image_a.shape[0], image_b.shape[0])
        width = min(image_a.shape[1], image_b.shape[1])
        if height < 1 or width < 1:
            return False
        image_a = cv2.resize(image_a, (width, height))
        image_b = cv2.resize(image_b, (width, height))
        return ssim(image_a, image_b, data_range=255) >= ssim_threshold
    except Exception:
        logger.debug("Could not compare frame similarity", exc_info=True)
        return False


def filter_similar_frames(frames, hash_threshold=5, ssim_threshold=0.92):
    """Drop frames that are visually near-identical to an earlier frame."""
    kept = []
    for frame in frames:
        if any(
            _are_similar_frames(
                frame,
                previous,
                hash_threshold=hash_threshold,
                ssim_threshold=ssim_threshold,
            )
            for previous in kept
        ):
            continue
        kept.append(frame)
    logger.info("Filtered %d keyframes to %d", len(frames), len(kept))
    return kept


def run_ocr_on_frames(frames, lang_list=None):
    """Extract frame text with RapidOCR, then EasyOCR or Tesseract fallback."""
    reader = None
    reader_name = None
    try:
        reader = _load_rapidocr_engine()
        reader_name = "rapidocr"
    except Exception as exc:
        logger.warning("RapidOCR could not initialize: %s", exc)

    if reader is None and easyocr is not None:
        try:
            reader = easyocr.Reader(lang_list or ["en"], gpu=False)
            reader_name = "easyocr"
        except Exception:
            logger.warning("EasyOCR could not initialize; trying Tesseract", exc_info=True)

    tesseract_command = None
    if pytesseract is not None:
        configured_command = os.environ.get("TESSERACT_CMD")
        if configured_command:
            pytesseract.pytesseract.tesseract_cmd = configured_command
        configured_command = pytesseract.pytesseract.tesseract_cmd
        tesseract_command = shutil.which(configured_command)
        if not tesseract_command and os.path.isfile(configured_command):
            tesseract_command = configured_command

    results = []
    if reader is None and tesseract_command is None:
        logger.warning(
            "OCR unavailable: RapidOCR, EasyOCR, and Tesseract could not be "
            "started (%s). Keyframe images will still be included.",
            _EASYOCR_IMPORT_ERROR or "not installed",
        )

    for frame in frames:
        image_path = frame.get("image_path") or frame.get("path")
        text = ""
        if image_path:
            if reader_name == "rapidocr":
                try:
                    detected = reader(image_path)
                    text = " ".join(
                        str(value).strip()
                        for value in (getattr(detected, "txts", None) or [])
                        if str(value).strip()
                    )
                except Exception:
                    logger.warning("RapidOCR failed on %s", image_path)
            elif reader_name == "easyocr":
                try:
                    detected = reader.readtext(image_path, detail=0)
                    text = " ".join(
                        str(value).strip()
                        for value in detected
                        if str(value).strip()
                    )
                except Exception:
                    logger.warning(
                        "EasyOCR failed on %s; trying Tesseract",
                        image_path,
                        exc_info=True,
                    )
            if (
                not text
                and tesseract_command is not None
                and pytesseract is not None
                and Image is not None
            ):
                try:
                    with Image.open(image_path) as image:
                        text = pytesseract.image_to_string(image)
                except Exception:
                    logger.warning("Tesseract failed on %s", image_path, exc_info=True)

        results.append(
            {
                "image_path": image_path,
                "start": float(frame.get("start", frame.get("timestamp", 0.0))),
                "end": float(frame.get("end", frame.get("timestamp", 0.0))),
                "timestamp": float(frame.get("timestamp", frame.get("start", 0.0))),
                "text": " ".join(text.split()),
            }
        )
    return results


@lru_cache(maxsize=2)
def _load_clip_ranker(model_name):
    """Load a cached CPU CLIP model and processor for frame relevance scoring."""
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model, processor


def _frame_context(frame, transcript_chunks):
    timestamp = float(frame.get("timestamp", frame.get("start", 0.0)))
    if not transcript_chunks:
        return ""

    def distance(chunk):
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))
        if start <= timestamp <= end:
            return 0.0
        return min(abs(timestamp - start), abs(timestamp - end))

    chunk = min(transcript_chunks, key=distance)
    transcript = " ".join(str(chunk.get("text", "")).split())
    return transcript[:450]


def _score_frames_with_clip(frames, transcript_chunks, model_name):
    import torch

    model, processor = _load_clip_ranker(model_name)
    images = []
    descriptions = []
    valid_frames = []
    for frame in frames:
        image_path = frame.get("image_path") or frame.get("path")
        if not image_path or Image is None:
            continue
        try:
            with Image.open(image_path) as source:
                images.append(source.convert("RGB"))
            context = _frame_context(frame, transcript_chunks)
            slide_text = " ".join(str(frame.get("ocr_text", "")).split())
            description = "Educational lecture image"
            if context:
                description += f" about {context}"
            if slide_text:
                description += f". Visible slide text: {slide_text[:250]}"
            descriptions.append(description)
            valid_frames.append(frame)
        except Exception:
            logger.debug("Could not prepare image for CLIP: %s", image_path)

    if not valid_frames:
        return []

    inputs = processor(
        text=descriptions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.inference_mode():
        outputs = model(**inputs)
        scores = outputs.logits_per_image.diagonal().cpu().tolist()
    return list(zip(valid_frames, scores))


def _fallback_frame_scores(frames):
    """Score readable, sharp, high-contrast frames if CLIP is unavailable."""
    measurements = []
    for frame in frames:
        text_score = min(len(str(frame.get("ocr_text", "")).split()) / 20, 1.0)
        sharpness = 0.0
        contrast = 0.0
        image_path = frame.get("image_path") or frame.get("path")
        if cv2 is not None and image_path:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
                contrast = float(image.std())
        measurements.append((frame, text_score, sharpness, contrast))

    def normalize(values):
        low = min(values, default=0.0)
        high = max(values, default=0.0)
        if high <= low:
            return [0.5] * len(values)
        return [(value - low) / (high - low) for value in values]

    text_scores = normalize([row[1] for row in measurements])
    sharpness_scores = normalize([row[2] for row in measurements])
    contrast_scores = normalize([row[3] for row in measurements])
    return [
        (
            row[0],
            0.55 * text_score + 0.30 * sharpness_score + 0.15 * contrast_score,
        )
        for row, text_score, sharpness_score, contrast_score in zip(
            measurements, text_scores, sharpness_scores, contrast_scores
        )
    ]


def select_important_frames(
    frames,
    transcript_chunks,
    max_frames=12,
    model_name=None,
):
    """Rank candidates by CLIP relevance and keep the best across time bins."""
    if not frames or max_frames <= 0:
        return []

    model_name = model_name or os.environ.get(
        "MEVS_FRAME_RANKER_MODEL", "openai/clip-vit-base-patch32"
    )
    try:
        scored_frames = _score_frames_with_clip(
            frames, transcript_chunks, model_name
        )
        if not scored_frames:
            raise RuntimeError("CLIP could not score any candidate frames")
    except Exception as exc:
        logger.warning(
            "CLIP frame ranking unavailable (%s); using OCR and image quality",
            exc,
        )
        scored_frames = _fallback_frame_scores(frames)

    scored_frames = [
        ({**frame, "importance_score": float(score)}, float(score))
        for frame, score in scored_frames
    ]
    frame_limit = min(int(max_frames), len(scored_frames))
    duration = max(
        (float(frame.get("end", frame.get("timestamp", 0.0)))
         for frame, _ in scored_frames),
        default=0.0,
    )
    bin_size = duration / frame_limit if duration > 0 else 1.0
    bins = {}
    for frame, score in scored_frames:
        timestamp = float(frame.get("timestamp", frame.get("start", 0.0)))
        bin_index = min(frame_limit - 1, int(timestamp / bin_size))
        if bin_index not in bins or score > bins[bin_index][1]:
            bins[bin_index] = (frame, score)

    selected = [item[0] for _, item in sorted(bins.items())]
    selected_ids = {id(frame) for frame in selected}
    remaining = sorted(
        (
            item
            for item in scored_frames
            if id(item[0]) not in selected_ids
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    for frame, _ in remaining:
        if len(selected) >= frame_limit:
            break
        selected.append(frame)

    selected.sort(
        key=lambda frame: float(frame.get("timestamp", frame.get("start", 0.0)))
    )
    logger.info(
        "Ranked %d candidate frames and selected %d using %s",
        len(frames),
        len(selected),
        model_name if scored_frames else "local image quality fallback",
    )
    return selected


@lru_cache(maxsize=1)
def _load_rapidocr_engine():
    """Create one cached RapidOCR engine using the ONNX Runtime backend."""
    from rapidocr import RapidOCR

    return RapidOCR()


@lru_cache(maxsize=4)
def _load_whisper_model(
    model_name: str,
):
    """Cache Whisper models so repeated requests do not reload them."""
    if whisper is None:
        raise RuntimeError(
            "openai-whisper is not installed."
        )

    logger.info(
        "Loading Whisper model '%s'",
        model_name,
    )

    ensure_ffmpeg_available()

    return whisper.load_model(
        model_name
    )


def transcribe_with_whisper(
    audio_path: str,
    model_name: str = "small",
    language: Optional[str] = None,
) -> Optional[List[Dict]]:
    """Transcribe audio and preserve segment timestamps."""
    if whisper is None:
        logger.error(
            "openai-whisper is not installed"
        )
        return None

    if not os.path.exists(audio_path):
        logger.error(
            "Audio file not found: %s",
            audio_path,
        )
        return None

    try:
        model = _load_whisper_model(
            model_name
        )

        logger.info(
            "Transcribing audio: %s",
            audio_path,
        )

        options = {
            "verbose": False,
        }

        if language:
            options["language"] = language

        result = model.transcribe(
            audio_path,
            **options,
        )

        formatted: List[Dict] = []

        for segment in result.get(
            "segments",
            [],
        ):
            text = _clean_text(
                segment.get(
                    "text",
                    "",
                )
            )

            if not text:
                continue

            start = float(
                segment.get(
                    "start",
                    0.0,
                )
            )

            end = float(
                segment.get(
                    "end",
                    start,
                )
            )

            formatted.append(
                {
                    "start": start,
                    "end": max(
                        start,
                        end,
                    ),
                    "text": text,
                }
            )

        logger.info(
            "Whisper produced %d segments",
            len(formatted),
        )

        return formatted

    except FileNotFoundError as exc:
        raise FFMPEGNotFoundError(
            "ffmpeg was not found while running Whisper."
        ) from exc
    except Exception:
        logger.exception(
            "Whisper transcription failed"
        )
        return None


# ---------------------------------------------------------------------------
# Text cleaning + chunking
# ---------------------------------------------------------------------------


def _clean_text(
    text: str,
) -> str:
    """Normalize whitespace and basic caption artifacts."""
    if not text:
        return ""

    text = str(text)

    # Remove common HTML-like caption artifacts.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _remove_repeated_phrases(
    text: str,
) -> str:
    """Remove common adjacent repeated caption chunks."""
    words = text.split()

    if len(words) < 8:
        return text.strip()

    changed = True

    while changed:
        changed = False
        max_size = min(
            20,
            len(words) // 2,
        )

        for size in range(
            max_size,
            3,
            -1,
        ):
            left = [
                word.lower().strip(",.!?;:")
                for word in words[
                    -2 * size:-size
                ]
            ]

            right = [
                word.lower().strip(",.!?;:")
                for word in words[-size:]
            ]

            if left == right:
                words = words[:-size]
                changed = True
                break

    return " ".join(words).strip()


def normalize_segments(
    segments: Sequence[Dict],
) -> List[Dict]:
    """Sort, clean and merge tiny adjacent transcript segments."""
    cleaned: List[Dict] = []

    for segment in segments:
        text = _clean_text(
            segment.get("text", "")
        )

        if not text:
            continue

        start = float(
            segment.get(
                "start",
                0.0,
            )
        )

        end = float(
            segment.get(
                "end",
                start,
            )
        )

        cleaned.append(
            {
                "start": max(0.0, start),
                "end": max(start, end),
                "text": text,
            }
        )

    cleaned.sort(
        key=lambda item: item["start"]
    )

    return cleaned


def chunk_segments(
    segments: List[Dict],
    chunk_duration: int = 120,
    max_words: int = 260,
) -> List[Dict]:
    """Group transcript segments into timestamped chunks.

    Target chunk length is approximately 2 minutes. The max_words guard keeps
    individual BART requests bounded and avoids unnecessarily large prompts.
    """
    if not segments:
        return []

    segments = normalize_segments(
        segments
    )

    chunks: List[Dict] = []
    current_text: List[str] = []
    current_words = 0
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for segment in segments:
        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        text = _remove_repeated_phrases(
            segment["text"]
        )

        if not text:
            continue

        words = len(
            text.split()
        )

        if current_start is None:
            current_start = start
            current_end = end

        should_flush = (
            bool(current_text)
            and (
                start - current_start >= chunk_duration
                or current_words + words > max_words
            )
        )

        if should_flush:
            chunks.append(
                {
                    "start": float(current_start),
                    "end": float(current_end),
                    "text": " ".join(current_text).strip(),
                }
            )

            current_text = []
            current_words = 0
            current_start = start
            current_end = end

        current_text.append(text)
        current_words += words
        current_end = max(
            float(current_end),
            end,
        )

    if current_text and current_start is not None:
        chunks.append(
            {
                "start": float(current_start),
                "end": float(current_end or current_start),
                "text": " ".join(current_text).strip(),
            }
        )

    logger.info(
        "Created %d transcript chunks (target=%ds, max_words=%d)",
        len(chunks),
        chunk_duration,
        max_words,
    )

    return chunks


def chunk_plain_text(
    text: str,
    chunk_duration: int = 120,
    max_words: int = 260,
) -> List[Dict]:
    """Chunk pasted transcript text, which has no reliable timestamps."""
    paragraphs = [
        _clean_text(part)
        for part in re.split(
            r"\n\s*\n",
            text or "",
        )
    ]

    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if paragraph
    ]

    if not paragraphs:
        return []

    chunks: List[Dict] = []
    current: List[str] = []
    current_words = 0
    start = 0.0

    for paragraph in paragraphs:
        paragraph_words = len(
            paragraph.split()
        )

        if (
            current
            and current_words
            + paragraph_words
            > max_words
        ):
            end = start + chunk_duration

            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "text": " ".join(current),
                }
            )

            start = end
            current = []
            current_words = 0

        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunks.append(
            {
                "start": start,
                "end": start + chunk_duration,
                "text": " ".join(current),
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Public transcript pipeline
# ---------------------------------------------------------------------------


def get_transcript_chunks(
    youtube_url: str,
    prefer_subtitles: bool = True,
    whisper_model: str = "small",
    allow_whisper_fallback: bool = True,
    chunk_duration: int = 120,
) -> Optional[List[Dict]]:
    """Get timestamped transcript chunks.

    Order:
        1. YouTube subtitles/captions.
        2. Temporary audio download.
        3. Cached Whisper model transcription.
        4. Chunking for BART.

    Temporary audio is always removed after Whisper processing.
    """
    if not youtube_url:
        return None

    if prefer_subtitles:
        subtitles = fetch_subtitles(
            youtube_url
        )

        if subtitles:
            logger.info(
                "Using YouTube subtitles; Whisper skipped"
            )

            return chunk_segments(
                subtitles,
                chunk_duration=chunk_duration,
            )

    if not allow_whisper_fallback:
        logger.info(
            "No usable subtitles and Whisper fallback is disabled"
        )
        return None

    audio_path: Optional[str] = None

    try:
        audio_path = download_audio(
            youtube_url
        )

        segments = transcribe_with_whisper(
            audio_path,
            model_name=whisper_model,
        )

        if not segments:
            logger.error(
                "Whisper returned no transcript segments"
            )
            return None

        return chunk_segments(
            segments,
            chunk_duration=chunk_duration,
        )

    except (
        FFMPEGNotFoundError,
        VideoDownloadError,
    ):
        logger.exception(
            "Unable to obtain audio/transcript"
        )
        raise

    finally:
        cleanup_temp_audio(
            audio_path
        )


def get_transcript_segments(
    youtube_url: str,
    prefer_subtitles: bool = True,
    whisper_model: str = "small",
    allow_whisper_fallback: bool = True,
) -> Optional[List[Dict]]:
    """Return raw/near-raw timestamped transcript segments.

    Useful when the visual module needs finer timing than the 2-minute BART
    chunks. The same subtitle-first / Whisper-fallback behavior is used.
    """
    if not youtube_url:
        return None

    if prefer_subtitles:
        subtitles = fetch_subtitles(
            youtube_url
        )

        if subtitles:
            return subtitles

    if not allow_whisper_fallback:
        return None

    audio_path: Optional[str] = None

    try:
        audio_path = download_audio(
            youtube_url
        )

        return transcribe_with_whisper(
            audio_path,
            model_name=whisper_model,
        )

    finally:
        cleanup_temp_audio(
            audio_path
        )
