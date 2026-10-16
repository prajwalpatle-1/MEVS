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
