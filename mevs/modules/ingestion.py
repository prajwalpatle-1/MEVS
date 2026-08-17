"""Module 1: Video ingestion and audio processing.

Provides utilities to fetch existing YouTube subtitles. It can also
download audio using `yt-dlp`, transcribe using OpenAI Whisper, perform
basic cleaning, and chunk transcripts into timestamped segments. Default
chunk size is approximately 2 minutes.

The module aims to be production-ready with clear error handling and
informative logging.
"""
from __future__ import annotations

import os
import re
import tempfile
import logging
from typing import List, Dict, Optional
import shutil
import zipfile
import urllib.request
import sys

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptAvailable,
    )
except Exception:  # pragma: no cover - allow graceful import failure
    YouTubeTranscriptApi = None

try:
    import yt_dlp
except Exception:  # pragma: no cover
    yt_dlp = None

try:
    import whisper
except Exception:  # pragma: no cover
    whisper = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FFMPEGNotFoundError(Exception):
    """Raised when ffmpeg/ffprobe are not available on the system."""


def _ffmpeg_in_path() -> Optional[str]:
    """Return path to ffmpeg binary if available, otherwise None."""
    # check explicit env overrides first
    for key in ("FFMPEG_PATH", "FFMPEG_LOCATION"):
        p = os.environ.get(key)
        if p:
            # if a bin path was provided, accept it
            candidate = os.path.join(p, "ffmpeg.exe") if os.name == "nt" else os.path.join(p, "ffmpeg")
            if os.path.exists(candidate):
                return os.path.abspath(p)
            # maybe the value points directly to the binary
            if os.path.exists(p):
                return os.path.abspath(os.path.dirname(p))
    # fallback to PATH
    ff = shutil.which("ffmpeg")
    if ff:
        return os.path.abspath(os.path.dirname(ff))
    return None


def _download_and_extract_ffmpeg(dest_dir: str) -> Optional[str]:
    """Download a lightweight ffmpeg build for Windows and extract it to dest_dir.

    Returns the path to the `bin` folder on success, or None on failure.
    """
    # Use a reputable static build; adjust URL if needed for newer releases.
    # Note: network download may be blocked in some environments.
    url = os.environ.get(
        "MEVS_FFMPEG_URL",
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    )
    try:
        os.makedirs(dest_dir, exist_ok=True)
        tmpzip = os.path.join(dest_dir, "ffmpeg_tmp.zip")
        logger.info("Downloading ffmpeg build from %s", url)
        with urllib.request.urlopen(url) as resp, open(tmpzip, "wb") as out:
            out.write(resp.read())
        logger.info("Extracting ffmpeg to %s", dest_dir)
        with zipfile.ZipFile(tmpzip, "r") as z:
            z.extractall(dest_dir)
        os.remove(tmpzip)
        # find bin folder inside extracted tree
        for root, dirs, files in os.walk(dest_dir):
            if "ffmpeg.exe" in files or "ffmpeg" in files:
                # bin is the parent containing ffmpeg
                return os.path.abspath(root)
        return None
    except Exception:
        logger.exception("Automatic ffmpeg download/extract failed")
        return None


def ensure_ffmpeg_available() -> str:
    """Ensure ffmpeg is available and return the bin folder path.

    If ffmpeg isn't present and the environment variable `MEVS_AUTO_INSTALL_FFMPEG`
    is set to a truthy value, this will attempt to download a static build into
    `.mevs_ffmpeg/` inside the project and add it to the PATH for the running
    process.
    """
    path = _ffmpeg_in_path()
    if path:
        return path

    auto = os.environ.get("MEVS_AUTO_INSTALL_FFMPEG")
    if not auto or auto.lower() not in ("1", "true", "yes"):
        raise FFMPEGNotFoundError("ffmpeg not found on PATH or FFMPEG_PATH not set")

    # attempt automatic download
    proj_dir = os.path.abspath(os.getcwd())
    dest = os.path.join(proj_dir, ".mevs_ffmpeg")
    bin_path = _download_and_extract_ffmpeg(dest)
    if not bin_path:
        raise FFMPEGNotFoundError("Automatic ffmpeg install failed")

    # prepend to PATH for current process so subprocess calls resolve ffmpeg
    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
    # also set FFMPEG_PATH for yt-dlp option
    os.environ["FFMPEG_PATH"] = bin_path
    logger.info("ffmpeg available at %s and added to PATH", bin_path)
    return bin_path



def _extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract the YouTube video id from a URL.

    Returns None if no id could be parsed.
    """
    if not youtube_url:
        return None
    # common patterns
    patterns = [
        r"v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, youtube_url)
        if m:
            return m.group(1)
    # fallback: last path component
    m = re.search(r"([0-9A-Za-z_-]{11})$", youtube_url)
    return m.group(1) if m else None


def fetch_subtitles(youtube_url: str) -> Optional[List[Dict]]:
    """Try to fetch existing YouTube subtitles via `youtube-transcript-api`.

    Returns a list of segments. Example segment keys:
    - 'text' (str)
    - 'start' (float)
    - 'duration' (float)
    or ``None`` if no subtitles are available.
    """
    if YouTubeTranscriptApi is None:
        logger.warning("youtube_transcript_api not installed")
        return None
    vid = _extract_video_id(youtube_url)
    if not vid:
        logger.error("Could not parse video id from URL: %s", youtube_url)
        return None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid)
        # transcript entries already contain 'text', 'start', 'duration'
        seg_count = len(transcript)
        msg = "Fetched %d subtitle segments from YouTube"
        logger.info(msg, seg_count)
        return transcript
    except TranscriptsDisabled:
        logger.info("Transcripts are disabled for video %s", vid)
        return None
    except NoTranscriptAvailable:
        logger.info("No transcript available for video %s", vid)
        return None
    except Exception as exc:
        logger.exception("Unexpected error fetching transcript: %s", exc)
        return None


def download_audio(
    youtube_url: str, out_path: Optional[str] = None
) -> Optional[str]:
    """Download audio from YouTube using `yt-dlp` and return local file path.

    out_path: optional absolute path for resulting file. If omitted, a
    temporary file is created and its path returned.
    """
    if yt_dlp is None:
        logger.error("yt_dlp not available; please install yt-dlp")
        return None
    if out_path is None:
        fd, temp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        out_path = temp

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    # Allow overriding ffmpeg/ffprobe location via environment for systems
    # where ffmpeg is not on PATH (useful on Windows).
    ffmpeg_loc = os.environ.get("FFMPEG_PATH") or os.environ.get("FFMPEG_LOCATION")
    if not ffmpeg_loc:
        # ensure availability or auto-install if configured
        try:
            ffmpeg_loc = ensure_ffmpeg_available()
        except FFMPEGNotFoundError:
            ffmpeg_loc = None
    if ffmpeg_loc:
        ydl_opts["ffmpeg_location"] = ffmpeg_loc
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            msg = "Downloading audio for %s to %s"
            logger.info(msg, youtube_url, out_path)
            ydl.download([youtube_url])
        if os.path.exists(out_path):
            msg = "Audio downloaded to %s"
            logger.info(msg, out_path)
            return out_path
        logger.error(
            "yt-dlp reported success but output file missing: %s", out_path
        )
        return None
    except Exception as exc:
        # Detect common ffmpeg/ffprobe missing failure from yt-dlp
        emsg = str(exc).lower()
        if "ffprobe" in emsg or "ffmpeg" in emsg:
            logger.exception("ffmpeg/ffprobe not found for yt-dlp: %s", exc)
            raise FFMPEGNotFoundError(
                "ffmpeg/ffprobe not found. Install ffmpeg or set FFMPEG_PATH."
            )
        logger.exception("Failed to download audio with yt-dlp")
        return None


def transcribe_with_whisper(
    audio_path: str, model_name: str = "small"
) -> Optional[List[Dict]]:
    """Transcribe audio using OpenAI Whisper. Returns list of segments.

    Each segment: {'start': float, 'end': float, 'text': str}
    """
    if whisper is None:
        logger.error("whisper package is not installed")
        return None
    if not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        return None
    try:
        logger.info(
            "Loading Whisper model '%s' (this may take time)...",
            model_name,
        )
        # Ensure ffmpeg available for Whisper subprocesses
        try:
            ensure_ffmpeg_available()
        except FFMPEGNotFoundError:
            # let the subsequent FileNotFoundError handler raise a clearer error
            pass
        model = whisper.load_model(model_name)
        logger.info("Transcribing audio: %s", audio_path)
        result = model.transcribe(audio_path)
        segments = result.get("segments", [])
        formatted = []
        for seg in segments:
            formatted.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", "").strip(),
                }
            )
        nseg = len(formatted)
        msg = "Whisper produced %d segments"
        logger.info(msg, nseg)
        return formatted
    except FileNotFoundError as exc:
        # Whisper/ffmpeg subprocess may raise FileNotFoundError when ffmpeg
        # is not available on PATH. Surface a clearer error to the caller.
        logger.exception("Whisper failed due to missing system binary: %s", exc)
        raise FFMPEGNotFoundError(
            "ffmpeg not found for Whisper. Install ffmpeg or set FFMPEG_PATH."
        )
    except Exception:
        logger.exception("Whisper transcription failed")
        return None


def _clean_text(text: str) -> str:
    """Simple cleaning for transcript/ocr text."""
    if not text:
        return ""
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # strip leading/trailing
    return text.strip()


def chunk_segments(
    segments: List[Dict], chunk_duration: int = 120
) -> List[Dict]:
    """Chunk a list of timestamped segments into larger timestamped chunks.

    Each returned chunk is: {'start': float, 'end': float, 'text': str}
    chunk_duration is in seconds (default 120 => 2 minutes)
    """
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.get("start", 0.0))
    chunks: List[Dict] = []
    cur_text_parts: List[str] = []
    cur_start = segments[0]["start"]
    cur_end = cur_start + chunk_duration
    last_end = cur_start

    for seg in segments:
        s = seg.get("start", 0.0)
        e = seg.get("end", s)
        t = _clean_text(seg.get("text", ""))
        if s >= cur_end:
            # flush current chunk
            chunks.append(
                {
                    "start": float(cur_start),
                    "end": float(last_end),
                    "text": " ".join(cur_text_parts).strip(),
                }
            )
            # start new chunk window anchored at this segment
            cur_start = s
            cur_end = cur_start + chunk_duration
            cur_text_parts = [t] if t else []
            last_end = e
        else:
            if t:
                cur_text_parts.append(t)
            last_end = max(last_end, e)

    # flush final
    if cur_text_parts:
        chunks.append(
            {
                "start": float(cur_start),
                "end": float(last_end),
                "text": " ".join(cur_text_parts).strip(),
            }
        )

    msg = "Created %d transcript chunks (chunk_duration=%ds)"
    logger.info(msg, len(chunks), chunk_duration)
    return chunks


def get_transcript_chunks(
    youtube_url: str,
    prefer_subtitles: bool = True,
    whisper_model: str = "small",
) -> Optional[List[Dict]]:
    """Top-level helper.

    Try fetching subtitles; otherwise fall back to downloading audio and
    transcribing with Whisper.

    Returns a list of chunks. Each chunk is a dict with keys 'start',
    'end', and 'text'. Returns ``None`` on fatal errors.
    """
    # try youtube subtitles
    if prefer_subtitles:
        subs = fetch_subtitles(youtube_url)
        if subs:
            # convert to whisper-like segments (start,end,text)
            segments = []
            for s in subs:
                entry = {
                    "start": float(s["start"]),
                    "end": float(s["start"] + s.get("duration", 0.0)),
                    "text": s.get("text", ""),
                }
                segments.append(entry)
            return chunk_segments(segments)

    # fallback to download+whisper
    tmp_audio = None
    try:
        tmp_audio = download_audio(youtube_url)
        if not tmp_audio:
            logger.error("Audio download failed; cannot transcribe")
            return None
        segs = transcribe_with_whisper(tmp_audio, model_name=whisper_model)
        if not segs:
            logger.error("Whisper transcription returned no segments")
            return None
        # Whisper segments already have start,end,text
        return chunk_segments(segs)
    finally:
        # try to remove temporary file
        if tmp_audio and os.path.exists(tmp_audio):
            try:
                os.remove(tmp_audio)
            except Exception:
                logger.debug(
                    "Failed to delete temporary audio: %s",
                    tmp_audio,
                )
