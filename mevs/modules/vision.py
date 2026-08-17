"""Module 2: Visual processing.

Provides helpers to download videos, detect scene boundaries, extract
representative keyframes, deduplicate similar frames, and run OCR on them.
"""
from __future__ import annotations

import os
import tempfile
import logging
from typing import List, Dict, Optional, Tuple

try:
    import yt_dlp
except Exception:
    yt_dlp = None

import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash

logger = logging.getLogger(__name__)

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
except Exception:
    VideoManager = None
    SceneManager = None
    ContentDetector = None

try:
    import easyocr
except Exception:
    easyocr = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def download_video(
    youtube_url: str, out_path: Optional[str] = None
) -> Optional[str]:
    if yt_dlp is None:
        logger.error("yt_dlp not installed")
        return None
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
    opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([youtube_url])
        return out_path
    except Exception:
        logger.exception("Failed to download video")
        return None


def detect_scenes(
    video_path: str, threshold: float = 30.0
) -> List[Tuple[float, float]]:
    """Return list of (start_sec, end_sec) scenes detected by
    PySceneDetect ContentDetector.
    """
    if VideoManager is None:
        logger.warning(
            "scenedetect not available — returning whole video as one scene"
        )
        # best-effort fallback: return the entire video length as one scene
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        duration = frame_count / max(1.0, fps)
        return [(0.0, float(duration))]
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        scenes: List[Tuple[float, float]] = []
        for start, end in scene_list:
            scenes.append((start.get_seconds(), end.get_seconds()))
        return scenes
    except Exception:
        logger.exception("Scene detection failed")
        # fallback: return whole-video range
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        duration = frame_count / max(1.0, fps)
        return [(0.0, float(duration))]
    finally:
        try:
            video_manager.release()
        except Exception:
            pass


def extract_keyframes(
    video_path: str,
    scenes: List[Tuple[float, float]],
    out_dir: Optional[str] = None,
) -> List[Dict]:
    """Extract a representative frame per scene.

    The function saves a frame (typically the middle frame of the scene)
    and returns a list of dicts with short metadata.

    Each dict contains keys 'image_path', 'start', and 'end'.
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(video_path), "keyframes")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    results: List[Dict] = []
    for idx, (s, e) in enumerate(scenes):
        mid = (s + e) / 2.0
        frame_no = int(mid * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            logger.debug("Failed to read frame at %s seconds", mid)
            continue
        out_path = os.path.join(out_dir, f"kf_{idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        results.append(
            {
                "image_path": out_path,
                "start": s,
                "end": e,
            }
        )
    cap.release()
    return results


def _are_similar(
    path_a: str, path_b: str, hash_thresh: int = 5, ssim_thresh: float = 0.9
) -> bool:
    try:
        ha = imagehash.phash(Image.open(path_a))
        hb = imagehash.phash(Image.open(path_b))
        if abs(ha - hb) <= hash_thresh:
            return True
        # fallback to SSIM
        ia = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2GRAY)
        ib = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2GRAY)
        h = min(ia.shape[0], ib.shape[0])
        w = min(ia.shape[1], ib.shape[1])
        iar = cv2.resize(ia, (w, h))
        ibr = cv2.resize(ib, (w, h))
        score = ssim(iar, ibr)
        return score >= ssim_thresh
    except Exception:
        return False


def filter_similar_frames(
    frames: List[Dict], hash_thresh: int = 5, ssim_thresh: float = 0.9
) -> List[Dict]:
    kept: List[Dict] = []
    for f in frames:
        path = f["image_path"]
        redundant = False
        for k in kept:
            similar = _are_similar(
                path,
                k["image_path"],
                hash_thresh=hash_thresh,
                ssim_thresh=ssim_thresh,
            )
            if similar:
                redundant = True
                break
        if not redundant:
            kept.append(f)
    logger.info(
        "Filtered %d -> %d frames after deduplication",
        len(frames),
        len(kept),
    )
    return kept


def run_ocr_on_frames(
    frames: List[Dict], lang_list: Optional[List[str]] = None
) -> List[Dict]:
    """Extract text from frames using EasyOCR or pytesseract fallback.

    Returns list of {'image_path','start','end','text'}.
    """
    results: List[Dict] = []
    reader = None
    if easyocr is not None:
        try:
            reader = easyocr.Reader(lang_list or ["en"], gpu=False)
        except Exception:
            reader = None
    for f in frames:
        imgp = f["image_path"]
        text = ""
        try:
            if reader is not None:
                res = reader.readtext(imgp)
                text = " ".join([r[1] for r in res if r[1].strip()])
            elif pytesseract is not None:
                img = Image.open(imgp)
                text = pytesseract.image_to_string(img)
        except Exception:
            logger.exception("OCR failed on %s", imgp)
        results.append(
            {
                "image_path": imgp,
                "start": f["start"],
                "end": f["end"],
                "text": text.strip(),
            }
        )
    return results
