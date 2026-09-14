import sys
import types

from mevs.api import app
from mevs.modules import fusion, ingestion, vision
from mevs.modules.ingestion import _extract_video_id


def test_format_pointwise_summary_uses_bullets():
    result = fusion.format_pointwise_summary(
        "AI extracts text from the video. It also identifies key slides. "
        "Then it creates a concise report."
    )
    assert "- " in result
    bullet_lines = [line for line in result.splitlines() if line.strip().startswith("- ")]
    assert len(bullet_lines) >= 2


def test_format_pointwise_summary_removes_near_duplicate_points():
    result = fusion.format_pointwise_summary(
        "Python is open source and free to use. "
        "Python is open source and free to use."
    )

    assert result.count("Python is open source") == 1


def test_summarize_chunks_uses_multimodal_prompt(monkeypatch):
    prompts = []

    class DummySummarizer:
        def __call__(self, prompt, **kwargs):
            assert "Slide text:" not in prompt
            assert "Analyze the following video segment" in prompt
            assert "On-Screen Text (OCR):" in prompt
            prompts.append(prompt)
            return [{"summary_text": "Point one. Point two."}]

    dummy = types.SimpleNamespace(pipeline=lambda *args, **kwargs: DummySummarizer())
    monkeypatch.setitem(sys.modules, "transformers", dummy)
    monkeypatch.setattr(fusion, "_SUMMARIZER_CACHE", {})

    chunks = [
        {"start": 0, "end": 10, "text": "chunk 1", "ocr_texts": ["slide title"]},
        {"start": 10, "end": 20, "text": "chunk 2", "ocr_texts": ["slide body"]},
    ]

    results = fusion.summarize_chunks(chunks)

    assert len(results) == 2
    assert any("chunk 1" in prompt for prompt in prompts)
    assert any("chunk 2" in prompt for prompt in prompts)
    assert any("slide title" in prompt for prompt in prompts)
    assert any("slide body" in prompt for prompt in prompts)
    assert all("Slide text:" not in result["summary"] for result in results)
    assert all("- " in result["summary"] for result in results)
    assert results[0]["start"] == 0
    assert results[0]["end"] == 10
    assert results[1]["start"] == 10
    assert results[1]["end"] == 20


def test_transcript_only_mode_does_not_download_audio(monkeypatch):
    monkeypatch.setattr(ingestion, "fetch_subtitles", lambda url: None)
    monkeypatch.setattr(
        ingestion,
        "download_audio",
        lambda url: (_ for _ in ()).throw(AssertionError("audio downloaded")),
    )

    assert ingestion.get_transcript_chunks(
        "https://youtu.be/dQw4w9WgXcQ",
        allow_whisper_fallback=False,
    ) is None


def test_extract_video_id():
    assert (
        _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("") is None


def test_sample_time_ranges_avoids_scene_detection(monkeypatch):
    class DummyCapture:
        def get(self, property_id):
            if property_id == vision.cv2.CAP_PROP_FPS:
                return 10
            return 1000

        def release(self):
            pass

    monkeypatch.setattr(vision.cv2, "VideoCapture", lambda path: DummyCapture())

    assert vision.sample_time_ranges("video.mp4", interval_seconds=30) == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 90.0),
        (90.0, 100.0),
    ]


def test_summarize_falls_back_to_transcript_when_video_download_fails(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "get_transcript_chunks",
        lambda *args, **kwargs: [{"start": 0, "end": 30, "text": "Lecture intro. Main concept is linear algebra."}],
    )
    monkeypatch.setattr(vision, "download_video", lambda url: None)
    monkeypatch.setattr(vision, "detect_scenes", lambda video_path: [])
    monkeypatch.setattr(vision, "extract_keyframes", lambda *args, **kwargs: [])
    monkeypatch.setattr(vision, "filter_similar_frames", lambda frames: [])
    monkeypatch.setattr(vision, "run_ocr_on_frames", lambda frames: [])

    import asyncio

    result = asyncio.run(
        app.summarize(
            app.SummarizeRequest(url="https://youtube.com/watch?v=dQw4w9WgXcQ")
        )
    )

    assert "linear algebra" in result["markdown"].lower()
    assert "mode" in result
