from mevs.modules import ingestion
from mevs.modules.ingestion import _extract_video_id


def test_extract_video_id():
    assert (
        _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("") is None


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
