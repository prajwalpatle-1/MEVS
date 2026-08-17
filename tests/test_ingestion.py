from mevs.modules.ingestion import _extract_video_id


def test_extract_video_id():
    assert (
        _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("") is None
