"""Module 3: Multimodal fusion and summarization."""
from __future__ import annotations

import base64
import os
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
_SUMMARIZER_CACHE = {}
DEFAULT_SUMMARIZER_MODEL = os.environ.get(
    "MEVS_SUMMARIZER_MODEL", "facebook/bart-large-cnn"
)


def _contains_prompt_echo(text: str) -> bool:
    """Detect when a model repeats summarization instructions instead of content."""
    if not text:
        return False
    lowered = text.lower()
    instruction_patterns = (
        "write the result in student notes style",
        "highlight the practical takeaway or outcome",
        "end with a short recap",
        "state the main idea",
        "explain the key concept",
        "return only",
        "do not repeat facts",
        "ignore greetings",
        "keep the wording simple",
        "summarize the following video segment",
        "analyze the following video segment",
        "write a clear educational overview",
    )
    return any(pattern in lowered for pattern in instruction_patterns)


def _content_words(text: str) -> set[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "the", "this", "to",
        "using", "with",
    }
    return {
        word
        for word in re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
        if word not in stop_words
    }


def _unique_bullets(bullets: List[str], max_items: int = 4) -> str:
    unique = []
    seen = set()
    seen_words = []
    filler_phrases = {
        "music",
        "thank you",
        "thanks for watching",
        "subscribe",
        "applause",
    }

    for bullet in bullets:
        text = bullet[2:].strip() if bullet.startswith("- ") else bullet.strip()
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not normalized or normalized in seen:
            continue
        if normalized in filler_phrases:
            continue
        if "thank you" in normalized and "watch" in normalized:
            continue

        words = _content_words(text)
        if any(
            len(words) >= 6
            and len(previous) >= 6
            and len(words & previous) / len(words | previous) >= 0.9
            for previous in seen_words
        ):
            continue

        seen.add(normalized)
        seen_words.append(words)
        unique.append(f"- {text.rstrip('.')}")

    return "\n".join(unique[:max_items]) or "- No useful key points were found in this segment."


def _remove_repeated_phrases(text: str) -> str:
    words = text.split()
    changed = True
    while changed:
        changed = False
        for size in range(min(20, len(words) // 2), 3, -1):
            left = [word.lower().strip(".,!?;:") for word in words[-2 * size:-size]]
            right = [word.lower().strip(".,!?;:") for word in words[-size:]]
            if left == right:
                words = words[:-size]
                changed = True
                break
    return " ".join(words)


def format_pointwise_summary(summary: str, max_points: int = 4) -> str:
    if not summary or not summary.strip():
        return "- No summary available."

    cleaned = " ".join(summary.strip().split())
    cleaned = re.sub(
        r"\b(?:audio transcript|on-screen text \(ocr\)|slide text)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"\b(?:point|bullet)\s*\d+\s*[:.-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:and|or)\s+(?:the\s+)?(?:main idea|summary|key point)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    if cleaned.startswith("- "):
        bullet_lines = []
        for line in summary.strip().splitlines():
            candidate = line.strip()
            if not candidate.startswith("- "):
                continue
            text = re.sub(
                r"\b(?:audio transcript|on-screen text \(ocr\)|slide text)\s*:\s*",
                "",
                candidate[2:].strip(),
                flags=re.IGNORECASE,
            ).strip()
            if text and not _contains_prompt_echo(text):
                bullet_lines.append(f"- {text}")
        if bullet_lines:
            return _unique_bullets(bullet_lines, max_items=max_points)

    sentences = [
        s.strip().rstrip(".")
        for s in re.split(r"(?<=[.!?])\s+|\s*;\s*", cleaned)
        if s.strip()
    ]

    filtered = []
    seen_sentence_words = []
    for sentence in sentences:
        if _contains_prompt_echo(sentence):
            continue
        words = _content_words(sentence)
        if any(
            len(words) >= 2
            and len(previous) >= 2
            and len(words & previous) / min(len(words), len(previous)) >= 0.9
            for previous in seen_sentence_words
        ):
            continue
        filtered.append(sentence)
        if words:
            seen_sentence_words.append(words)

    sentences = filtered or sentences
    if not sentences:
        return f"- {cleaned}"

    if len(sentences) <= 2:
        return _unique_bullets([f"- {s}" for s in sentences], max_items=max_points)

    bullets = []
    current = []
    for sentence in sentences:
        current.append(sentence)
        joined = " ".join(current)
        if len(joined.split()) >= 18 or len(current) >= 2:
            bullets.append(f"- {joined}")
            current = []
    if current:
        bullets.append(f"- {' '.join(current)}")

    return _unique_bullets(bullets, max_items=max_points)


def _deduplicate_across_chunks(results: List[Dict]) -> None:
    seen_words = []
    for result in results:
        bullets = result["summary"].splitlines()
        kept = []
        for bullet in bullets:
            words = _content_words(bullet)
            if not words:
                continue
            if any(
                len(words) >= 6
                and len(previous) >= 6
                and len(words & previous) / len(words | previous) >= 0.9
                for previous in seen_words
            ):
                continue
            kept.append(bullet)
            seen_words.append(words)
        result["summary"] = "\n".join(kept) or "- No new key points in this segment."
        result["markdown"] = (
            f"### {result.get('start', 0):.1f}s - "
            f"{result.get('end', 0):.1f}s\n\n{result['summary']}\n"
        )


def align_ocr_with_chunks(
    ocr_results: List[Dict], chunks: List[Dict]
) -> List[Dict]:
    """Attach OCR text to transcript chunks by timestamp overlap."""
    for chunk in chunks:
        chunk["ocr_texts"] = []
        chunk["visuals"] = []

    for ocr in ocr_results:
        o_start = float(ocr.get("start", ocr.get("timestamp", 0.0)))
        o_end = float(ocr.get("end", o_start + 1.0))
        for chunk in chunks:
            c_start = float(chunk.get("start", 0.0))
            c_end = float(chunk.get("end", c_start))
            if o_start < c_end and o_end > c_start:
                text = str(ocr.get("text", "")).strip()
                if text and text not in chunk["ocr_texts"]:
                    chunk["ocr_texts"].append(text)
                if ocr.get("image_bytes"):
                    chunk["visuals"].append(ocr)

    return chunks


def build_summary_prompt(
    transcript: str,
    ocr_texts: List[str],
    point_range: str = "2 to 3",
    previous_summary: str = "",
) -> str:
    """Build source text for BART from transcript + OCR."""
    del point_range, previous_summary
    cleaned_transcript = _remove_repeated_phrases(
        " ".join((transcript or "").split())
    )
    source_parts = [cleaned_transcript] if cleaned_transcript else []

    cleaned_ocr = list(
        dict.fromkeys(text.strip() for text in ocr_texts if text.strip())
    )
    if cleaned_ocr:
        source_parts.append("Slide text (OCR): " + " | ".join(cleaned_ocr))

    return "\n\n".join(source_parts)


def _generate_summary(summarizer, prompt: str, final: bool = False) -> str:
    source_words = max(1, len(prompt.split()))
    max_length = min(142, max(8, round(source_words * (0.55 if final else 0.7))))
    min_length = min(max_length - 1, max(4, round(source_words * 0.12)))

    output = summarizer(
        prompt,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        truncation=True,
    )
    result = output[0]
    return (
        result.get("summary_text") or result.get("generated_text") or ""
    ).strip()


def _load_summarizer(model_name: str):
    from transformers import pipeline

    if model_name not in _SUMMARIZER_CACHE:
        task = (
            "text2text-generation"
            if "t5" in model_name.lower()
            else "summarization"
        )
        device = int(os.environ.get("MEVS_BART_DEVICE", "-1"))
        _SUMMARIZER_CACHE[model_name] = pipeline(
            task,
            model=model_name,
            device=device,
        )
    return _SUMMARIZER_CACHE[model_name]


def summarize_overview(
    summaries: List[Dict], model_name: str = DEFAULT_SUMMARIZER_MODEL
) -> str:
    if not summaries:
        return "- No useful key points were found in this video."

    summarizer = _load_summarizer(model_name)
    source = "\n".join(item["summary"] for item in summaries if item.get("summary"))
    overview = format_pointwise_summary(
        _generate_summary(summarizer, source, final=True), max_points=7
    )

    points = [line for line in overview.splitlines() if line.startswith("- ")]
    if len(points) < 5:
        source_points = [
            line.strip()
            for item in summaries
            for line in item["summary"].splitlines()
            if line.strip().startswith("- ")
        ]
        overview = _unique_bullets(
            [*points, *source_points], max_items=7
        )
    return overview


def _reduce_summaries(summarizer, summaries: List[str], final: bool = False) -> str:
    current = [summary for summary in summaries if summary.strip()]
    while len(" ".join(current)) > 1500 and len(current) > 1:
        reduced = []
        for index in range(0, len(current), 3):
            batch = "\n".join(current[index:index + 3])
            reduced.append(_generate_summary(summarizer, batch))
        current = reduced
    return _generate_summary(summarizer, "\n".join(current), final=final)


def summarize_chunks(
    chunks: List[Dict], model_name: str = DEFAULT_SUMMARIZER_MODEL
) -> List[Dict]:
    """Summarize transcript chunks plus OCR text using BART."""
    if not chunks:
        return []

    summarizer = _load_summarizer(model_name)
    results: List[Dict] = []

    for chunk in chunks:
        transcript = (chunk.get("text") or "").strip()
        ocr_texts = chunk.get("ocr_texts", [])
        if not transcript and not ocr_texts:
            continue

        prompt = _build_prompt(chunk)

        if len(prompt.split()) > 400:
            words = transcript.split()
            section_summaries = []
            for start in range(0, len(words), 220):
                sub_transcript = " ".join(words[start:start + 220])
                sub_prompt = _build_prompt({"text": sub_transcript, "ocr_texts": ocr_texts})
                section_summaries.append(_generate_summary(summarizer, sub_prompt))
            summary = _reduce_summaries(
                summarizer, section_summaries, final=True
            )
        else:
            summary = _generate_summary(summarizer, prompt, final=True)

        if _contains_prompt_echo(summary):
            fallback_text = transcript or " ".join(ocr_texts or [])
            summary = fallback_text if fallback_text else summary

        bullet_summary = format_pointwise_summary(summary)
        md = (
            f"### {chunk.get('start', 0):.1f}s - "
            f"{chunk.get('end', 0):.1f}s\n\n{bullet_summary}\n"
        )
        results.append(
            {
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
                "summary": bullet_summary,
                "markdown": md,
            }
        )

    _deduplicate_across_chunks(results)
    return results


def _build_prompt(
    chunk: Dict,
    *,
    include_ocr: bool = True,
) -> str:
    """Build source text for a BART summary from transcript and OCR."""
    transcript = (chunk.get("text") or "").strip()
    ocr_texts = chunk.get("ocr_texts", []) or []

    cleaned_transcript = _remove_repeated_phrases(" ".join(transcript.split()))
    source_parts = [cleaned_transcript] if cleaned_transcript else []

    if include_ocr:
        cleaned_ocr = list(dict.fromkeys(text.strip() for text in ocr_texts if text.strip()))
        if cleaned_ocr:
            source_parts.append("Slide text (OCR): " + " | ".join(cleaned_ocr))

    return "\n\n".join(source_parts)


def _image_data_uri(image_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")


def _nearest_visual_to_summary(summary: Dict, keyframes: List[Dict]) -> Optional[Dict]:
    if not keyframes:
        return None

    start = float(summary.get("start", 0.0))
    end = float(summary.get("end", start))
    center = (start + end) / 2.0

    overlapping = []
    for frame in keyframes:
        t = float(frame.get("timestamp", frame.get("start", 0.0)))
        if start <= t <= end:
            overlapping.append(frame)

    pool = overlapping or list(keyframes)
    return min(
        pool,
        key=lambda frame: abs(
            float(frame.get("timestamp", frame.get("start", 0.0))) - center
        ),
    )


def assemble_markdown(
    summaries: List[Dict],
    keyframes: List[Dict],
    out_dir: Optional[str] = None,
    title: Optional[str] = None,
    overview: Optional[str] = None,
    *,
    save_markdown: bool = False,
) -> str:
    """Build Markdown with unique local keyframe references."""
    out_dir = out_dir or os.getcwd()

    parts = [f"# {title or 'Video Summary'}", ""]
    if overview:
        parts.extend(["## Video Summary", "", overview, ""])

    note_bullets = []
    for summary in summaries:
        for line in summary.get("summary", "").splitlines():
            candidate = line.strip()
            if not candidate.startswith("- "):
                continue
            text = candidate[2:].strip()
            lowered = text.lower()
            if lowered.startswith(
                ("the video segment is", "this segment is", "no new key points")
            ):
                continue
            note_bullets.append(f"- {text}")

    parts.extend(["## Student Notes", "", _unique_bullets(note_bullets), ""])

    attached_times = set()
    attached_paths = set()
    for summary in summaries:
        visual = _nearest_visual_to_summary(summary, keyframes)
        if not visual:
            continue

        timestamp = float(visual.get("timestamp", visual.get("start", 0.0)))
        time_key = round(timestamp, 1)
        if time_key in attached_times:
            continue

        image_path = visual.get("image_path") or visual.get("path")
        if image_path:
            rel_path = os.path.relpath(image_path, out_dir).replace("\\", "/")
            if rel_path in attached_paths:
                continue
            attached_paths.add(rel_path)
            time_text = _format_timestamp(timestamp)
            alt = f"Important visual at {time_text}"
            parts.extend([
                f"### Important Visual — {time_text}",
                "",
                f"![{alt}]({rel_path})",
                "",
            ])
            attached_times.add(time_key)
            continue

        image_bytes = visual.get("image_bytes")
        if not image_bytes:
            continue

        rel_path = _image_data_uri(image_bytes)
        time_text = _format_timestamp(timestamp)
        alt = f"Important visual at {time_text}"
        parts.extend([
            f"### Important Visual — {time_text}",
            "",
            f"![{alt}]({rel_path})",
            "",
        ])
        attached_times.add(time_key)

    markdown = "\n".join(parts).strip() + "\n"

    if save_markdown:
        target_dir = os.path.abspath(
            os.environ.get("MEVS_MARKDOWN_DIR", out_dir)
        )
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, "summary.md")
        saved_markdown = re.sub(
            r"(?ms)^### Important Visual — [^\n]+\n+!\[[^\]]*\]"
            r"\(data:image/(?:jpeg|png);base64,[A-Za-z0-9+/=]+\)\n*",
            "",
            markdown,
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(saved_markdown)
        logger.info("Saved Markdown summary to %s", path)

    return markdown


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
