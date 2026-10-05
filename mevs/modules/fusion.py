"""Module 3: Multimodal fusion and summarization."""
from __future__ import annotations

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


def format_pointwise_summary(summary: str, max_points: int = 4) -> str:
    """Convert a plain summary string into a short, coherent bullet list."""
    if not summary or not summary.strip():
        return "- No summary available."

    cleaned = " ".join(summary.strip().split())
    cleaned = re.sub(
        r"\b(?:audio transcript|on-screen text \(ocr\)|slide text)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"\b(?:point|bullet)\s*\d+\s*[:.-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:and|or)\s+(?:the\s+)?(?:main idea|summary|key point)\b", " ", cleaned, flags=re.IGNORECASE)
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
            if not text:
                continue
            low = text.lower()
            if _contains_prompt_echo(text):
                continue
            bullet_lines.append(text)
        if bullet_lines:
            return _unique_bullets(
                [f"- {b}" for b in bullet_lines], max_items=max_points
            )
        return cleaned

    sentences = [
        s.strip().rstrip(".")
        for s in re.split(r"(?<=[.!?])\s+|\s*;\s*", cleaned)
        if s.strip()
    ]

    filtered = []
    seen_sentence_words = []
    for s in sentences:
        if _contains_prompt_echo(s):
            continue
        words = _content_words(s)
        if any(
            len(words) >= 2
            and len(previous) >= 2
            and len(words & previous) / min(len(words), len(previous)) >= 0.9
            for previous in seen_sentence_words
        ):
            continue
        filtered.append(s)
        if words:
            seen_sentence_words.append(words)
    sentences = filtered or sentences
    if not sentences:
        return f"- {cleaned}"

    if max_points > 4 and len(sentences) >= 5:
        return _unique_bullets(
            [f"- {sentence}" for sentence in sentences], max_items=max_points
        )

    if len(sentences) <= 2:
        return _unique_bullets([f"- {s}" for s in sentences])

    bullets = []
    current = []
    for sentence in sentences:
        current.append(sentence)
        joined = " ".join(current)
        if len(joined.split()) >= 18 or len(current) >= 2:
            bullets.append(joined)
            current = []
    if current:
        bullets.append(" ".join(current))

    bullets = [b.strip() for b in bullets if b.strip()]
    if not bullets:
        return f"- {cleaned}"
    return _unique_bullets([f"- {b}" for b in bullets], max_items=max_points)


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
    """Remove repeated bullets and common transcript filler noise."""
    unique = []
    seen = set()
    seen_words = []
    filler_phrases = {"music", "thank you", "thanks for watching", "subscribe", "applause"}

    for bullet in bullets:
        text = bullet[2:].strip()
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
    """Collapse adjacent repeated caption phrases caused by subtitle overlap."""
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


def _deduplicate_across_chunks(results: List[Dict]) -> None:
    """Remove near-identical points repeated in later chunk summaries."""
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
    """Align OCR entries to transcript chunks by timestamp overlap."""
    for c in chunks:
        c["ocr_texts"] = set() # Use set to avoid duplicate OCR reads from adjacent frames

    for o in ocr_results:
        for c in chunks:
            if o["start"] < c["end"] and o["end"] > c["start"]:
                if o.get("text"):
                    c["ocr_texts"].add(o["text"].strip())

    # Convert sets back to lists for downstream compatibility
    for c in chunks:
        c["ocr_texts"] = list(c["ocr_texts"])
    return chunks


def build_summary_prompt(
    transcript: str,
    ocr_texts: List[str],
    point_range: str = "2 to 3",
    previous_summary: str = "",
) -> str:
    """Build clean source text for BART's summarization pipeline."""
    cleaned_transcript = _remove_repeated_phrases(" ".join((transcript or "").split()))
    source_parts = [cleaned_transcript] if cleaned_transcript else []
    cleaned_ocr = list(dict.fromkeys(text.strip() for text in ocr_texts if text.strip()))
    if cleaned_ocr:
        source_parts.append("Slide text (OCR): " + " | ".join(cleaned_ocr))
    return "\n\n".join(source_parts)


def _build_prompt(chunk: Dict) -> str:
    """Build BART source text from a transcript chunk and its OCR text."""
    return build_summary_prompt(
        chunk.get("text", ""), chunk.get("ocr_texts", []) or []
    )


def _generate_summary(
    summarizer, prompt: str, final: bool = False
) -> str:
    """Run BART with output lengths scaled to the source and repetition control."""
    source_words = len(prompt.split())
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
    return (result.get("summary_text") or result.get("generated_text") or "").strip()


def _load_summarizer(model_name: str):
    from transformers import pipeline

    if model_name not in _SUMMARIZER_CACHE:
        task = (
            "text2text-generation"
            if "t5" in model_name.lower()
            else "summarization"
        )
        _SUMMARIZER_CACHE[model_name] = pipeline(
            task,
            model=model_name,
            device=-1,
        )
    return _SUMMARIZER_CACHE[model_name]


def summarize_overview(
    summaries: List[Dict], model_name: str = DEFAULT_SUMMARIZER_MODEL
) -> str:
    """Create one video-level overview from all timestamped summaries."""
    if not summaries:
        return "- No useful key points were found in this video."

    try:
        summarizer = _load_summarizer(model_name)
    except Exception as exc:
        logger.exception("Could not load summarization model '%s'", model_name)
        raise RuntimeError("Summarization model could not be loaded.") from exc

    source = "\n".join(item["summary"] for item in summaries)
    overview = format_pointwise_summary(
        _generate_summary(summarizer, source, final=True), max_points=7
    )
    overview_points = [
        line for line in overview.splitlines() if line.startswith("- ")
    ]
    if len(overview_points) < 5:
        source_points = [
            line.strip()
            for item in summaries
            for line in item["summary"].splitlines()
            if line.strip().startswith("- ")
        ]
        overview = _unique_bullets(
            [*overview_points, *source_points], max_items=7
        )
    return overview


def _reduce_summaries(
    summarizer,
    summaries: List[str],
    final: bool = False,
    point_range: str = "2 to 3",
) -> str:
    """Reduce partial summaries using map-reduce to fit context windows."""
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
    """Summarize transcript chunks and OCR into timestamped bullet points."""
    if not chunks:
        return []
    try:
        summarizer = _load_summarizer(model_name)
    except Exception as exc:
        logger.exception("Could not load summarization model '%s'", model_name)
        raise RuntimeError("Summarization model could not be loaded.") from exc

    results: List[Dict] = []
    for chunk in chunks:
        transcript = (chunk.get("text") or "").strip()
        ocr_texts = chunk.get("ocr_texts", [])

        if not transcript and not ocr_texts:
            continue

        prompt = build_summary_prompt(
            transcript,
            ocr_texts,
        )

        # Split long prompts to stay within the summarizer's input context window.
        if len(prompt.split()) > 400:
            words = transcript.split()
            section_summaries = []
            for start in range(0, len(words), 220):
                sub_transcript = " ".join(words[start:start+220])
                sub_prompt = build_summary_prompt(
                    sub_transcript,
                    ocr_texts,
                )
                section_summaries.append(_generate_summary(summarizer, sub_prompt))
            summary = _reduce_summaries(summarizer, section_summaries, final=True)
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
        results.append({
            "start": chunk.get("start", 0),
            "end": chunk.get("end", 0),
            "summary": bullet_summary,
            "markdown": md,
        })
    _deduplicate_across_chunks(results)
    return results


def assemble_markdown(
    summaries: List[Dict],
    keyframes: List[Dict],
    out_dir: str,
    title: Optional[str] = None,
    overview: Optional[str] = None,
) -> str:
    """Assembles final markdown with embedded keyframes."""
    os.makedirs(out_dir, exist_ok=True)
    md_parts = [f"# {title or 'Video Summary'}\n"]
    if overview:
        md_parts.extend(["## Video Summary\n", overview, ""])

    note_bullets = []
    for s in summaries:
        for line in s["summary"].splitlines():
            candidate = line.strip()
            if not candidate.startswith("- "):
                continue
            text = candidate[2:].strip()
            lowered = text.lower()
            if lowered.startswith(("the video segment is", "this segment is", "no new key points")):
                continue
            note_bullets.append(f"- {text}")

    md_parts.extend(["## Student Notes\n", _unique_bullets(note_bullets), ""])

    attached_keyframes = set()
    for s in summaries:
        for k in keyframes:
            if k["start"] <= s["start"] < k["end"]:
                image_path = os.path.normcase(
                    os.path.abspath(k["image_path"])
                )
                if image_path in attached_keyframes:
                    break
                rel = os.path.relpath(k["image_path"], out_dir)
                md_parts.append(f"![keyframe]({rel})\n")
                attached_keyframes.add(image_path)
                break

    md = "\n".join(md_parts)
    out_path = os.path.join(out_dir, "summary.md")
    with open(out_path, "w", encoding="utf8") as fh:
        fh.write(md)
    return md