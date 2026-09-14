"""Module 3: Multimodal fusion and summarization."""
from __future__ import annotations

import os
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
_SUMMARIZER_CACHE = {}


def format_pointwise_summary(summary: str) -> str:
    """Convert a plain summary string into a short, coherent bullet list."""
    if not summary or not summary.strip():
        return "- No summary available."

    cleaned = " ".join(summary.strip().split())
    if cleaned.startswith("- "):
        bullet_lines = [
            line.strip()
            for line in summary.strip().splitlines()
            if line.strip().startswith("- ")
        ]
        if bullet_lines:
            return _unique_bullets(bullet_lines)
        return cleaned

    sentences = [
        s.strip().rstrip(".")
        for s in re.split(r"(?<=[.!?])\s+", cleaned)
        if s.strip()
    ]
    if not sentences:
        return f"- {cleaned}"

    if len(sentences) <= 2:
        return _unique_bullets([f"- {s}" for s in sentences])

    bullets = []
    current = []
    for sentence in sentences:
        current.append(sentence)
        if len(" ".join(current).split()) >= 18 or len(current) >= 2:
            bullets.append(" ".join(current))
            current = []
    if current:
        bullets.append(" ".join(current))

    bullets = [b.strip() for b in bullets if b.strip()]
    if not bullets:
        return f"- {cleaned}"
    return _unique_bullets([f"- {b}" for b in bullets])


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


def _unique_bullets(bullets: List[str]) -> str:
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
            words and previous and len(words & previous) / len(words | previous) >= 0.75
            for previous in seen_words
        ):
            continue

        seen.add(normalized)
        seen_words.append(words)
        unique.append(f"- {text.rstrip('.')}")

    return "\n".join(unique[:4]) or "- No useful key points were found in this segment."


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
                len(words & previous) / len(words | previous) >= 0.75
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
    """Create a universal, multimodal instruction for video summarization."""
    cleaned_transcript = _remove_repeated_phrases(" ".join((transcript or "").split()))
    ocr_joined = " | ".join(ocr_texts) if ocr_texts else "None"

    context = previous_summary.strip() or "None"
    return (
        f"Analyze the following video segment using its audio transcript and on-screen text (OCR). "
        f"Provide a concise summary in {point_range} bullet points.\n\n"
        "Guidelines:\n"
        "1. Capture the core narrative, main concepts, and actionable steps.\n"
        "2. Integrate context from the on-screen text (OCR) if it clarifies the spoken transcript.\n"
        "3. Ignore greetings, calls to subscribe, sponsorships, and repeated framing.\n"
        "4. Do not repeat facts already covered in the previous chunk summary.\n"
        "5. Return ONLY 2 or 3 short bullet points, without conversational text.\n\n"
        f"Previous chunk summary (avoid repeating these points):\n{context}\n\n"
        f"Audio Transcript:\n{cleaned_transcript}\n\n"
        f"On-Screen Text (OCR):\n{ocr_joined}"
    )


def _generate_summary(
    summarizer, prompt: str, final: bool = False
) -> str:
    """Run the summarization model on a bounded input."""
    source_words = len(prompt.split())
    # Increased token limits for better logical coherence
    max_length = max(96, min(180 if final else 128, source_words))
    min_length = min(32 if final else 24, max(12, max_length // 5))

    output = summarizer(
        prompt,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )
    result = output[0]
    return (result.get("summary_text") or result.get("generated_text") or "").strip()


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
            # Use a simpler reduction prompt for intermediate steps
            reduce_prompt = (
                "Combine these key points into 2 or 3 short, non-repeating bullets. "
                "Keep only useful facts and actions:\n" + batch
            )
            reduced.append(_generate_summary(summarizer, reduce_prompt))
        current = reduced

    final_prompt = (
        f"Synthesize these video segment summaries into {point_range} short, "
        "non-repeating final bullet points:\n" + "\n".join(current)
    )
    return _generate_summary(summarizer, final_prompt, final=final)


def summarize_chunks(
    chunks: List[Dict], model_name: str = "google/flan-t5-large"
) -> List[Dict]:
    """Summarize transcript chunks and OCR into timestamped bullet points."""
    if not chunks:
        return []
    try:
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
                device=-1, # Set to 0 if running on GPU
            )
        summarizer = _SUMMARIZER_CACHE[model_name]
    except Exception as exc:
        logger.exception("Could not load summarization model '%s'", model_name)
        raise RuntimeError("Summarization model could not be loaded.") from exc

    results: List[Dict] = []
    previous_summaries = []
    for chunk in chunks:
        transcript = (chunk.get("text") or "").strip()
        ocr_texts = chunk.get("ocr_texts", [])

        if not transcript and not ocr_texts:
            continue

        # Create the multimodal prompt
        prompt = build_summary_prompt(
            transcript,
            ocr_texts,
            point_range="2 to 3",
            previous_summary="\n".join(previous_summaries)[-1800:],
        )

        # If the prompt is too large for T5 (usually 512 tokens), we split the raw text
        # For a truly robust pipeline, consider swapping T5 for a model with a larger context window (e.g., Llama-3, Mistral)
        if len(prompt.split()) > 400:
            words = transcript.split()
            section_summaries = []
            for start in range(0, len(words), 300):
                sub_transcript = " ".join(words[start:start+300])
                sub_prompt = build_summary_prompt(
                    sub_transcript,
                    ocr_texts,
                    previous_summary="\n".join(previous_summaries)[-1800:],
                )
                section_summaries.append(_generate_summary(summarizer, sub_prompt))
            summary = _reduce_summaries(summarizer, section_summaries, final=True)
        else:
            summary = _generate_summary(summarizer, prompt, final=True)

        bullet_summary = format_pointwise_summary(summary)
        previous_summaries.append(bullet_summary)
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
) -> str:
    """Assembles final markdown with embedded keyframes."""
    os.makedirs(out_dir, exist_ok=True)
    md_parts = [f"# {title or 'Video Summary'}\n"]

    for s in summaries:
        md_parts.append(s["markdown"])

        # Attach closest keyframe
        for k in keyframes:
            if k["start"] <= s["start"] <= k["end"] or (s["start"] >= k["start"] and s["start"] <= k["end"]):
                rel = os.path.relpath(k["image_path"], out_dir)
                md_parts.append(f"![keyframe]({rel})\n")
                break

    md = "\n".join(md_parts)
    out_path = os.path.join(out_dir, "summary.md")
    with open(out_path, "w", encoding="utf8") as fh:
        fh.write(md)
    return md