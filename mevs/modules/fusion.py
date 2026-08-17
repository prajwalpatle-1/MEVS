"""Module 3: Multimodal fusion and summarization."""
from __future__ import annotations

import os
import logging
from typing import List, Dict, Optional

from transformers import pipeline

logger = logging.getLogger(__name__)


def align_ocr_with_chunks(
    ocr_results: List[Dict], chunks: List[Dict]
) -> List[Dict]:
    """Align OCR entries to transcript chunks by timestamp overlap.

    Returns list of chunk dicts with added 'ocr_texts': List[str]
    """
    for c in chunks:
        c["ocr_texts"] = []
    for o in ocr_results:
        for c in chunks:
            # overlap if o.start < c.end and o.end > c.start
            if o["start"] < c["end"] and o["end"] > c["start"]:
                if o.get("text"):
                    c["ocr_texts"].append(o["text"])
    return chunks


def _build_prompt(chunk: Dict) -> str:
    header = f"Timestamp: {chunk['start']:.1f}s - {chunk['end']:.1f}s\n"
    transcript = chunk.get("text", "")
    ocr_text = "\n".join(chunk.get("ocr_texts", []))
    prompt_parts = [
        "You are an expert educator. Given the transcript excerpt and",
        "extracted slide text, produce Structured Educational Notes with:",
        "1) Key Concepts, 2) Definitions, 3) Bullet Points,",
        "4) Example or short explanation. Output as Markdown.",
        "\n\n",
        header,
        "\nTranscript:\n",
        transcript,
        "\n\nSlide Text:\n",
        ocr_text,
        "\n\nNotes:\n",
    ]
    return "".join(prompt_parts)


def summarize_chunks(
    chunks: List[Dict], model_name: str = "sshleifer/distilbart-cnn-12-6"
) -> List[Dict]:
    """Summarize each chunk using a local summarization model.

    Returns list of {'start','end','summary','markdown'}.
    """
    summarizer = pipeline(
        "summarization",
        model=model_name,
    )
    results: List[Dict] = []
    for c in chunks:
        prompt = _build_prompt(c)
        try:
            out = summarizer(
                prompt, max_length=200, min_length=50, do_sample=False
            )
            summary = out[0]["summary_text"]
        except Exception:
            logger.exception(
                "Summarization failed for chunk %s-%s", c["start"], c["end"]
            )
            summary = ""
        md = f"### {c['start']:.1f}s - {c['end']:.1f}s\n\n{summary}\n"
        # embed image links if available
        if c.get("ocr_texts"):
            slide_lines = [f"- Slide text: {t}" for t in c["ocr_texts"]]
            md += "\n".join(slide_lines)
            md += "\n"
        results.append(
            {
                "start": c["start"],
                "end": c["end"],
                "summary": summary,
                "markdown": md,
            }
        )
    return results


def assemble_markdown(
    summaries: List[Dict],
    keyframes: List[Dict],
    out_dir: str,
    title: Optional[str] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    md_parts = [f"# {title or 'Video Summary'}\n"]
    for s in summaries:
        md_parts.append(s["markdown"])
        # attach closest keyframe if any
        for k in keyframes:
            cond1 = k["start"] <= s["start"] <= k["end"]
            cond2 = s["start"] >= k["start"] and s["start"] <= k["end"]
            if cond1 or cond2:
                rel = os.path.relpath(k["image_path"], out_dir)
                img_md = f"![keyframe]({rel})\n"
                md_parts.append(img_md)
                break
    md = "\n".join(md_parts)
    out_path = os.path.join(out_dir, "summary.md")
    with open(out_path, "w", encoding="utf8") as fh:
        fh.write(md)
    return md
