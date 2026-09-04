"""Module 3: Multimodal fusion and summarization."""
from __future__ import annotations

import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
_SUMMARIZER_CACHE = {}


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
    transcript = chunk.get("text", "")
    ocr_text = "\n".join(chunk.get("ocr_texts", []))
    if ocr_text:
        return f"{transcript}\nSlide text: {ocr_text}"
    return transcript


def summarize_chunks(
    chunks: List[Dict], model_name: str = "sshleifer/distilbart-cnn-12-6"
) -> List[Dict]:
    """Summarize each chunk using a local summarization model.

    Returns list of {'start','end','summary','markdown'}.
    """
    """Generate concise summaries from transcript and OCR text.

    The model is loaded once per process and each input is capped so long
    videos do not exceed the model tokenizer or CPU memory budget.
    """
    if not chunks:
        return []
    try:
        from transformers import pipeline

        if model_name not in _SUMMARIZER_CACHE:
            _SUMMARIZER_CACHE[model_name] = pipeline(
                "summarization",
                model=model_name,
                device=-1,
            )
        summarizer = _SUMMARIZER_CACHE[model_name]
    except Exception as exc:
        logger.exception("Could not load summarization model '%s'", model_name)
        raise RuntimeError(
            "Summarization model could not be loaded. Install a compatible "
            "Transformers 4.x release and ensure the model is downloaded."
        ) from exc

    results: List[Dict] = []
    for c in chunks:
        # DistilBART accepts up to 1024 tokens. Keep the source smaller for
        # predictable CPU time and leave room for OCR text.
        prompt = _build_prompt(c)[:6000]
        if not prompt.strip():
            summary = "No usable transcript or slide text was found."
            out = []
        else:
            out = None
        source_words = len(prompt.split())
        max_length = max(24, min(160, source_words))
        min_length = min(30, max(8, max_length // 4))
        try:
            if out is None:
                out = summarizer(
                    prompt,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                )
                summary = out[0]["summary_text"].strip()
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
