from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Word:
    text: str
    conf: float
    bbox: List[int]  # [x1,y1,x2,y2]


@dataclass
class Line:
    text: str
    conf: float
    bbox: List[int]
    words: List[Word]


@dataclass
class PageResult:
    page_index: int
    width: int
    height: int
    text: str
    lines: List[Line]


def to_dict(page_results: List[PageResult], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "meta": meta,
        "text": "\n\n".join([p.text for p in page_results]).strip(),
        "pages": [
            {
                "page_index": p.page_index,
                "width": p.width,
                "height": p.height,
                "text": p.text,
                "lines": [
                    {
                        "text": ln.text,
                        "conf": ln.conf,
                        "bbox": ln.bbox,
                        "words": [
                            {"text": w.text, "conf": w.conf, "bbox": w.bbox}
                            for w in ln.words
                        ],
                    }
                    for ln in p.lines
                ],
            }
            for p in page_results
        ],
    }
