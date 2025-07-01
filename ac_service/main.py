"""FastAPI service wrapping SensitiveMatcher."""
from fastapi import FastAPI
from pydantic import BaseModel
import os

from src.ac_core import SensitiveMatcher

DICT_PATH = os.getenv("DICT_PATH", "data/sensitive.txt")
matcher = SensitiveMatcher(DICT_PATH)

app = FastAPI(title="AC Match Service", version="0.1.0")


class Req(BaseModel):
    text: str


@app.post("/ac-match")
def ac_match(req: Req):
    hits = [
        {"word": w, "start": s, "end": e}
        for w, s, e in matcher.find(req.text)
    ]
    return {
        "matched": bool(hits),
        "hit_count": len(hits),
        "hits": hits,
    }
