from fastapi.testclient import TestClient
from ac_service.main import app

client = TestClient(app)


def test_ac_match_hit():
    resp = client.post("/ac-match", json={"text": "包含 大boss 敏感词"})
    data = resp.json()
    assert resp.status_code == 200
    assert data["matched"] is True
    assert any(hit["word"] == "大boss" for hit in data["hits"])


def test_ac_match_no_hit():
    resp = client.post("/ac-match", json={"text": "safe text"})
    data = resp.json()
    assert resp.status_code == 200
    assert data == {"matched": False, "hit_count": 0, "hits": []}
