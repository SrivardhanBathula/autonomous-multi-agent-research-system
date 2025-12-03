import redis
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

class SessionManager:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.redis.setex(f"session:{session_id}", self.ttl,
            json.dumps({"created": datetime.utcnow().isoformat(), "status": "active"}))
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        data = self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else None

    def update_session(self, session_id: str, data: Dict[str, Any]):
        existing = self.get_session(session_id) or {}
        existing.update(data)
        self.redis.setex(f"session:{session_id}", self.ttl, json.dumps(existing))

    def delete_session(self, session_id: str):
        self.redis.delete(f"session:{session_id}")
