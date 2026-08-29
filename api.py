import os
import json
import logging
import sqlite3
import hashlib
import secrets
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

# -----------------------------------------------------------------------------
# 1. Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recallspection-api")

# -----------------------------------------------------------------------------
# 2. Environment variables
# -----------------------------------------------------------------------------
MEMORY_FILE = os.getenv("RECALLSPECTION_MEMORY_FILE", "memory.json")
DB_FILE = os.getenv("RECALLSPECTION_DB_FILE", "keys.db")
SWSTM_MODE = os.getenv("SWSTM_MODE", "flat")

# -----------------------------------------------------------------------------
# 3. Database: multiple API keys with usage tracking
# -----------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            plan TEXT NOT NULL,
            usage INTEGER DEFAULT 0,
            limit INTEGER DEFAULT 1000,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_used TEXT,
            is_active INTEGER DEFAULT 1
        )
    ''')
    # Add an index for faster lookups
    conn.execute('CREATE INDEX IF NOT EXISTS idx_key_id ON api_keys(key_id)')
    conn.commit()
    conn.close()

def create_api_key(owner: str, plan: str = "free") -> str:
    """Generate a new API key and store it in DB."""
    key = f"rk_{secrets.token_urlsafe(24)}"  # rk = recallspection key
    limit_map = {
        "free": 1000,
        "pro": 100000,
        "enterprise": 1000000,
        "agent_free": 5000,
        "agent_pro": 500000,
        "agent_enterprise": 5000000,
    }
    limit = limit_map.get(plan, 1000)
    conn = get_db()
    conn.execute(
        "INSERT INTO api_keys (key_id, owner, plan, limit) VALUES (?, ?, ?, ?)",
        (key, owner, plan, limit)
    )
    conn.commit()
    conn.close()
    return key

def get_key_info(key: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM api_keys WHERE key_id = ? AND is_active = 1",
        (key,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def increment_usage(key: str) -> int:
    conn = get_db()
    conn.execute(
        "UPDATE api_keys SET usage = usage + 1, last_used = CURRENT_TIMESTAMP WHERE key_id = ?",
        (key,)
    )
    conn.commit()
    # Get updated usage
    row = conn.execute("SELECT usage, limit FROM api_keys WHERE key_id = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else 0

def get_remaining_usage(key: str) -> int:
    conn = get_db()
    row = conn.execute("SELECT limit, usage FROM api_keys WHERE key_id = ?", (key,)).fetchone()
    conn.close()
    if row:
        return row[0] - row[1]
    return 0

# -----------------------------------------------------------------------------
# 4. Lazy imports (SWSTM/ExactMemory)
# -----------------------------------------------------------------------------
swstm = None
exact = None

def get_exact():
    global exact
    if exact is None:
        from recallspection.exact import ExactMemory
        exact = ExactMemory()
        logger.info("ExactMemory initialized")
    return exact

def get_swstm():
    global swstm
    if swstm is None:
        from recallspection.swstm import SWSTMEngine
        swstm = SWSTMEngine(mode=SWSTM_MODE, flat_num_slots=200)
        logger.info(f"SWSTMEngine initialized (mode={SWSTM_MODE})")
    return swstm

# -----------------------------------------------------------------------------
# 5. Persistent storage (load/save memory to JSON)
# -----------------------------------------------------------------------------
def save_memory():
    """Save both ExactMemory and SWSTM to JSON."""
    data = {}
    try:
        if exact is not None:
            storage = {}
            for k, v in exact._storage.items():
                storage[k.hex()] = v.hex()
            data['exact'] = storage
            data['exact_fact_count'] = exact._fact_count
        if swstm is not None:
            if hasattr(swstm.memory, 'value_map'):
                data['swstm_value_map'] = swstm.memory.value_map
            data['swstm_fact_count'] = swstm.fact_count
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f)
        logger.info(f"Memory saved to {MEMORY_FILE}")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

def load_memory():
    global exact, swstm
    if not os.path.exists(MEMORY_FILE):
        logger.info("No existing memory file, starting fresh.")
        return
    try:
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
        if 'exact' in data:
            exact = get_exact()
            exact._storage = {}
            for k_hex, v_hex in data['exact'].items():
                exact._storage[bytes.fromhex(k_hex)] = bytes.fromhex(v_hex)
            exact._fact_count = data.get('exact_fact_count', len(exact._storage))
            logger.info(f"Loaded ExactMemory with {len(exact._storage)} facts.")
        if 'swstm_value_map' in data:
            swstm = get_swstm()
            if hasattr(swstm.memory, 'value_map'):
                swstm.memory.value_map = data['swstm_value_map']
            swstm.fact_count = data.get('swstm_fact_count', len(data['swstm_value_map']))
            logger.info(f"Loaded SWSTM with {swstm.fact_count} facts.")
    except Exception as e:
        logger.error(f"Failed to load memory: {e}")

# -----------------------------------------------------------------------------
# 6. FastAPI app with lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB + load memory
    init_db()
    load_memory()
    logger.info("Recallspection API started")
    yield
    # Shutdown: save memory
    save_memory()
    logger.info("Recallspection API shutting down")

app = FastAPI(
    title="Recallspection API",
    description="Dual-core exact memory with API keys, usage tracking, and agent detection",
    version="18.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# 7. API Key security with agent detection
# -----------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def is_agent_request(request: Request) -> bool:
    """Detect if request is from an AI agent (vs human)."""
    user_agent = request.headers.get("user-agent", "").lower()
    agent_patterns = [
        "python", "curl", "wget", "requests", "langchain", "llamaindex",
        "openai", "anthropic", "cohere", "mistral", "transformers",
        "pytorch", "tensorflow", "jupyter", "colab", "bot", "spider"
    ]
    for pattern in agent_patterns:
        if pattern in user_agent:
            return True
    # Also check if request is from a known AI framework
    referer = request.headers.get("referer", "").lower()
    if "colab" in referer or "notebook" in referer:
        return True
    return False

async def validate_api_key(request: Request, api_key: str = Depends(api_key_header)):
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key. Please provide X-API-Key header.")
    
    key_info = get_key_info(api_key)
    if key_info is None:
        raise HTTPException(status_code=403, detail="Invalid API Key or key deactivated.")
    
    # Check usage limit
    remaining = get_remaining_usage(api_key)
    if remaining <= 0:
        raise HTTPException(
            status_code=402,
            detail=f"Usage limit exceeded. Plan: {key_info['plan']}, Used: {key_info['usage']}, Limit: {key_info['limit']}. Please upgrade."
        )
    
    # Increment usage
    increment_usage(api_key)
    
    # Attach key_info to request state for later use (e.g., logging)
    request.state.key_info = key_info
    request.state.is_agent = is_agent_request(request)
    
    return key_info

# -----------------------------------------------------------------------------
# 8. Pydantic models
# -----------------------------------------------------------------------------
class AddRequest(BaseModel):
    key: str
    value: str

class AddResponse(BaseModel):
    status: str
    message: str
    backend: str = "swstm"
    remaining: int

class GetResponse(BaseModel):
    answers: List[str]
    backend: str = "swstm"
    remaining: int
    message: Optional[str] = None

class KeyResponse(BaseModel):
    api_key: str
    owner: str
    plan: str
    limit: int
    remaining: int

class UsageResponse(BaseModel):
    owner: str
    plan: str
    used: int
    limit: int
    remaining: int

# -----------------------------------------------------------------------------
# 9. Public endpoints (no auth required)
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Recallspection v18.0.0 is running",
        "docs": "/docs",
        "health": "/health",
        "signup": "/signup (create an API key)",
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend": "SWSTM v7.0 + ExactMemory",
        "swstm_loaded": swstm is not None,
        "exact_loaded": exact is not None,
        "facts_swstm": swstm.fact_count if swstm else 0,
        "facts_exact": exact._fact_count if exact else 0,
        "db_connected": os.path.exists(DB_FILE),
    }

# -----------------------------------------------------------------------------
# 10. Signup endpoint (create API key)
# -----------------------------------------------------------------------------
@app.post("/signup")
async def signup(owner: str, plan: str = "free"):
    """Create a new API key. Plans: free, pro, enterprise, agent_free, agent_pro, agent_enterprise."""
    valid_plans = ["free", "pro", "enterprise", "agent_free", "agent_pro", "agent_enterprise"]
    if plan not in valid_plans:
        raise HTTPException(status_code=400, detail=f"Invalid plan. Choose from: {valid_plans}")
    
    key = create_api_key(owner, plan)
    key_info = get_key_info(key)
    return KeyResponse(
        api_key=key,
        owner=key_info["owner"],
        plan=key_info["plan"],
        limit=key_info["limit"],
        remaining=key_info["limit"] - key_info["usage"],
    )

# -----------------------------------------------------------------------------
# 11. Usage endpoint (check remaining quota)
# -----------------------------------------------------------------------------
@app.get("/usage")
async def usage(api_key: str = Depends(validate_api_key)):
    """Check your current usage and remaining quota."""
    key_info = await api_key  # validate_api_key returns the key_info dict
    return UsageResponse(
        owner=key_info["owner"],
        plan=key_info["plan"],
        used=key_info["usage"],
        limit=key_info["limit"],
        remaining=key_info["limit"] - key_info["usage"],
    )

# -----------------------------------------------------------------------------
# 12. Admin endpoints (list/revoke keys)
# -----------------------------------------------------------------------------
@app.get("/admin/keys")
async def list_keys(admin_key: str = Header(...)):
    """List all API keys (admin only). Set RECALLSPECTION_ADMIN_KEY env var."""
    ADMIN_KEY = os.getenv("RECALLSPECTION_ADMIN_KEY", "")
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    conn = get_db()
    rows = conn.execute("SELECT key_id, owner, plan, usage, limit, created_at, last_used, is_active FROM api_keys").fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.post("/admin/revoke/{key_id}")
async def revoke_key(key_id: str, admin_key: str = Header(...)):
    """Revoke an API key (admin only)."""
    ADMIN_KEY = os.getenv("RECALLSPECTION_ADMIN_KEY", "")
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    conn = get_db()
    conn.execute("UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": f"Key {key_id} revoked"}

# -----------------------------------------------------------------------------
# 13. Protected endpoints (require API key)
# -----------------------------------------------------------------------------
@app.post("/add", response_model=AddResponse)
async def add_fact(
    request: Request,
    add_req: AddRequest,
    backend: str = "swstm",
    key_info: dict = Depends(validate_api_key),
):
    """Add a fact. Backend: swstm (default) or exact."""
    try:
        if backend == "exact":
            mem = get_exact()
            mem.add(add_req.key, add_req.value)
            remaining = get_remaining_usage(key_info["key_id"])
            return AddResponse(
                status="ok",
                message="Added to ExactMemory",
                backend="exact",
                remaining=remaining,
            )
        else:
            mem = get_swstm()
            if mem is None:
                raise HTTPException(status_code=503, detail="SWSTM engine not available")
            result = mem.add(add_req.key, add_req.value)
            remaining = get_remaining_usage(key_info["key_id"])
            return AddResponse(
                status="ok",
                message=result,
                backend="swstm",
                remaining=remaining,
            )
    except Exception as e:
        logger.exception("Error in /add")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get", response_model=GetResponse)
async def get_fact(
    request: Request,
    key: str,
    top_k: int = 1,
    backend: str = "swstm",
    key_info: dict = Depends(validate_api_key),
):
    """Retrieve value(s) for a query."""
    try:
        if backend == "exact":
            mem = get_exact()
            result = mem.get(key)
            remaining = get_remaining_usage(key_info["key_id"])
            if result is not None:
                return GetResponse(
                    answers=[str(result)],
                    backend="exact",
                    remaining=remaining,
                )
            else:
                return GetResponse(
                    answers=[],
                    backend="exact",
                    remaining=remaining,
                    message="Not found",
                )
        else:
            mem = get_swstm()
            if mem is None:
                raise HTTPException(status_code=503, detail="SWSTM engine not available")
            results = mem.get(key, top_k=top_k)
            remaining = get_remaining_usage(key_info["key_id"])
            if results:
                return GetResponse(
                    answers=results,
                    backend="swstm",
                    remaining=remaining,
                )
            else:
                return GetResponse(
                    answers=[],
                    backend="swstm",
                    remaining=remaining,
                    message="No match",
                )
    except Exception as e:
        logger.exception("Error in /get")
        raise HTTPException(status_code=500, detail=str(e))

# Exact endpoints (also protected)
@app.post("/exact/add")
async def add_exact(
    request: Request,
    add_req: AddRequest,
    key_info: dict = Depends(validate_api_key),
):
    mem = get_exact()
    mem.add(add_req.key, add_req.value)
    remaining = get_remaining_usage(key_info["key_id"])
    return {"status": "ok", "remaining": remaining}

@app.get("/exact/get")
async def get_exact(
    request: Request,
    key: str,
    key_info: dict = Depends(validate_api_key),
):
    mem = get_exact()
    result = mem.get(key)
    remaining = get_remaining_usage(key_info["key_id"])
    return {
        "answer": result,
        "remaining": remaining,
    } if result is not None else {
        "answer": None,
        "remaining": remaining,
        "message": "Not found",
    }

# -----------------------------------------------------------------------------
# 14. Agent-specific detection endpoint (returns plan suggestion)
# -----------------------------------------------------------------------------
@app.get("/agent-info")
async def agent_info(request: Request, key_info: dict = Depends(validate_api_key)):
    """Return info about the current request (agent vs human)."""
    is_agent = request.state.is_agent
    return {
        "is_agent": is_agent,
        "plan": key_info["plan"],
        "remaining": get_remaining_usage(key_info["key_id"]),
        "suggestion": "Consider upgrading to agent plan for higher limits." if is_agent and key_info["plan"].startswith("free") else None,
    }

# -----------------------------------------------------------------------------
# 15. Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
