import os
import json
import logging
import hashlib
import time
from contextlib import asynccontextmanager
from typing import List, Optional
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
API_KEY = os.getenv("RECALLSPECTION_API_KEY", "supersecretkey")  # Change this!
MEMORY_FILE = os.getenv("RECALLSPECTION_MEMORY_FILE", "memory.json")
AUTO_SAVE_INTERVAL = int(os.getenv("RECALLSPECTION_AUTO_SAVE", "60"))  # seconds

# -----------------------------------------------------------------------------
# 3. Lazy imports (SWSTM/ExactMemory)
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
        swstm = SWSTMEngine(mode=os.getenv("SWSTM_MODE", "flat"), flat_num_slots=200)
        logger.info("SWSTMEngine initialized")
    return swstm

# -----------------------------------------------------------------------------
# 4. Persistent storage: load/save the memory state
# -----------------------------------------------------------------------------
def save_memory():
    """Save both ExactMemory and SWSTM (if used) to a JSON file."""
    data = {}
    try:
        # Save ExactMemory (simple dict)
        if exact is not None:
            # ExactMemory stores bytes -> bytes; we need to convert to hex/string
            storage = {}
            for k, v in exact._storage.items():
                storage[k.hex()] = v.hex()  # convert bytes to hex strings
            data['exact'] = storage
            data['exact_fact_count'] = exact._fact_count

        # Save SWSTM (we need to serialize its internal state)
        if swstm is not None:
            # SWSTM has memory that is complex (torch tensors, etc.)
            # For simplicity, we'll save only the fact_count and value_map for flat mode.
            # For full persistence, you'd need to save the entire model parameters.
            # We'll implement a basic version that saves fact_count and value_map.
            if hasattr(swstm.memory, 'value_map'):
                data['swstm_value_map'] = swstm.memory.value_map
            data['swstm_fact_count'] = swstm.fact_count

        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f)
        logger.info(f"Memory saved to {MEMORY_FILE}")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

def load_memory():
    """Load memory state from JSON file."""
    global exact, swstm
    if not os.path.exists(MEMORY_FILE):
        logger.info("No existing memory file found, starting fresh.")
        return
    try:
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
        # Load ExactMemory
        if 'exact' in data:
            exact = get_exact()
            exact._storage = {}
            for k_hex, v_hex in data['exact'].items():
                exact._storage[bytes.fromhex(k_hex)] = bytes.fromhex(v_hex)
            exact._fact_count = data.get('exact_fact_count', len(exact._storage))
            logger.info(f"Loaded ExactMemory with {len(exact._storage)} facts.")
        # Load SWSTM (flat mode)
        if 'swstm_value_map' in data and swstm is not None:
            swstm = get_swstm()
            if hasattr(swstm.memory, 'value_map'):
                swstm.memory.value_map = data['swstm_value_map']
            swstm.fact_count = data.get('swstm_fact_count', len(data['swstm_value_map']))
            logger.info(f"Loaded SWSTM with {swstm.fact_count} facts.")
    except Exception as e:
        logger.error(f"Failed to load memory: {e}")

# -----------------------------------------------------------------------------
# 5. FastAPI app with lifespan (save on shutdown, load on startup)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load memory
    load_memory()
    yield
    # Shutdown: save memory
    save_memory()

app = FastAPI(
    title="Recallspection API",
    description="Dual-core exact memory (SWSTM + ExactMemory) with auth and persistence",
    version="18.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# 6. API Key security
# -----------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Optional: public endpoints without auth (health, root)
@app.get("/")
async def root():
    return {"message": "Recallspection v18.0.0 is running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend": "SWSTM v7.0 + ExactMemory",
        "swstm_loaded": swstm is not None,
        "exact_loaded": exact is not None,
        "facts_swstm": swstm.fact_count if swstm else 0,
        "facts_exact": exact._fact_count if exact else 0,
    }

# All protected endpoints require API key
@app.post("/add")
async def add_fact(
    request: Request,
    backend: str = "swstm",
    api_key: str = Depends(validate_api_key)
):
    data = await request.json()
    key = data.get("key")
    value = data.get("value")
    if not key or not value:
        raise HTTPException(status_code=400, detail="Missing key or value")
    try:
        if backend == "exact":
            mem = get_exact()
            mem.add(key, value)
            return {"status": "ok", "message": "Added to ExactMemory", "backend": "exact"}
        else:
            mem = get_swstm()
            if mem is None:
                raise HTTPException(status_code=503, detail="SWSTM engine not available")
            result = mem.add(key, value)
            return {"status": "ok", "message": result, "backend": "swstm"}
    except Exception as e:
        logger.exception("Error in /add")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get")
async def get_fact(
    key: str,
    top_k: int = 1,
    backend: str = "swstm",
    api_key: str = Depends(validate_api_key)
):
    try:
        if backend == "exact":
            mem = get_exact()
            result = mem.get(key)
            if result is not None:
                return {"answers": [str(result)], "backend": "exact"}
            else:
                return {"answers": [], "backend": "exact", "message": "Not found"}
        else:
            mem = get_swstm()
            if mem is None:
                raise HTTPException(status_code=503, detail="SWSTM engine not available")
            results = mem.get(key, top_k=top_k)
            if results:
                return {"answers": results, "backend": "swstm"}
            else:
                return {"answers": [], "backend": "swstm", "message": "No match"}
    except Exception as e:
        logger.exception("Error in /get")
        raise HTTPException(status_code=500, detail=str(e))

# Exact endpoints (also protected)
@app.post("/exact/add")
async def add_exact(request: Request, api_key: str = Depends(validate_api_key)):
    data = await request.json()
    key = data.get("key")
    value = data.get("value")
    if not key or not value:
        raise HTTPException(status_code=400, detail="Missing key or value")
    mem = get_exact()
    mem.add(key, value)
    return {"status": "ok"}

@app.get("/exact/get")
async def get_exact(key: str, api_key: str = Depends(validate_api_key)):
    mem = get_exact()
    result = mem.get(key)
    return {"answer": result} if result is not None else {"answer": None, "message": "Not found"}

# -----------------------------------------------------------------------------
# 7. Periodic auto‑save (optional – you can run a background task)
# -----------------------------------------------------------------------------
# For simplicity, we rely on shutdown save. You can add a background task if needed.

# -----------------------------------------------------------------------------
# 8. Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
