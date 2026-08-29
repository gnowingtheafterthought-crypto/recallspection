import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recallspection-api")

# -----------------------------------------------------------------------------
# Lazy imports – avoid heavy load at startup
# -----------------------------------------------------------------------------
swstm = None
exact = None

def get_exact():
    global exact
    if exact is None:
        try:
            from recallspection.exact import ExactMemory
            exact = ExactMemory()
            logger.info("ExactMemory initialized")
        except Exception as e:
            logger.error(f"Failed to load ExactMemory: {e}")
            raise
    return exact

def get_swstm():
    global swstm
    if swstm is None:
        try:
            from recallspection.swstm import SWSTMEngine
            # Load with minimal config; force flat mode to avoid hierarchical/PQ complexity
            swstm = SWSTMEngine(
                mode=os.getenv("SWSTM_MODE", "flat"),
                flat_num_slots=200,
                key_dim=384,
                temperature=0.01,
                margin=0.2,
            )
            logger.info("SWSTMEngine initialized")
        except Exception as e:
            logger.error(f"Failed to load SWSTMEngine: {e}")
            # Fallback to ExactMemory only
            swstm = None
            raise
    return swstm

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Recallspection API",
    description="Dual-core exact memory (SWSTM + ExactMemory)",
    version="18.0.0",
)

class AddRequest(BaseModel):
    key: str
    value: str

class AddResponse(BaseModel):
    status: str
    message: str
    backend: str = "swstm"

class GetResponse(BaseModel):
    answers: List[str]
    backend: str = "swstm"
    message: Optional[str] = None

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Recallspection v18.0.0 is running"}

@app.get("/health")
async def health():
    swstm_ok = swstm is not None
    exact_ok = exact is not None
    return {
        "status": "healthy",
        "backend": "SWSTM v7.0 + ExactMemory",
        "swstm_loaded": swstm_ok,
        "exact_loaded": exact_ok,
    }

@app.post("/add")
async def add_fact(request: AddRequest, backend: str = Query("swstm", enum=["swstm", "exact"])):
    try:
        if backend == "exact":
            mem = get_exact()
            mem.add(request.key, request.value)
            return AddResponse(status="ok", message="Added to ExactMemory", backend="exact")
        else:
            mem = get_swstm()
            if mem is None:
                raise HTTPException(status_code=503, detail="SWSTM engine not available")
            result = mem.add(request.key, request.value)
            return AddResponse(status="ok", message=result, backend="swstm")
    except Exception as e:
        logger.exception("Error in /add")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get")
async def get_fact(key: str, top_k: int = 1, backend: str = Query("swstm", enum=["swstm", "exact"])):
    try:
        if backend == "exact":
            mem = get_exact()
            result = mem.get(key)
            if result is not None:
                return GetResponse(answers=[str(result)], backend="exact")
            else:
                return GetResponse(answers=[], backend="exact", message="Not found")
        else:
            mem = get_swstm()
            if mem is None:
                raise HTTPException(status_code=503, detail="SWSTM engine not available")
            results = mem.get(key, top_k=top_k)
            if results:
                return GetResponse(answers=results, backend="swstm")
            else:
                return GetResponse(answers=[], backend="swstm", message="No match")
    except Exception as e:
        logger.exception("Error in /get")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# (Optional) Exact endpoints
# -----------------------------------------------------------------------------
@app.post("/exact/add")
async def add_exact(request: AddRequest):
    mem = get_exact()
    mem.add(request.key, request.value)
    return {"status": "ok"}

@app.get("/exact/get")
async def get_exact(key: str):
    mem = get_exact()
    result = mem.get(key)
    return {"answer": result} if result is not None else {"answer": None, "message": "Not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
