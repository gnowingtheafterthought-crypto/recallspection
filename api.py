"""
Recallspection API – v18.0.0 (SWSTM backend + ExactMemory fallback)
Serves both neural (SWSTM) and cryptographic (ExactMemory) memory engines.
"""

import os
import json
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ----- Import both engines -----
from recallspection.swstm import SWSTMEngine
from recallspection.exact import ExactMemory

# -----------------------------------------------------------------------------
# 1. Initialize engines
# -----------------------------------------------------------------------------

# SWSTM engine (primary)
swstm = SWSTMEngine(
    mode=os.getenv("SWSTM_MODE", "auto"),
    flat_num_slots=int(os.getenv("SWSTM_FLAT_SLOTS", 200)),
    hierarchical_num_clusters=int(os.getenv("SWSTM_HIER_CLUSTERS", 50)),
    hierarchical_slots_per_expert=int(os.getenv("SWSTM_HIER_SLOTS", 2000)),
    pq_num_clusters=int(os.getenv("SWSTM_PQ_CLUSTERS", 1000)),
    pq_facts_per_cluster=int(os.getenv("SWSTM_PQ_PER_CLUSTER", 1000)),
    auto_threshold_flat=int(os.getenv("SWSTM_THRESHOLD_FLAT", 1000)),
    auto_threshold_hier=int(os.getenv("SWSTM_THRESHOLD_HIER", 50000)),
)

# ExactMemory (compliance fallback)
exact = ExactMemory()

# -----------------------------------------------------------------------------
# 2. FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Recallspection API",
    description="Exact associative memory with dual‑core (SWSTM v7.0 + ExactMemory)",
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
# 3. Main endpoints (SWSTM by default)
# -----------------------------------------------------------------------------

@app.post("/add", response_model=AddResponse)
async def add_fact(
    request: AddRequest,
    backend: str = Query("swstm", enum=["swstm", "exact"]),
):
    """
    Add a fact.
    - backend: "swstm" (neural, fuzzy) or "exact" (cryptographic, tamper‑evident).
    """
    try:
        if backend == "exact":
            exact.add(request.key, request.value)
            return AddResponse(status="ok", message="Added to ExactMemory", backend="exact")
        else:
            result = swstm.add(request.key, request.value)
            return AddResponse(status="ok", message=result, backend="swstm")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get", response_model=GetResponse)
async def get_fact(
    key: str,
    top_k: int = 1,
    backend: str = Query("swstm", enum=["swstm", "exact"]),
):
    """
    Retrieve value(s) for a query.
    - backend: "swstm" or "exact".
    """
    try:
        if backend == "exact":
            result = exact.get(key)
            if result is not None:
                return GetResponse(answers=[result], backend="exact")
            else:
                return GetResponse(answers=[], backend="exact", message="Not found in ExactMemory")
        else:
            results = swstm.get(key, top_k=top_k)
            if results:
                return GetResponse(answers=results, backend="swstm")
            else:
                return GetResponse(answers=[], backend="swstm", message="No matching fact found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Also accept POST for large queries
@app.post("/get", response_model=GetResponse)
async def get_fact_post(request: dict):
    key = request.get("key")
    top_k = request.get("top_k", 1)
    backend = request.get("backend", "swstm")
    return await get_fact(key, top_k, backend)

# -----------------------------------------------------------------------------
# 4. PQ‑specific endpoint (trigger compression)
# -----------------------------------------------------------------------------

@app.post("/fit_pq")
async def fit_pq():
    """
    For PQ mode: fit router centroids, PQ codebooks, and compress all pending keys.
    Only needed if SWSTM is in PQ mode and facts were added without automatic fit.
    """
    try:
        swstm.fit_pq()
        return {"status": "ok", "message": f"PQ fitted with {swstm.fact_count} facts."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# 5. ExactMemory‑only endpoints (for compliance)
# -----------------------------------------------------------------------------

@app.post("/exact/add")
async def add_exact(request: AddRequest):
    exact.add(request.key, request.value)
    return {"status": "ok", "message": "Added to ExactMemory"}

@app.get("/exact/get")
async def get_exact(key: str):
    result = exact.get(key)
    if result is not None:
        return {"answer": result}
    return {"answer": None, "message": "Not found"}

# -----------------------------------------------------------------------------
# 6. Health check
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "backend": "SWSTM v7.0 + ExactMemory",
        "swstm_facts": swstm.fact_count,
        "swstm_mode": type(swstm.memory).__name__ if swstm.memory else "uninitialized",
    }

# -----------------------------------------------------------------------------
# 7. Run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
