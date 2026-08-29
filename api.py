"""
Recallspection API – v18.0.0 (SWSTM backend)
Serves the SWSTM neural memory engine for fuzzy/paraphrase queries.
Compatibility layer: endpoints remain exactly the same.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ----- NEW: SWSTM engine instead of ExactMemory -----
from recallspection.swstm import SWSTMEngine
# from recallspection.exact import ExactMemory   # kept for reference / fallback

# -----------------------------------------------------------------------------
# 1. Initialize the engine
# -----------------------------------------------------------------------------

# Use "auto" mode: flat for <1000 facts, hierarchical for more.
# You can also force "flat" or "hierarchical" by changing mode=...
memory = SWSTMEngine(
    mode="auto",
    flat_num_slots=200,
    hierarchical_num_clusters=50,
    hierarchical_slots_per_expert=2000,
)

# -----------------------------------------------------------------------------
# 2. FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Recallspection API",
    description="Exact associative memory with SWSTM v7.0",
    version="18.0.0",
)

class AddRequest(BaseModel):
    key: str          # can also be dict/bytes, but we keep simple
    value: str

class AddResponse(BaseModel):
    status: str
    message: str

class GetRequest(BaseModel):
    key: str
    top_k: Optional[int] = 1

class GetResponse(BaseModel):
    answers: List[str]
    message: Optional[str] = None

# -----------------------------------------------------------------------------
# 3. Endpoints
# -----------------------------------------------------------------------------

@app.post("/add", response_model=AddResponse)
async def add_fact(request: AddRequest):
    """
    Add a fact to the memory.
    - key: any natural language string (e.g., "capital of France")
    - value: the answer (e.g., "Paris")
    """
    try:
        result = memory.add(request.key, request.value)
        return AddResponse(status="ok", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get", response_model=GetResponse)
async def get_fact(key: str, top_k: int = 1):
    """
    Retrieve the top‑k values for a query.
    The query can be a paraphrase or exact match.
    """
    try:
        results = memory.get(key, top_k=top_k)
        if results:
            return GetResponse(answers=results)
        else:
            return GetResponse(answers=[], message="No matching fact found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get", response_model=GetResponse)   # also accept POST for large queries
async def get_fact_post(request: GetRequest):
    return await get_fact(request.key, request.top_k)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "backend": "SWSTM v7.0", "facts": memory.fact_count}

# -----------------------------------------------------------------------------
# 4. (Optional) Keep ExactMemory for compliance via a separate endpoint
# -----------------------------------------------------------------------------
# If you want to expose the old hash‑table core, you can instantiate it and
# add an /exact endpoint. For now, we keep it commented.

# from recallspection.exact import ExactMemory
# exact_memory = ExactMemory()
# @app.post("/exact/add")
# async def add_exact(...): ...

# -----------------------------------------------------------------------------
# 5. Run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
