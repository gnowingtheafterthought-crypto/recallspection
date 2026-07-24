# ================================================================
# 🧠 RECALLSPECTION API v12 — Supabase Edition
# ================================================================
import os
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from collections import defaultdict, deque
import hashlib
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()  # Optional: load from .env if present

app = FastAPI(title="Recallspection API", version="v12")
api_key_header = APIKeyHeader(name="X-API-Key")

# ---------- SUPABASE SETUP ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- ENGINE ----------
class FixedPrototypeSWSTM:
    # ... (unchanged, same as your existing engine) ...
    # (I'll keep it compact here, but paste your full engine code below)

# Instantiate engine
engine = FixedPrototypeSWSTM(dim=50)

# ---------- MODELS ----------
class StoreRequest(BaseModel):
    subject: list
    object: list
    value: str = None

class RetrieveRequest(BaseModel):
    query: list
    return_raw: bool = True

class ComposeRequest(BaseModel):
    start: list
    end: list

class ChainRequest(BaseModel):
    start: list
    shift: list
    hops: int = 10

# ---------- AUTH (Supabase) ----------
def validate_api_key(api_key: str = Security(api_key_header)):
    try:
        resp = supabase.table("users").select("id", "plan").eq("api_key", api_key).eq("is_active", True).execute()
        if not resp.data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return resp.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ---------- ENDPOINTS ----------
@app.post("/store")
def store(req: StoreRequest, user=Security(validate_api_key)):
    subj = torch.tensor(req.subject, dtype=torch.float32)
    obj = torch.tensor(req.object, dtype=torch.float32)
    engine.store(subj, obj, value=req.value)
    return {"status": "stored"}

@app.post("/retrieve")
def retrieve(req: RetrieveRequest, user=Security(validate_api_key)):
    query = torch.tensor(req.query, dtype=torch.float32)
    result, value, sim, conf = engine.retrieve(query, return_raw=req.return_raw)
    if conf:
        return {"value": value, "similarity": sim, "found": True}
    return {"found": False, "similarity": sim}

@app.post("/compose")
def compose(req: ComposeRequest, user=Security(validate_api_key)):
    start = torch.tensor(req.start, dtype=torch.float32)
    end = torch.tensor(req.end, dtype=torch.float32)
    rel = engine.compose_path(start, end)
    if rel is not None:
        return {"relation": rel.tolist()}
    return {"found": False}

@app.post("/chain")
def chain(req: ChainRequest, user=Security(validate_api_key)):
    start = torch.tensor(req.start, dtype=torch.float32)
    shift = torch.tensor(req.shift, dtype=torch.float32)
    hops = engine.run_chain(start, shift, req.hops)
    return {"hops": hops}

@app.get("/")
def root():
    return {"name": "Recallspection API", "status": "online", "version": "v12"}
