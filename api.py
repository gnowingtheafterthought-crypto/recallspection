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

load_dotenv()

app = FastAPI(title="Recallspection API", version="v12")
api_key_header = APIKeyHeader(name="X-API-Key")

# ---------- SUPABASE SETUP ----------
SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_KEY=os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🛠️ FIX FOR SUPABASE 401 ERROR WITH NEW sb_secret_ KEYS
# The new Supabase keys fail if passed in the Authorization header. 
# We strip it out to prevent the "Invalid API key" database error.
try:
    if "Authorization" in supabase.postgrest.session.headers:
        del supabase.postgrest.session.headers["Authorization"]
    print("✅ Supabase client initialized. Stripped forbidden Authorization header.")
except Exception as e:
    print(f"Header cleanup warning: {e}")

# ---------- ENGINE (Full Implementation) ----------
class FixedPrototypeSWSTM:
    def __init__(self, dim=50, num_prototypes=256, conf_thresh=0.85):
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.conf_thresh = conf_thresh
        self.prototypes = torch.randn(num_prototypes, dim)
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        self.prototypes_np = self.prototypes.numpy().astype('float32')
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.prototypes_np)
        self.slots = {i: [] for i in range(num_prototypes)}
        self.vec_to_id = {}
        self.id_to_vec = {}
        self.next_id = 0
        self.graph = defaultdict(list)
        self.next_ptr_map = {}
        self.transition_history = defaultdict(list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prototypes = self.prototypes.to(self.device)

    def _normalize(self, v):
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).float()
        return F.normalize(v, p=2, dim=0) if v.dim() == 1 else F.normalize(v, p=2, dim=1)

    def _get_id(self, vec):
        vec = self._normalize(vec)
        key = vec.detach().cpu().numpy().tobytes()
        md5 = hashlib.md5(key).digest()
        if md5 not in self.vec_to_id:
            self.vec_to_id[md5] = self.next_id
            self.id_to_vec[self.next_id] = vec.detach()
            self.next_id += 1
        return self.vec_to_id[md5]

    def _get_routing_slots(self, vec):
        vec_np = vec.detach().cpu().numpy().astype('float32').reshape(1, -1)
        distances, indices = self.index.search(vec_np, 1)
        return [int(indices[0][0])]

    def store(self, subject_vec, object_vec=None, value=None, allow_overwrite=False):
        if isinstance(subject_vec, np.ndarray):
            subject_vec = torch.from_numpy(subject_vec).float().to(self.device)
        if object_vec is not None and isinstance(object_vec, np.ndarray):
            object_vec = torch.from_numpy(object_vec).float().to(self.device)
        raw_sub = subject_vec.detach().clone()
        sub = self._normalize(subject_vec)
        sub_id = self._get_id(sub)
        if object_vec is not None:
            raw_obj = object_vec.detach().clone()
            obj = self._normalize(object_vec)
            obj_id = self._get_id(obj)
            rel = (obj - sub).detach()
            if sub_id in self.next_ptr_map and self.next_ptr_map[sub_id] != obj_id and not allow_overwrite:
                return False
            if (sub_id, obj_id) not in self.transition_history[sub_id]:
                self.transition_history[sub_id].append(obj_id)
            slot_id = self._get_routing_slots(sub)[0]
            self.slots[slot_id].append((sub.detach(), raw_sub, value or f"fact_{sub_id}", obj.detach(), raw_obj))
            self.next_ptr_map[sub_id] = obj_id
            self.graph[sub_id].append((obj_id, rel))
            return True
        else:
            slot_id = self._get_routing_slots(sub)[0]
            self.slots[slot_id].append((sub.detach(), raw_sub, value or f"key_{sub_id}", None, None))
            return True

    def retrieve(self, query_vec, return_raw=True):
        if isinstance(query_vec, np.ndarray):
            query_vec = torch.from_numpy(query_vec).float().to(self.device)
        query = self._normalize(query_vec)
        slots = self._get_routing_slots(query)
        best_sim, best_item = -1.0, None
        for slot_id in slots:
            for norm_sub, raw_sub, value, norm_obj, raw_obj in self.slots.get(slot_id, []):
                sim = torch.dot(query, norm_sub).item()
                if sim > best_sim:
                    best_sim = sim
                    best_item = (raw_sub, value, raw_obj)
        if best_sim < self.conf_thresh:
            return None, None, best_sim, False
        if return_raw:
            return best_item[0], best_item[1], best_sim, True
        else:
            return best_item[2], best_item[1], best_sim, True

    def compose_path(self, start_vec, end_vec):
        start_id = self._get_id(self._normalize(start_vec))
        end_id = self._get_id(self._normalize(end_vec))
        if start_id == end_id:
            return torch.zeros(self.dim).to(self.device)
        queue, visited = deque([start_id]), {start_id}
        while queue:
            curr = queue.popleft()
            for nbr, _ in self.graph.get(curr, []):
                if nbr == end_id:
                    return self.id_to_vec[end_id] - self.id_to_vec[start_id]
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return None

    def run_chain(self, start_vec, shift, num_hops):
        current = self._normalize(start_vec)
        successes = 0
        for _ in range(num_hops):
            nxt = self._normalize(current + shift)
            self.store(current, nxt, value=f"step_{successes+1}")
            current = nxt
            successes += 1
        return successes

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
