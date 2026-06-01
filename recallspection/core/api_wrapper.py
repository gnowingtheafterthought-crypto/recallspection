from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from recallspection.core.swstem import SWSTM
from recallspection.core.tiered_index import TieredIndex

app = FastAPI(title='Recallspection SaaS API')

# Global instances for the managed service
dim = 512
vault = SWSTM(dim=dim)
index = TieredIndex(dim=dim)

class MemoryEntry(BaseModel):
    vector: List[float]
    label: str

class QueryRequest(BaseModel):
    vector: List[float]
    k: int = 1

@app.post('/store')
async def store_memory(entry: MemoryEntry):
    try:
        vec = np.array(entry.vector).astype('float32')
        vault.add(vec, entry.label)
        index.add(vec.reshape(1, -1))
        return {'status': 'success', 'vault_ptr': vault.ptr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/recall')
async def recall_memory(request: QueryRequest):
    vec = np.array(request.vector).astype('float32')
    sims, ids = index.search(vec.reshape(1, -1), k=request.k)
    
    results = []
    for i in range(len(ids[0])):
        idx = ids[0][i]
        results.append({
            'label': vault.values[idx],
            'similarity': float(sims[0][i])
        })
    return {'results': results}
