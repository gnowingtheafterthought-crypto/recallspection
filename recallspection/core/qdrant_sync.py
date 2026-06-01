import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantLiveSync:
    def __init__(self, collection_name='recallspection_vault', dim=512):
        self.client = QdrantClient(':memory:')
        self.collection_name = collection_name
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            )
