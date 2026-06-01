import requests
import numpy as np

# Configuration for the Managed SaaS
API_URL = "https://api.recallspection.com"

def store_memory_remotely(label, vector):
    payload = {"vector": vector.tolist(), "label": label}
    response = requests.post(f"{API_URL}/store", json=payload)
    return response.json()

def recall_memory_remotely(vector):
    payload = {"vector": vector.tolist(), "k": 1}
    response = requests.post(f"{API_URL}/recall", json=payload)
    return response.json()

print("Example client configuration initialized.")
