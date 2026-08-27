#!/usr/bin/env python3
"""
RECALLSPECTION API — Production-grade Exact Memory Server

Upgraded to v17:
- Cryptographic exact core (SHA3-256 + zlib)
- Persistent memory (saves to memory.db)
- Web interface served at root
- Endpoints: /chat, /history, /stats, /forget

Deployment:
    uvicorn api:app --host 0.0.0.0 --port 8000

Live at:
    https://recallspection.onrender.com
"""

import os
import sys
import json
import time
import random
import re
import hashlib
import zlib
import struct
import pickle
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# 1. EXACTMEMORY CORE (with Persistence)
# -----------------------------------------------------------------------------
try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

@dataclass
class ExactConfig:
    quorum_size: int = 3
    compress: bool = True
    hash_algorithm: str = "blake3" if BLAKE3_AVAILABLE else "sha3_256"

class ExactMemory:
    def __init__(self, config: Optional[ExactConfig] = None):
        self.config = config or ExactConfig()
        self._storage: Dict[str, bytes] = {}
        self._metadata: Dict[str, bytes] = {}
        self._write_count = 0
        self._read_count = 0
        self._verification_failures = 0
        self._hash_len = 32

    def _hash(self, data: bytes) -> bytes:
        if self.config.hash_algorithm == "blake3" and BLAKE3_AVAILABLE:
            return blake3.blake3(data).digest()
        return hashlib.sha3_256(data).digest()

    def _pack_metadata(self, q_hashes: List[bytes], v_hash: bytes, ts: float) -> bytes:
        return b''.join(q_hashes) + v_hash + struct.pack('<d', ts)

    def _unpack_metadata(self, packed: bytes) -> Tuple[List[bytes], bytes, float]:
        h = self._hash_len
        q = self.config.quorum_size
        qh = [packed[i*h:(i+1)*h] for i in range(q)]
        off = q*h
        vh = packed[off:off+h]
        ts = struct.unpack('<d', packed[off+h:off+h+8])[0]
        return qh, vh, ts

    def add(self, key: str, value: Any) -> bool:
        try:
            v_bytes = json.dumps(value, sort_keys=True).encode('utf-8')
            v_hash = self._hash(v_bytes)
            if self.config.compress:
                stored = zlib.compress(v_bytes, level=6)
            else:
                stored = v_bytes

            key_b = key.encode()
            base = self._hash(key_b)
            q_hashes = [self._hash(key_b + base + str(i).encode()) for i in range(self.config.quorum_size)]

            self._storage[key] = stored
            self._metadata[key] = self._pack_metadata(q_hashes, v_hash, time.time())
            self._write_count += 1
            return True
        except Exception:
            return False

    def get(self, key: str) -> Optional[Any]:
        self._read_count += 1
        if key not in self._storage:
            return None
        stored = self._storage[key]
        packed = self._metadata.get(key)
        if packed is None:
            return None
        try:
            qh, vh, ts = self._unpack_metadata(packed)
            key_b = key.encode()
            base = self._hash(key_b)
            for i in range(self.config.quorum_size):
                if qh[i] != self._hash(key_b + base + str(i).encode()):
                    self._verification_failures += 1
                    return None
            if self.config.compress:
                v_bytes = zlib.decompress(stored)
            else:
                v_bytes = stored
            if self._hash(v_bytes) != vh:
                self._verification_failures += 1
                return None
            return json.loads(v_bytes.decode())
        except Exception:
            return None

    def stats(self) -> Dict[str, Any]:
        return {
            'writes': self._write_count,
            'reads': self._read_count,
            'verification_failures': self._verification_failures,
            'stored': len(self._storage),
            'exact_match_ratio': 1.0 if self._verification_failures == 0 else 0.0,
        }

    def clear(self):
        self._storage.clear()
        self._metadata.clear()
        self._write_count = 0
        self._read_count = 0
        self._verification_failures = 0

    def save(self, filename: str = "memory.db"):
        data = {
            'storage': self._storage,
            'metadata': self._metadata,
            'write_count': self._write_count,
            'read_count': self._read_count,
            'verification_failures': self._verification_failures,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename: str = "memory.db"):
        if not os.path.exists(filename):
            return
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self._storage = data['storage']
        self._metadata = data['metadata']
        self._write_count = data['write_count']
        self._read_count = data['read_count']
        self._verification_failures = data['verification_failures']

# -----------------------------------------------------------------------------
# 2. RULE-BASED NLP ENGINE (StdlibMind)
# -----------------------------------------------------------------------------
class StdlibMind:
    def __init__(self, memory: ExactMemory):
        self.memory = memory
        self.memory.add("agent_name", "Recallspection Assistant")
        self.memory.add("session_start", time.time())

        self.rules = [
            (r"hi|hello|hey|howdy", self.respond_greeting),
            (r"good morning|good afternoon|good evening", self.respond_greeting),
            (r"my name is (\w+)", self.remember_name),
            (r"i am (\w+)", self.remember_name),
            (r"i like (.*)", self.remember_preference),
            (r"i prefer (.*)", self.remember_preference),
            (r"my favorite (.*) is (.*)", self.remember_favorite),
            (r"what is my name\??", self.ask_name),
            (r"who am i\??", self.ask_name),
            (r"what do i like\??", self.ask_preference),
            (r"what is my favorite (.*)", self.ask_favorite),
            (r"what time is it", self.ask_time),
            (r"how many memories do you have", self.ask_memory_count),
            (r"what do you remember", self.ask_all_memories),
            (r".*", self.fallback),
        ]

    def respond_greeting(self, match):
        return random.choice(["Hello! How can I help?", "Hi! Tell me something about yourself.", "Greetings! I remember exactly."])

    def remember_name(self, match):
        name = match.group(1)
        self.memory.add("user_name", name)
        return f"Nice to meet you, {name}! I'll remember that."

    def remember_preference(self, match):
        pref = match.group(1)
        self.memory.add("user_preference", pref)
        return f"Okay, I'll remember that you like {pref}."

    def remember_favorite(self, match):
        category, item = match.group(1), match.group(2)
        self.memory.add(f"favorite_{category.strip()}", item)
        return f"Got it! Your favorite {category} is {item}."

    def ask_name(self, match):
        name = self.memory.get("user_name")
        return f"Your name is {name}." if name else "I don't know your name yet."

    def ask_preference(self, match):
        pref = self.memory.get("user_preference")
        return f"You told me you like {pref}." if pref else "You haven't told me what you like yet."

    def ask_favorite(self, match):
        category = match.group(1).strip()
        item = self.memory.get(f"favorite_{category}")
        return f"Your favorite {category} is {item}." if item else f"You haven't told me your favorite {category} yet."

    def ask_time(self, match):
        return f"The current time is {time.strftime('%H:%M:%S')}."

    def ask_memory_count(self, match):
        return f"I have stored {self.memory.stats()['stored']} facts."

    def ask_all_memories(self, match):
        keys = list(self.memory._storage.keys())
        if not keys:
            return "My memory is empty."
        return f"I remember: {', '.join(keys[:5])}" + (f" ... and {len(keys)-5} more." if len(keys) > 5 else "")

    def fallback(self, match):
        return random.choice([
            "I'm not sure I understand. Tell me your name, or what you like.",
            "Try saying: 'My name is ...' or 'I like ...'"
        ])

    def process(self, user_input: str) -> Optional[str]:
        user_input = user_input.strip().lower()
        for pattern, handler in self.rules:
            match = re.match(pattern, user_input)
            if match and handler:
                return handler(match)
        return self.fallback(None)

# -----------------------------------------------------------------------------
# 3. FASTAPI APPLICATION
# -----------------------------------------------------------------------------
app = FastAPI(title="Recallspection API", version="17.0.0")

# Global memory instance
memory = None
mind = None

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# -----------------------------------------------------------------------------
# 4. LIFECYCLE EVENTS (Load/Save)
# -----------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global memory, mind
    memory = ExactMemory()
    if os.path.exists("memory.db"):
        memory.load("memory.db")
        print("✓ Loaded existing memory.")
    else:
        print("✓ No existing memory found. Starting fresh.")
    mind = StdlibMind(memory)

@app.on_event("shutdown")
def shutdown_event():
    if memory:
        memory.save("memory.db")
        print("✓ Memory saved.")

# -----------------------------------------------------------------------------
# 5. ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface."""
    return HTML_PAGE

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message."""
    global memory, mind
    if not mind:
        raise HTTPException(503, "Memory not initialized.")
    
    user_input = request.message.strip()
    if not user_input:
        return ChatResponse(response="Please say something.")
    
    # Special commands
    if user_input.lower() == "/forget":
        memory.clear()
        memory.save("memory.db")
        return ChatResponse(response="Memory cleared.")
    
    if user_input.lower() == "/exit":
        return ChatResponse(response="Goodbye! Remembering everything.")
    
    # Process via Mind
    response = mind.process(user_input)
    if response:
        # Store turn in memory
        turn = {"user": user_input, "agent": response, "timestamp": time.time()}
        memory.add(f"turn_{int(time.time())}", turn)
        memory.save("memory.db")
        return ChatResponse(response=response)
    else:
        return ChatResponse(response="I didn't understand that.")

@app.get("/history")
async def history():
    """Get all stored memories as JSON."""
    if not memory:
        raise HTTPException(503, "Memory not initialized.")
    data = {}
    for k, v in memory._storage.items():
        try:
            v_bytes = zlib.decompress(v)
            data[k] = json.loads(v_bytes.decode())
        except:
            data[k] = "[binary data]"
    return JSONResponse(data)

@app.get("/stats")
async def stats():
    """Get memory statistics."""
    if not memory:
        raise HTTPException(503, "Memory not initialized.")
    return JSONResponse(memory.stats())

@app.post("/forget")
async def forget():
    """Clear all memory."""
    if not memory:
        raise HTTPException(503, "Memory not initialized.")
    memory.clear()
    memory.save("memory.db")
    return JSONResponse({"status": "Memory cleared."})

# -----------------------------------------------------------------------------
# 6. HTML PAGE (Embedded)
# -----------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Recallspection — Local AI Memory</title>
    <style>
        body {
            background: #0a0a1a;
            color: #e0e0ff;
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        #chat {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #333366;
            padding: 20px;
            border-radius: 8px;
            background: #12122a;
            margin-bottom: 20px;
        }
        #chat div {
            margin: 6px 0;
            padding: 8px 12px;
            border-radius: 6px;
        }
        .user { color: #88ddff; text-align: right; }
        .agent { color: #ddff88; text-align: left; }
        #input-area {
            display: flex;
            gap: 10px;
        }
        #input-area input {
            flex: 1;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #333366;
            background: #1a1a33;
            color: #fff;
            font-size: 16px;
        }
        #input-area button {
            padding: 12px 24px;
            border-radius: 6px;
            border: none;
            background: #4466ff;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
        }
        #input-area button:hover {
            background: #6688ff;
        }
        .status {
            color: #8888aa;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .timestamp {
            color: #555577;
            font-size: 11px;
            margin: 0 0 0 10px;
        }
    </style>
</head>
<body>
    <div class="status">🧠 RECALLSPECTION — Offline. Exact. Tamper‑Evident.</div>
    <div id="chat"></div>
    <div id="input-area">
        <input id="msg" type="text" placeholder="Type your message..." autofocus>
        <button id="send">Send</button>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const msg = document.getElementById('msg');
        const sendBtn = document.getElementById('send');

        function addMessage(text, type) {
            const div = document.createElement('div');
            div.className = type;
            const ts = new Date().toLocaleTimeString();
            div.innerHTML = text + ' <span class="timestamp">' + ts + '</span>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function sendMessage() {
            const text = msg.value.trim();
            if (!text) return;
            addMessage(text, 'user');
            msg.value = '';

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                addMessage(data.response, 'agent');
            } catch (e) {
                addMessage('Error: ' + e.message, 'agent');
            }
        }

        msg.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        sendBtn.addEventListener('click', sendMessage);

        // Initial greeting
        addMessage("Hello! I'm an exact memory agent. Tell me something about yourself.", 'agent');
        msg.focus();
    </script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# 7. RUN (for local testing)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)