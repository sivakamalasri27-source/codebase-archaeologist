from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=15000,  # 15 seconds, in milliseconds
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]

for model in CANDIDATES:
    print("Trying", model, "...", flush=True)
    try:
        r = client.models.generate_content(model=model, contents="Say hi in Tamil")
        print("WORKS:", model, "->", repr(r.text), flush=True)
        break
    except Exception as e:
        print("FAILED:", model, "->", type(e).__name__, str(e)[:120], flush=True)