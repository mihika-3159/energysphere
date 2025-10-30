import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # ensure .env variables are loaded even if imported

MODEL_PATH = Path("models/transformer_forecast.pt")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "models")

def download_model():
    if MODEL_PATH.exists():
        print("Model already exists locally. Skipping download.")
        return
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠️ Supabase credentials missing. Check your .env file.")
        return

    url = f"{SUPABASE_URL}/storage/v1/object/{MODEL_BUCKET}/transformer_forecast.pt"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    print(f"Downloading model from: {url}")
    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(r.content)
        print("✅ Model downloaded successfully and saved to models/transformer_forecast.pt")
    else:
        print(f"❌ Failed to download model: {r.status_code} - {r.text}")
